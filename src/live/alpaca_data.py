"""Fetch recent daily bars from Alpaca's market-data API.

Used for live signal generation instead of Yahoo Finance: Yahoo frequently blocks
or rate-limits cloud IPs (so yfinance works locally but fails in CI), whereas Alpaca
data uses the same keys we already authenticate the broker with. Free accounts get
the IEX feed, which is plenty for daily bars.
"""

from __future__ import annotations

import pandas as pd

from .config import AlpacaConfig

OHLCV = ["open", "high", "low", "close", "volume"]


def fetch_recent_bars(symbols: list[str], start: str, config: AlpacaConfig | None = None) -> dict:
    """Return {symbol: OHLCV DataFrame} of daily bars since ``start`` (YYYY-MM-DD)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    config = config or AlpacaConfig.from_env()
    client = StockHistoricalDataClient(config.api_key, config.secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start).to_pydatetime(),
        feed=DataFeed.IEX,  # free-tier feed
    )
    raw = client.get_stock_bars(request).df
    if raw.empty:
        raise RuntimeError(f"Alpaca returned no bars for {symbols} since {start}")

    frames = {}
    available = set(raw.index.get_level_values(0))
    for sym in symbols:
        if sym not in available:
            continue
        sub = raw.xs(sym, level=0)[OHLCV].copy()
        # Alpaca timestamps are tz-aware UTC; normalize to naive daily index.
        sub.index = pd.DatetimeIndex(sub.index).tz_convert(None).normalize()
        sub.index.name = "date"
        frames[sym] = sub
    if not frames:
        raise RuntimeError(f"None of {symbols} had Alpaca data since {start}")
    return frames

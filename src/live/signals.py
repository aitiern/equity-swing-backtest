"""Compute today's desired holdings from a backtest strategy.

Runs the SAME strategy code through the SAME engine on recent data, then reads the
strategy's final positions. This guarantees the live signal is identical to what the
backtest would do — no separate, drift-prone "live" implementation.

Data source: Alpaca when keys are available (reliable on CI, and the same account we
trade through), falling back to Yahoo Finance otherwise (e.g. a keyless --dry-run).
"""

from __future__ import annotations

from collections import deque

from ..data import DataHandler
from ..engine import Backtest
from ..execution import SimulatedExecutionHandler
from ..portfolio import Portfolio
from ..strategy import STRATEGIES


def fetch_frames(symbols: list[str], start: str = "2021-01-01") -> dict:
    """Recent OHLCV per symbol. Prefer Alpaca; fall back to yfinance if no keys/Alpaca
    error (so a keyless dry-run still works locally)."""
    try:
        from .alpaca_data import fetch_recent_bars
        from .config import AlpacaConfig

        return fetch_recent_bars(symbols, start, AlpacaConfig.from_env())
    except Exception as exc:
        from ..ml.execute import load_frames

        print(f"[data] Alpaca unavailable ({exc}); using yfinance.")
        return load_frames(symbols, start, None)


def desired_holdings(symbols: list[str], strategy_name: str, frames: dict):
    """Return (holdings, latest_prices): the symbols the strategy wants to be long
    right now, plus each symbol's most recent close. ``frames`` is caller-supplied
    OHLCV (from :func:`fetch_frames`) so the data source stays decoupled from logic."""
    if strategy_name not in STRATEGIES:
        raise KeyError(f"unknown strategy '{strategy_name}'. Known: {', '.join(STRATEGIES)}")
    events: deque = deque()
    data = DataHandler.from_frames(frames, events)
    strategy = STRATEGIES[strategy_name](data, events)
    portfolio = Portfolio(data, events)
    execution = SimulatedExecutionHandler(data, events)
    result = Backtest(data, strategy, portfolio, execution, events).run()

    holdings = {s for s, q in result.positions.items() if q > 0}
    prices = {s: data.latest_close(s) for s in data.symbols}
    return holdings, prices

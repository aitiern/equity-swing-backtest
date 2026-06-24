"""Market data handling.

The DataHandler streams bars one timestamp at a time so the strategy can only ever
see the past — this is what prevents lookahead bias, the most common way a backtest
lies to you. ``get_latest_bars`` exposes a trailing window; there is no way to peek
at future bars by construction.
"""

from __future__ import annotations

import os
from collections import deque

import pandas as pd

from .events import MarketEvent

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataHandler:
    """Streams historic OHLCV bars for one or more symbols.

    Construct it from CSVs (:meth:`from_csv_dir`) or from in-memory DataFrames
    (:meth:`from_frames`). The synthetic :meth:`demo` source lets the whole pipeline
    run with no external data so a reviewer can clone and execute immediately.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], events: deque):
        self.events = events
        self.symbols = list(frames)
        self._frames = {s: self._validate(df, s) for s, df in frames.items()}

        # Master timeline = union of all symbol indexes, sorted.
        index = sorted(set().union(*[df.index for df in self._frames.values()]))
        self._timeline: list[pd.Timestamp] = index
        self._cursor = -1

        # Trailing window of bars actually "seen" so far, per symbol.
        self._seen: dict[str, deque] = {s: deque(maxlen=2048) for s in self.symbols}
        self.continue_backtest = True
        self.current_time: pd.Timestamp | None = None

    # ----- construction helpers -------------------------------------------------
    @staticmethod
    def _validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{symbol}: missing required columns {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: index must be a DatetimeIndex (parse dates on load)")
        return df.sort_index()

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], events: deque) -> DataHandler:
        return cls(frames, events)

    @classmethod
    def from_csv_dir(cls, directory: str, symbols: list[str], events: deque) -> DataHandler:
        """Load ``<directory>/<symbol>.csv`` files with a parseable date column."""
        frames = {}
        for sym in symbols:
            path = os.path.join(directory, f"{sym}.csv")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            frames[sym] = df
        return cls(frames, events)

    @classmethod
    def from_yfinance(
        cls,
        symbols: list[str],
        start: str,
        end: str | None,
        events: deque,
        interval: str = "1d",
        cache_dir: str | None = "data",
    ) -> DataHandler:
        """Download real OHLCV bars from Yahoo Finance (auto-adjusted for splits/divs).

        If ``cache_dir`` is set, each symbol is saved as ``<cache_dir>/<symbol>.csv`` so
        re-runs are reproducible and offline-friendly — and so a result can be tied to
        the exact data it came from."""
        import yfinance as yf

        frames = {}
        for sym in symbols:
            raw = yf.download(
                sym, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned no data for {sym} ({start}..{end})")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)  # flatten (Price, Ticker)
            raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            raw.index.name = "date"
            frames[sym] = raw
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                raw.to_csv(os.path.join(cache_dir, f"{sym}.csv"))
        return cls(frames, events)

    @classmethod
    def demo(cls, events: deque, symbol: str = "DEMO", n: int = 750, seed: int = 7) -> DataHandler:
        """Deterministic synthetic random-walk series. DEMO DATA ONLY — never present
        results computed on this as real strategy performance."""
        import numpy as np

        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-01", periods=n)
        # Geometric random walk with a faint drift so charts look plausible.
        rets = rng.normal(0.0003, 0.012, n)
        close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
        high = close * (1 + rng.uniform(0, 0.01, n))
        low = close * (1 - rng.uniform(0, 0.01, n))
        open_ = close.shift(1).fillna(close.iloc[0])
        vol = rng.integers(1_000_000, 5_000_000, n)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=dates,
        )
        return cls({symbol: df}, events)

    # ----- streaming API --------------------------------------------------------
    def update_bars(self) -> None:
        """Advance one timestamp and emit a MarketEvent. Call once per outer loop."""
        self._cursor += 1
        if self._cursor >= len(self._timeline):
            self.continue_backtest = False
            return
        ts = self._timeline[self._cursor]
        self.current_time = ts
        for sym, df in self._frames.items():
            if ts in df.index:
                bar = df.loc[ts]
                self._seen[sym].append((ts, bar))
        self.events.append(MarketEvent())

    def get_latest_bars(self, symbol: str, n: int = 1) -> list:
        """Return up to the last ``n`` (timestamp, bar) pairs seen for ``symbol``."""
        window = self._seen.get(symbol)
        if not window:
            return []
        return list(window)[-n:]

    def latest_close(self, symbol: str) -> float | None:
        bars = self.get_latest_bars(symbol, 1)
        return float(bars[-1][1]["close"]) if bars else None

    # ----- reference data (whole-series; benchmark/reporting only) --------------
    def close_frame(self) -> pd.DataFrame:
        """Full close prices for every symbol, aligned on the master timeline.

        Uses the entire series — fine for a buy-and-hold benchmark, which is a
        reference, not a tradeable signal. Never call this from a Strategy: doing so
        would leak future prices and reintroduce the lookahead bias the streaming
        API exists to prevent."""
        return pd.DataFrame(
            {s: df["close"] for s, df in self._frames.items()},
            index=pd.DatetimeIndex(self._timeline),
        ).ffill()

    def benchmark_equity(self, initial_capital: float) -> pd.Series:
        """Equal-weight buy-and-hold of all symbols, normalised to start at capital."""
        closes = self.close_frame()
        if closes.empty:
            return pd.Series(dtype=float)
        norm = closes.div(closes.iloc[0])  # each symbol -> growth multiple from t0
        portfolio = norm.mean(axis=1)  # equal weight
        return portfolio * initial_capital

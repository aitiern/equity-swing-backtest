"""Pure helpers for the dashboard (kept out of streamlit_app.py so they can be
unit-tested without a Streamlit runtime)."""

from __future__ import annotations

import pandas as pd

STRATEGY_LABEL = "Strategy (paper)"
BENCHMARK_LABEL = "SPY (buy & hold)"


def normalize_benchmark(closes: pd.Series, start_equity: float) -> pd.Series:
    """Scale a price series so it starts at ``start_equity`` (buy-and-hold from day 1)."""
    closes = closes.dropna()
    if closes.empty:
        return closes
    return closes / closes.iloc[0] * start_equity


def combine_equity_and_benchmark(prog: pd.DataFrame, benchmark: pd.Series | None) -> pd.DataFrame:
    """Align the logged paper equity with an (already-normalized) benchmark on a daily
    index, forward-filling the sparse strategy log so both plot as continuous lines."""
    ts = pd.to_datetime(prog["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    strat = pd.Series(list(prog["equity"]), index=ts, name=STRATEGY_LABEL)
    strat = strat[~strat.index.duplicated(keep="last")]

    if benchmark is None or len(benchmark) == 0:
        return strat.to_frame()

    bench = benchmark.copy()
    idx = pd.DatetimeIndex(bench.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    bench.index = idx.normalize()
    bench.name = BENCHMARK_LABEL

    combined = pd.concat([strat, bench], axis=1).sort_index().ffill()
    return combined.dropna(how="all")

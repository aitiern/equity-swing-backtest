"""Tests for the dashboard's benchmark-alignment helpers (no Streamlit needed)."""

import pandas as pd

from src.live.dashboard_utils import (
    BENCHMARK_LABEL,
    STRATEGY_LABEL,
    combine_equity_and_benchmark,
    normalize_benchmark,
)


def test_normalize_benchmark_starts_at_equity():
    closes = pd.Series([400.0, 404.0, 408.0])
    norm = normalize_benchmark(closes, 100_000.0)
    assert norm.iloc[0] == 100_000.0
    assert round(norm.iloc[2], 2) == 102_000.0  # +2% -> 102k


def test_combine_aligns_ffills_and_labels():
    prog = pd.DataFrame({
        "timestamp": ["2024-01-02T21:30:00+00:00", "2024-01-04T21:30:00+00:00"],
        "equity": [100_000.0, 101_000.0],
    })
    bench = pd.Series(
        [100_000.0, 100_500.0, 101_000.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    combined = combine_equity_and_benchmark(prog, bench)
    assert list(combined.columns) == [STRATEGY_LABEL, BENCHMARK_LABEL]
    # Strategy log is sparse (no 1/3) -> forward-filled from 1/2.
    assert combined.loc["2024-01-03", STRATEGY_LABEL] == 100_000.0
    assert combined.loc["2024-01-04", BENCHMARK_LABEL] == 101_000.0


def test_combine_without_benchmark_returns_strategy_only():
    prog = pd.DataFrame({"timestamp": ["2024-01-02T21:30:00+00:00"], "equity": [100_000.0]})
    combined = combine_equity_and_benchmark(prog, None)
    assert list(combined.columns) == [STRATEGY_LABEL]

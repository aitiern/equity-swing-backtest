"""Tests for the metrics that go in the README results table.

Numbers a reviewer sees should be backed by tests. These check the metric math
against hand-computable cases and verify the portfolio accounting conserves cash.
"""

import numpy as np
import pandas as pd

from src import metrics
from src.portfolio import Portfolio
from src.events import FillEvent


def _series(values):
    idx = pd.bdate_range("2022-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_max_drawdown_simple():
    eq = _series([100, 120, 60, 90])  # peak 120 -> trough 60 = -50%
    assert metrics.max_drawdown(eq) == -0.5


def test_no_drawdown_when_monotonic():
    eq = _series([100, 101, 102, 103])
    assert metrics.max_drawdown(eq) == 0.0


def test_sharpe_zero_for_flat_curve():
    eq = _series([100, 100, 100, 100])
    assert metrics.sharpe_ratio(eq) == 0.0


def test_cagr_doubling_in_one_year():
    eq = _series(np.linspace(100, 200, metrics.TRADING_DAYS))
    assert abs(metrics.cagr(eq) - 1.0) < 0.05  # ~100% over one year


def test_win_rate():
    assert metrics.win_rate([10, -5, 20, -1]) == 0.5
    assert metrics.win_rate([]) == 0.0


def test_round_trip_pnl_accounting():
    """Buy 10 @100 then sell 10 @110 with no costs -> +100 realised, cash restored +100."""

    class _Data:
        symbols = ["X"]
        current_time = None

        def latest_close(self, _):
            return 110.0

    pf = Portfolio(_Data(), events=None, initial_capital=1_000.0)
    pf.update_fill(FillEvent("X", None, "BUY", 10, 100.0, 0.0))
    assert pf.positions["X"] == 10
    assert pf.cash == 0.0  # spent 1000

    pf.update_fill(FillEvent("X", None, "SELL", 10, 110.0, 0.0))
    assert pf.positions["X"] == 0
    assert pf.cash == 1_100.0  # got 1100 back
    assert pf.closed_trades == [100.0]

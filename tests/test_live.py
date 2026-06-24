"""Tests for the paper-trader's pure logic (sizing + order reconciliation).

These run with no Alpaca keys and no network — the network surface lives only in
broker.py, which is mocked out of these paths entirely.
"""

import numpy as np
import pandas as pd

from src.live.signals import desired_holdings
from src.live.trade import compute_target_shares, reconcile


def _synthetic_frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 * (1 + pd.Series(rng.normal(0.001, 0.01, n), index=dates)).cumprod()
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=dates)


def test_desired_holdings_runs_on_supplied_frames():
    # No network: frames are passed in directly, decoupled from the data source.
    frames = {"X": _synthetic_frame(seed=1), "Y": _synthetic_frame(seed=2)}
    holdings, prices = desired_holdings(["X", "Y"], "ma_cross", frames)
    assert holdings.issubset({"X", "Y"})
    assert prices["X"] > 0 and prices["Y"] > 0


def test_target_shares_equal_weight_only_for_holdings():
    symbols = ["A", "B", "C", "D"]
    prices = dict.fromkeys(symbols, 100.0)
    holdings = {"A", "B"}
    # equity 100k, 4 names, 0.95 -> $23,750 each -> 237 shares at $100.
    targets = compute_target_shares(100_000, holdings, symbols, prices, target_pct=0.95)
    assert targets["A"] == 237
    assert targets["B"] == 237
    assert targets["C"] == 0  # not a desired holding
    assert targets["D"] == 0


def test_target_shares_handles_missing_price():
    targets = compute_target_shares(100_000, {"A"}, ["A"], {"A": None})
    assert targets["A"] == 0  # no price -> no position


def test_reconcile_buys_sells_and_flattens():
    targets = {"A": 10, "B": 0}
    current = {"A": 4, "C": 5}  # hold 4 of A, want 10; hold 5 of C, not in universe
    orders = reconcile(targets, current)
    assert ("A", 6, "BUY") in orders       # top up A
    assert ("C", 5, "SELL") in orders       # flatten the orphan position
    assert all(sym != "B" for sym, _, _ in orders)  # already flat, no order


def test_reconcile_noop_when_aligned():
    assert reconcile({"A": 10}, {"A": 10}) == []

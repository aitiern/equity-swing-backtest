"""Tests for the ML signal-execution helpers (thresholding, top-k, precision@k)."""

import numpy as np
import pandas as pd

from src.ml.execute import precision_at_k, signals_from_proba, topk_signals


def _proba():
    idx = pd.bdate_range("2022-01-01", periods=4)
    return {
        "A": pd.Series([0.9, 0.1, np.nan, 0.8], index=idx),
        "B": pd.Series([0.2, 0.7, 0.6, 0.3], index=idx),
    }


def test_signals_from_proba_threshold_and_nan():
    sig = signals_from_proba(_proba(), threshold=0.5)
    assert list(sig["A"]) == [1, 0, 0, 1]  # NaN -> flat
    assert list(sig["B"]) == [0, 1, 1, 0]


def test_topk_selects_highest_each_rebalance():
    sig = topk_signals(_proba(), k=1, rebalance=1)
    # Day 0: A(0.9) > B(0.2) -> hold A. Day 1: B(0.7) > A(0.1) -> hold B.
    assert sig["A"].iloc[0] == 1 and sig["B"].iloc[0] == 0
    assert sig["B"].iloc[1] == 1 and sig["A"].iloc[1] == 0


def test_precision_at_k_matches_hand_count():
    idx = pd.bdate_range("2022-01-01", periods=2)
    proba = {"A": pd.Series([0.9, 0.4], index=idx), "B": pd.Series([0.3, 0.8], index=idx)}
    # Day 0 top pick A; Day 1 top pick B. Targets: A up on day0 (hit), B down on day1 (miss).
    targets = {"A": pd.Series([1.0, 0.0], index=idx), "B": pd.Series([0.0, 0.0], index=idx)}
    assert precision_at_k(proba, targets, k=1) == 0.5

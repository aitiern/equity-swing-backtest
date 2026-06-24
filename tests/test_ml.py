"""Tests for the ML signal pipeline.

The critical property is no lookahead: a feature value at day d must be identical
whether computed on the full series or on the series truncated at d. If that holds,
features cannot encode the future. We also smoke-test the walk-forward runner.
"""

import numpy as np
import pandas as pd

from src.ml.features import FEATURE_COLS, build_features
from src.ml.walkforward import walk_forward_signals


def _synthetic_ohlcv(n=900, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n)
    close = 100 * (1 + pd.Series(rng.normal(0.0004, 0.011, n), index=dates)).cumprod()
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def test_features_have_no_lookahead():
    df = _synthetic_ohlcv()
    full = build_features(df)
    # Pick a date well into the series; recompute features using only data up to it.
    cutoff = full.index[600]
    truncated = build_features(df.loc[:cutoff])
    for col in FEATURE_COLS:
        assert np.isclose(full.loc[cutoff, col], truncated.loc[cutoff, col]), col


def test_target_is_next_day_direction():
    df = _synthetic_ohlcv(n=500)
    feats = build_features(df)
    # Pick a row with a known target that is not the last row.
    sample = feats.index[50]
    assert not np.isnan(feats.loc[sample, "target"])
    nxt = df.index[df.index.get_loc(sample) + 1]
    assert (df.loc[nxt, "close"] > df.loc[sample, "close"]) == bool(feats.loc[sample, "target"])


def test_walk_forward_signals_are_binary_and_out_of_sample():
    frames = {"A": _synthetic_ohlcv(seed=1), "B": _synthetic_ohlcv(seed=2)}
    wf = walk_forward_signals(frames, min_train=300, step=60)
    assert wf.folds > 0
    assert wf.first_prediction is not None
    for _sym, sig in wf.signals.items():
        assert set(sig.unique()).issubset({0, 1})
        # No signal should fire before the first prediction date (everything past is OOS).
        assert (sig.loc[: wf.first_prediction].iloc[:-1] == 0).all()
    assert 0.0 <= wf.accuracy <= 1.0

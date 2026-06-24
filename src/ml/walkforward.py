"""Walk-forward signal generation for a sector basket.

The model is retrained repeatedly on a rolling/expanding window of PAST data and
used to predict the next out-of-sample slice, stepping forward through time. No
future row ever enters a training set — this is the only honest way to backtest an
ML strategy, and the reason every prediction here is genuinely out-of-sample.

One model per sector: all symbols in the basket are pooled into a single training
set (a cross-sectional model), then used to score each symbol independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLS, build_features


@dataclass
class WalkForwardResult:
    signals: dict[str, pd.Series]  # symbol -> 0/1 position per date (long when 1)
    first_prediction: pd.Timestamp | None
    n_predictions: int
    accuracy: float  # directional accuracy over all OOS predictions
    base_rate: float  # share of up-days (the naive "always long" hit rate)
    folds: int
    proba: dict[str, pd.Series] = field(default_factory=dict)  # symbol -> P(up)


def _make_model(model_params: dict | None) -> HistGradientBoostingClassifier:
    params = {
        "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15,
        "l2_regularization": 1.0, "random_state": 0,
    }
    if model_params:
        params.update(model_params)
    return HistGradientBoostingClassifier(**params)


def walk_forward_signals(
    frames: dict[str, pd.DataFrame],
    min_train: int = 504,   # ~2 trading years before the first prediction
    step: int = 63,         # retrain ~quarterly
    threshold: float = 0.5,
    lookback: int | None = None,  # None = expanding window; int = rolling window (days)
    model_params: dict | None = None,
) -> WalkForwardResult:
    # Per-symbol features, then a pooled long-form table (date index + symbol column).
    feats = {sym: build_features(df) for sym, df in frames.items()}
    pooled = []
    for sym, f in feats.items():
        g = f.copy()
        g["symbol"] = sym
        pooled.append(g)
    pool = pd.concat(pooled).sort_index()

    dates = np.array(sorted(pool.index.unique()))
    signals = {sym: pd.Series(0, index=feats[sym].index, dtype=int) for sym in frames}
    proba = {sym: pd.Series(np.nan, index=feats[sym].index, dtype=float) for sym in frames}

    correct = total = folds = 0
    first_prediction = None

    i = min_train
    while i < len(dates):
        boundary = dates[i]
        next_i = min(i + step, len(dates))
        # 1-day embargo: train only on rows strictly before the day before `boundary`,
        # so no training label can peek into the prediction window.
        embargo = dates[i - 1]
        train = pool[(pool.index < embargo) & pool["target"].notna()]
        if lookback is not None:
            train = train[train.index >= embargo - pd.Timedelta(days=lookback)]

        block_end = dates[next_i] if next_i < len(dates) else dates[-1] + pd.Timedelta(days=1)
        block = pool[(pool.index >= boundary) & (pool.index < block_end)]

        if len(train) >= 100 and not block.empty and train["target"].nunique() > 1:
            model = _make_model(model_params)
            model.fit(train[FEATURE_COLS], train["target"].astype(int))
            up_col = list(model.classes_).index(1)
            p_up = model.predict_proba(block[FEATURE_COLS])[:, up_col]

            for (ts, row), p in zip(block.iterrows(), p_up, strict=True):
                sym = row["symbol"]
                proba[sym].at[ts] = p
                position = int(p > threshold)
                signals[sym].at[ts] = position
                if not np.isnan(row["target"]):  # score only where outcome is known
                    total += 1
                    correct += int(position == int(row["target"]))
            folds += 1
            if first_prediction is None:
                first_prediction = boundary

        i = next_i

    base_rate = float(pool["target"].dropna().mean())
    accuracy = correct / total if total else 0.0
    return WalkForwardResult(
        signals=signals,
        first_prediction=first_prediction,
        n_predictions=total,
        accuracy=accuracy,
        base_rate=base_rate,
        folds=folds,
        proba=proba,
    )

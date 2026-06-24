"""Feature engineering for the next-day-direction classifier.

EVERY feature is backward-looking — built from rolling windows that only see past
bars — so feature rows carry no future information. The single forward-looking
column is ``target`` (did tomorrow close up?), which is used ONLY as the training
label and never as a model input. This split is what keeps the pipeline honest.
"""

from __future__ import annotations

import pandas as pd

# The columns fed to the model. `target` is deliberately NOT in here.
FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_21",
    "mom_63", "rsi_14", "vol_21",
    "dist_sma_50", "dist_sma_200", "vol_ratio",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, vectorised (EWM with alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of FEATURE_COLS + ``target`` aligned to ``df``'s dates.

    ``df`` must have lowercase open/high/low/close/volume columns and a DatetimeIndex.
    Rows where any feature is undefined (early history) are dropped; the final row's
    ``target`` is NaN (no next day) and is dropped only when training, not predicting.
    """
    close = df["close"]
    ret_1 = close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = ret_1
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_21"] = close.pct_change(21)
    out["mom_63"] = close.pct_change(63)
    out["rsi_14"] = _rsi(close, 14)
    out["vol_21"] = ret_1.rolling(21).std()
    out["dist_sma_50"] = close / close.rolling(50).mean() - 1
    out["dist_sma_200"] = close / close.rolling(200).mean() - 1
    out["vol_ratio"] = df["volume"] / df["volume"].rolling(21).mean()

    # Target: 1 if the NEXT day closes higher than today. Forward-looking by design;
    # used only as the label. shift(-1) means the last row's target is NaN.
    out["target"] = (close.shift(-1) > close).astype("float")
    out.loc[close.shift(-1).isna(), "target"] = float("nan")

    # Drop rows with any undefined feature (keeps target NaN on the last row intact).
    return out.dropna(subset=FEATURE_COLS)

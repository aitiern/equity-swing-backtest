"""Equity-curve chart. A chart at the top of the README is what makes a screener
keep reading, so this saves a clean PNG to docs/equity-curve.png by default."""

from __future__ import annotations

import os

import pandas as pd


def save_equity_curve(
    equity: pd.Series,
    path: str = "docs/equity-curve.png",
    benchmark: pd.Series | None = None,
    title: str = "Strategy equity curve",
    oos_start: pd.Timestamp | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless: works in CI / over SSH with no display
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_eq.plot(equity.index, equity.values, label="Strategy", linewidth=1.4)
    if benchmark is not None and not benchmark.empty:
        scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax_eq.plot(scaled.index, scaled.values, label="Benchmark", linewidth=1.0, alpha=0.7)
    if oos_start is not None:
        # Shade the held-out period so the in-sample/out-of-sample boundary is obvious.
        ax_eq.axvline(oos_start, color="gray", linestyle="--", linewidth=1.0)
        ax_eq.axvspan(oos_start, equity.index[-1], color="gray", alpha=0.07, label="Out-of-sample")
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    drawdown = equity / equity.cummax() - 1.0
    ax_dd.fill_between(drawdown.index, drawdown.values * 100, 0, color="crimson", alpha=0.4)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

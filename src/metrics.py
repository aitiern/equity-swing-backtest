"""Performance metrics computed from the equity curve.

These are the numbers that go in the README results table. They are deliberately
plain and auditable — no magic, so you can defend every figure in an interview.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualised_volatility(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return daily_returns(equity).std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    rets = daily_returns(equity)
    if rets.std(ddof=1) == 0 or rets.empty:
        return 0.0
    excess = rets - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / rets.std(ddof=1)


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    rets = daily_returns(equity)
    downside = rets[rets < 0]
    if downside.std(ddof=1) == 0 or downside.empty:
        return 0.0
    excess = rets.mean() - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess / downside.std(ddof=1)


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline, as a negative fraction (e.g. -0.23)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    return cagr(equity, periods_per_year) / mdd if mdd > 0 else 0.0


def win_rate(closed_trades: List[float]) -> float:
    if not closed_trades:
        return 0.0
    wins = sum(1 for p in closed_trades if p > 0)
    return wins / len(closed_trades)


def summary(equity: pd.Series, closed_trades: List[float], periods_per_year: int = TRADING_DAYS) -> Dict[str, float]:
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
        "cagr": cagr(equity, periods_per_year),
        "ann_volatility": annualised_volatility(equity, periods_per_year),
        "sharpe": sharpe_ratio(equity, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(equity, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "num_trades": len(closed_trades),
        "win_rate": win_rate(closed_trades),
    }


def format_summary(stats: Dict[str, float]) -> str:
    pct = {"total_return", "cagr", "ann_volatility", "max_drawdown", "win_rate"}
    lines = ["Performance summary", "-" * 32]
    for k, v in stats.items():
        if k in pct:
            lines.append(f"{k:>16}: {v * 100:8.2f}%")
        elif k == "num_trades":
            lines.append(f"{k:>16}: {int(v):8d}")
        else:
            lines.append(f"{k:>16}: {v:8.2f}")
    return "\n".join(lines)

"""Shared helpers to turn ML probabilities into positions and run them through the
engine. Centralised so every ML runner (single, sweep, cross-sectional) executes
signals identically — same costs, same sizing, same accounting.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from ..data import DataHandler
from ..engine import Backtest
from ..execution import SimulatedExecutionHandler
from ..portfolio import Portfolio
from ..strategy import MLSignalStrategy


def load_frames(symbols: list[str], start: str, end: str | None, cache_dir: str = "data"):
    """Download (and cache) OHLCV, then read it back as raw per-symbol frames."""
    DataHandler.from_yfinance(symbols, start, end, deque(), cache_dir=cache_dir)
    return {
        sym: pd.read_csv(f"{cache_dir}/{sym}.csv", index_col=0, parse_dates=True)
        for sym in symbols
    }


def signals_from_proba(proba: dict[str, pd.Series], threshold: float) -> dict[str, pd.Series]:
    """Long when P > threshold, flat otherwise. Undefined (pre-prediction) -> flat."""
    out = {}
    for sym, p in proba.items():
        s = (p > threshold).astype("float")
        s[p.isna()] = 0.0
        out[sym] = s.astype(int)
    return out


def topk_signals(proba: dict[str, pd.Series], k: int, rebalance: int = 5) -> dict[str, pd.Series]:
    """Hold the k highest-probability names, refreshed every ``rebalance`` bars.

    Models a long-only cross-sectional selection book: own the names the model ranks
    highest within the sector, rebalance periodically to keep turnover (and costs)
    in check."""
    matrix = pd.DataFrame(proba).dropna(how="all")
    holdings = pd.DataFrame(0, index=matrix.index, columns=matrix.columns)
    current: list[str] = []
    for i, date in enumerate(matrix.index):
        if i % rebalance == 0:
            row = matrix.loc[date].dropna()
            if len(row):
                current = list(row.sort_values(ascending=False).head(k).index)
        if current:
            holdings.loc[date, current] = 1
    return {sym: holdings[sym].astype(int) for sym in matrix.columns}


def precision_at_k(proba: dict[str, pd.Series], targets: dict[str, pd.Series], k: int) -> float:
    """Of the k names the model ranks highest each day, what fraction actually had a
    positive label (i.e. outperformed)? This is the honest selection metric — ranking
    skill, not 0.5-threshold classification accuracy. Compare it to the base rate k/N."""
    pmatrix = pd.DataFrame(proba)
    tmatrix = pd.DataFrame(targets)
    hits = picks = 0
    for date in pmatrix.index:
        row = pmatrix.loc[date].dropna()
        if len(row) < k:
            continue
        top = row.sort_values(ascending=False).head(k).index
        outcomes = tmatrix.loc[date, top].dropna()
        hits += int(outcomes.sum())
        picks += len(outcomes)
    return hits / picks if picks else 0.0


def backtest_signals(
    symbols: list[str],
    signals: dict[str, pd.Series],
    capital: float = 100_000.0,
    target_pct: float = 0.95,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
    cache_dir: str = "data",
):
    """Execute precomputed 0/1 signals through the real engine. Returns (result, data)."""
    events: deque = deque()
    data = DataHandler.from_csv_dir(cache_dir, symbols, events)
    strategy = MLSignalStrategy(data, events, signals)
    portfolio = Portfolio(data, events, initial_capital=capital, target_pct=target_pct)
    execution = SimulatedExecutionHandler(
        data, events, slippage_bps=slippage_bps, commission_bps=commission_bps
    )
    result = Backtest(data, strategy, portfolio, execution, events).run()
    return result, data

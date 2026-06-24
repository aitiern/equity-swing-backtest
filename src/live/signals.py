"""Compute today's desired holdings from a backtest strategy.

Runs the SAME strategy code through the SAME engine on recent data, then reads the
strategy's final positions. This guarantees the live signal is identical to what the
backtest would do — no separate, drift-prone "live" implementation.
"""

from __future__ import annotations

from collections import deque

from ..data import DataHandler
from ..engine import Backtest
from ..execution import SimulatedExecutionHandler
from ..ml.execute import load_frames
from ..portfolio import Portfolio
from ..strategy import STRATEGIES


def desired_holdings(symbols: list[str], strategy_name: str,
                     start: str = "2021-01-01", end: str | None = None):
    """Return (holdings, latest_prices): the set of symbols the strategy wants to be
    long right now, plus each symbol's most recent close."""
    if strategy_name not in STRATEGIES:
        raise KeyError(f"unknown strategy '{strategy_name}'. Known: {', '.join(STRATEGIES)}")
    load_frames(symbols, start, end)  # download + cache
    events: deque = deque()
    data = DataHandler.from_csv_dir("data", symbols, events)
    strategy = STRATEGIES[strategy_name](data, events)
    portfolio = Portfolio(data, events)
    execution = SimulatedExecutionHandler(data, events)
    result = Backtest(data, strategy, portfolio, execution, events).run()

    holdings = {s for s, q in result.positions.items() if q > 0}
    prices = {s: data.latest_close(s) for s in symbols}
    return holdings, prices

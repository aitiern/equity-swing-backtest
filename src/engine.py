"""The event loop that ties everything together.

Outer loop advances market data one bar at a time. The inner loop drains the event
queue, dispatching each event to the component that handles it. Processing the queue
to empty before the next bar guarantees events resolve in causal order
(Market -> Signal -> Order -> Fill) within a single timestamp.
"""

from __future__ import annotations

from collections import deque

from .data import DataHandler
from .execution import SimulatedExecutionHandler
from .portfolio import Portfolio
from .strategy import Strategy


class Backtest:
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: SimulatedExecutionHandler,
        events: deque,
    ):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events = events

    def run(self) -> Portfolio:
        while True:
            self.data.update_bars()
            if not self.data.continue_backtest:
                break

            while self.events:
                event = self.events.popleft()
                if event.type == "MARKET":
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                elif event.type == "SIGNAL":
                    self.portfolio.update_signal(event)
                elif event.type == "ORDER":
                    self.execution.execute_order(event)
                elif event.type == "FILL":
                    self.portfolio.update_fill(event)

        return self.portfolio

"""Strategy layer.

THIS is the only file you must edit to make the repo yours. Subclass ``Strategy``
and implement ``calculate_signals``: read the trailing bars, decide LONG/SHORT/EXIT,
and append SignalEvents. The strategy never sees cash, position size, or PnL — that
separation keeps the alpha logic honest and independently testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Deque

from .data import DataHandler
from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    def __init__(self, data: DataHandler, events: Deque):
        self.data = data
        self.events = events

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """Inspect the latest bars and append SignalEvents to ``self.events``."""
        raise NotImplementedError


class MovingAverageCrossStrategy(Strategy):
    """=========================================================================
    PLACEHOLDER DEMO STRATEGY — REPLACE WITH YOUR REAL SIGNAL BEFORE PUBLISHING.

    A textbook fast/slow SMA crossover. It exists only so the pipeline runs end to
    end out of the box. Publishing this as "your strategy" would be dishonest and a
    reviewer will recognise it instantly. Swap the body of ``calculate_signals`` for
    the actual logic you trade, then delete this warning.
    ========================================================================="""

    def __init__(self, data: DataHandler, events: Deque, short: int = 20, long: int = 50):
        super().__init__(data, events)
        if short >= long:
            raise ValueError("short window must be smaller than long window")
        self.short = short
        self.long = long
        self._invested = {s: False for s in data.symbols}

    def calculate_signals(self, event: MarketEvent) -> None:
        for symbol in self.data.symbols:
            bars = self.data.get_latest_bars(symbol, self.long)
            if len(bars) < self.long:
                continue  # not enough history yet
            closes = [b[1]["close"] for b in bars]
            ts = bars[-1][0]
            fast = sum(closes[-self.short:]) / self.short
            slow = sum(closes) / self.long

            if fast > slow and not self._invested[symbol]:
                self.events.append(SignalEvent(symbol, ts, "LONG"))
                self._invested[symbol] = True
            elif fast < slow and self._invested[symbol]:
                self.events.append(SignalEvent(symbol, ts, "EXIT"))
                self._invested[symbol] = False

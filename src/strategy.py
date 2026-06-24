"""Strategy library.

To add a strategy, subclass ``Strategy`` and implement ``calculate_signals``: read
the trailing bars, decide LONG/SHORT/EXIT, and append SignalEvents. The strategy
never sees cash, position size, or PnL — that separation keeps the alpha logic
honest and independently testable. Register the class in ``STRATEGIES`` (bottom of
file) to make it selectable via ``--strategy`` on the CLI.

These are simple, well-known rule sets meant as honest baselines. A strategy is only
worth anything if it survives real costs AND holds up out-of-sample — always check
both before believing a result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from .data import DataHandler
from .events import MarketEvent, SignalEvent
from .indicators import bollinger, highest, lowest, rsi


class Strategy(ABC):
    def __init__(self, data: DataHandler, events: deque):
        self.data = data
        self.events = events

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """Inspect the latest bars and append SignalEvents to ``self.events``."""
        raise NotImplementedError

    # Small helpers shared by the concrete strategies below.
    def _closes(self, symbol: str, n: int) -> list[float]:
        return [b[1]["close"] for b in self.data.get_latest_bars(symbol, n)]

    def _series(self, symbol: str, field: str, n: int) -> list[float]:
        return [b[1][field] for b in self.data.get_latest_bars(symbol, n)]

    def _last_ts(self, symbol: str):
        bars = self.data.get_latest_bars(symbol, 1)
        return bars[-1][0] if bars else None


class MovingAverageCrossStrategy(Strategy):
    """Trend following: go long when the fast SMA is above the slow SMA, exit when it
    crosses back below. A simple, well-understood baseline — not an edge on its own."""

    def __init__(self, data: DataHandler, events: deque, short: int = 20, long: int = 50):
        super().__init__(data, events)
        if short >= long:
            raise ValueError("short window must be smaller than long window")
        self.short = short
        self.long = long
        self._invested = dict.fromkeys(data.symbols, False)

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


class RSIMeanReversion(Strategy):
    """Mean reversion: buy when RSI is oversold (a dip), exit once it recovers.
    Bets that short-term sell-offs bounce. Tends to win often but small, and lose
    big in sustained downtrends — watch the drawdown, not just the win rate."""

    def __init__(self, data, events, period: int = 14, oversold: float = 30.0, exit_level: float = 55.0):
        super().__init__(data, events)
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self._invested = dict.fromkeys(data.symbols, False)

    def calculate_signals(self, event: MarketEvent) -> None:
        for symbol in self.data.symbols:
            closes = self._closes(symbol, self.period + 1)
            value = rsi(closes, self.period)
            if value is None:
                continue
            ts = self._last_ts(symbol)
            if not self._invested[symbol] and value < self.oversold:
                self.events.append(SignalEvent(symbol, ts, "LONG"))
                self._invested[symbol] = True
            elif self._invested[symbol] and value > self.exit_level:
                self.events.append(SignalEvent(symbol, ts, "EXIT"))
                self._invested[symbol] = False


class DonchianBreakout(Strategy):
    """Momentum: buy an N-day high (breakout), exit on an M-day low. Rides large
    trends, pays for it with many small whipsaw losses — the opposite shape of the
    mean-reversion strategies. The channel uses only prior bars, never today's."""

    def __init__(self, data, events, entry: int = 20, exit: int = 10):
        super().__init__(data, events)
        self.entry = entry
        self.exit = exit
        self._invested = dict.fromkeys(data.symbols, False)

    def calculate_signals(self, event: MarketEvent) -> None:
        need = max(self.entry, self.exit) + 1
        for symbol in self.data.symbols:
            highs = self._series(symbol, "high", need)
            lows = self._series(symbol, "low", need)
            closes = self._closes(symbol, need)
            if len(closes) < need:
                continue
            close = closes[-1]
            ts = self._last_ts(symbol)
            # Channel from the bars BEFORE the current one (exclude today's bar).
            upper = highest(highs[:-1], self.entry)
            lower = lowest(lows[:-1], self.exit)
            if not self._invested[symbol] and upper is not None and close > upper:
                self.events.append(SignalEvent(symbol, ts, "LONG"))
                self._invested[symbol] = True
            elif self._invested[symbol] and lower is not None and close < lower:
                self.events.append(SignalEvent(symbol, ts, "EXIT"))
                self._invested[symbol] = False


class BollingerReversion(Strategy):
    """Volatility-adaptive mean reversion: buy when price closes below the lower
    band (k std-devs under the moving average), exit when it reverts to the mean."""

    def __init__(self, data, events, period: int = 20, k: float = 2.0):
        super().__init__(data, events)
        self.period = period
        self.k = k
        self._invested = dict.fromkeys(data.symbols, False)

    def calculate_signals(self, event: MarketEvent) -> None:
        for symbol in self.data.symbols:
            closes = self._closes(symbol, self.period)
            bands = bollinger(closes, self.period, self.k)
            if bands is None:
                continue
            lower, mid, _upper = bands
            close = closes[-1]
            ts = self._last_ts(symbol)
            if not self._invested[symbol] and close < lower:
                self.events.append(SignalEvent(symbol, ts, "LONG"))
                self._invested[symbol] = True
            elif self._invested[symbol] and close >= mid:
                self.events.append(SignalEvent(symbol, ts, "EXIT"))
                self._invested[symbol] = False


# Registry: maps --strategy names to classes. Add new strategies here.
STRATEGIES = {
    "ma_cross": MovingAverageCrossStrategy,
    "rsi": RSIMeanReversion,
    "donchian": DonchianBreakout,
    "bollinger": BollingerReversion,
}

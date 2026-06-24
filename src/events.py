"""Event objects that flow through the backtest queue.

The simulation is event-driven: each bar of market data emits a chain of events
(Market -> Signal -> Order -> Fill) that are processed one at a time. Keeping the
events as small immutable records makes the control flow easy to follow and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class Event:
    """Base class so the engine can type-switch on ``event.type``."""

    type: str


@dataclass
class MarketEvent(Event):
    """A new bar is available. Triggers the strategy and a portfolio mark-to-market."""

    type: str = "MARKET"


@dataclass
class SignalEvent(Event):
    """A directional view produced by the strategy. Carries no sizing — that is the
    portfolio's job, which keeps strategy logic independent of capital allocation."""

    symbol: str
    timestamp: datetime
    direction: str  # "LONG", "SHORT", or "EXIT"
    strength: float = 1.0  # optional conviction in [0, 1]; sizing may use it
    type: str = "SIGNAL"


@dataclass
class OrderEvent(Event):
    """A sized order the portfolio wants executed."""

    symbol: str
    timestamp: datetime
    direction: str  # "BUY" or "SELL"
    quantity: int
    type: str = "ORDER"

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("OrderEvent.quantity must be non-negative; use direction for side")


@dataclass
class FillEvent(Event):
    """The result of an order hitting the (simulated) market: actual price + costs."""

    symbol: str
    timestamp: datetime
    direction: str  # "BUY" or "SELL"
    quantity: int
    fill_price: float  # price actually paid/received, slippage included
    commission: float
    type: str = "FILL"

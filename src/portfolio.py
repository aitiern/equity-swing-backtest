"""Portfolio: position sizing, accounting, and the equity curve.

Responsibilities:
  * turn a SignalEvent into a sized OrderEvent (risk/capital lives here, not in the strategy),
  * apply FillEvents to cash and positions, tracking realised PnL per round-trip trade,
  * mark the book to market on every bar and record the equity curve the metrics use.

Sizing here is fixed-fractional: each entry targets ``target_pct`` of current equity.
Swap ``_size_order`` for vol-targeting or fixed-risk sizing when you are ready.
"""

from __future__ import annotations

from typing import Deque, Dict, List, Tuple

import pandas as pd

from .data import DataHandler
from .events import FillEvent, OrderEvent, SignalEvent


class Portfolio:
    def __init__(
        self,
        data: DataHandler,
        events: Deque,
        initial_capital: float = 100_000.0,
        target_pct: float = 0.95,
        allow_short: bool = False,
    ):
        self.data = data
        self.events = events
        self.initial_capital = float(initial_capital)
        self.target_pct = target_pct
        self.allow_short = allow_short

        self.cash = float(initial_capital)
        self.positions: Dict[str, int] = {s: 0 for s in data.symbols}
        self.avg_price: Dict[str, float] = {s: 0.0 for s in data.symbols}

        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.closed_trades: List[float] = []  # realised PnL per closed/reduced lot

    # ----- mark to market -------------------------------------------------------
    def update_timeindex(self, _event) -> None:
        """Called on each MarketEvent: value the book and append to the equity curve."""
        ts = self.data.current_time
        market_value = 0.0
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.data.latest_close(sym)
            if price is not None:
                market_value += qty * price
        self.equity_curve.append((ts, self.cash + market_value))

    # ----- signal -> order ------------------------------------------------------
    def update_signal(self, signal: SignalEvent) -> None:
        order = self._build_order(signal)
        if order is not None:
            self.events.append(order)

    def _build_order(self, signal: SignalEvent):
        sym = signal.symbol
        price = self.data.latest_close(sym)
        if price is None or price <= 0:
            return None
        current = self.positions[sym]

        if signal.direction == "EXIT":
            if current == 0:
                return None
            direction = "SELL" if current > 0 else "BUY"
            return OrderEvent(sym, signal.timestamp, direction, abs(current))

        if signal.direction == "LONG":
            target_qty = self._size_order(price, signal.strength)
            delta = target_qty - current
            if delta == 0:
                return None
            return OrderEvent(sym, signal.timestamp, "BUY" if delta > 0 else "SELL", abs(delta))

        if signal.direction == "SHORT":
            if not self.allow_short:
                return None
            target_qty = -self._size_order(price, signal.strength)
            delta = target_qty - current
            if delta == 0:
                return None
            return OrderEvent(sym, signal.timestamp, "BUY" if delta > 0 else "SELL", abs(delta))

        return None

    def _size_order(self, price: float, strength: float) -> int:
        """Fixed-fractional sizing against current total equity."""
        equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        budget = equity * self.target_pct * max(0.0, min(1.0, strength))
        return int(budget // price)

    # ----- fill -> position/cash ------------------------------------------------
    def update_fill(self, fill: FillEvent) -> None:
        sym = fill.symbol
        signed = fill.quantity if fill.direction == "BUY" else -fill.quantity

        old_qty = self.positions[sym]
        old_avg = self.avg_price[sym]
        new_qty = old_qty + signed

        # Realise PnL on any portion that reduces or closes the existing position.
        if old_qty != 0 and (old_qty > 0) != (signed > 0):
            closed = min(abs(signed), abs(old_qty))
            direction_sign = 1 if old_qty > 0 else -1
            pnl = (fill.fill_price - old_avg) * closed * direction_sign
            self.closed_trades.append(pnl - fill.commission)

        # Update average entry price.
        if new_qty == 0:
            self.avg_price[sym] = 0.0
        elif old_qty == 0 or (old_qty > 0) == (signed > 0):
            # Opening or adding in the same direction -> weighted average.
            self.avg_price[sym] = (old_avg * abs(old_qty) + fill.fill_price * abs(signed)) / abs(new_qty)
        elif (new_qty > 0) != (old_qty > 0):
            # Flipped through zero -> the remainder is a fresh position at fill price.
            self.avg_price[sym] = fill.fill_price
        # else: pure reduction, average price unchanged.

        self.positions[sym] = new_qty
        self.cash -= fill.fill_price * signed + fill.commission

    # ----- output ---------------------------------------------------------------
    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        ts, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="equity")

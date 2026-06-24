"""Simulated order execution.

Turns an OrderEvent into a FillEvent, applying slippage and commission. Being
explicit and conservative about costs is exactly what separates a serious backtest
from a toy one — so these are first-class, configurable inputs, not afterthoughts.
"""

from __future__ import annotations

from typing import Deque

from .data import DataHandler
from .events import FillEvent, OrderEvent


class SimulatedExecutionHandler:
    def __init__(
        self,
        data: DataHandler,
        events: Deque,
        slippage_bps: float = 5.0,
        commission_bps: float = 1.0,
        min_commission: float = 1.0,
    ):
        """
        slippage_bps:   adverse price move per fill, in basis points (5 = 0.05%).
        commission_bps: cost per trade as bps of notional.
        min_commission: floor on per-trade commission, in account currency.
        """
        self.data = data
        self.events = events
        self.slippage = slippage_bps / 10_000.0
        self.commission_bps = commission_bps / 10_000.0
        self.min_commission = min_commission

    def execute_order(self, order: OrderEvent) -> None:
        ref_price = self.data.latest_close(order.symbol)
        if ref_price is None:
            return  # no market data yet; drop the order

        # Slippage always works against us: pay up to buy, sell into weakness.
        if order.direction == "BUY":
            fill_price = ref_price * (1 + self.slippage)
        else:
            fill_price = ref_price * (1 - self.slippage)

        notional = fill_price * order.quantity
        commission = max(self.min_commission, notional * self.commission_bps)

        self.events.append(
            FillEvent(
                symbol=order.symbol,
                timestamp=order.timestamp,
                direction=order.direction,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
            )
        )

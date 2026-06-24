"""Thin wrapper over the Alpaca paper-trading API.

Isolates every network call behind a small, mockable surface so the trading logic
can be unit-tested without hitting Alpaca. Imports of the SDK are lazy so the rest
of the project (backtests, dashboard skeleton) works without alpaca-py installed.
"""

from __future__ import annotations

from .config import AlpacaConfig


class AlpacaBroker:
    def __init__(self, config: AlpacaConfig | None = None):
        from alpaca.trading.client import TradingClient

        self.config = config or AlpacaConfig.from_env()
        # paper=True is enforced both here and in AlpacaConfig.
        self.client = TradingClient(self.config.api_key, self.config.secret_key, paper=True)

    # ----- account / positions --------------------------------------------------
    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def account_snapshot(self) -> dict:
        a = self.client.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "last_equity": float(a.last_equity),
        }

    def positions(self) -> dict[str, int]:
        """symbol -> signed share quantity currently held."""
        return {p.symbol: int(float(p.qty)) for p in self.client.get_all_positions()}

    def position_details(self) -> list[dict]:
        out = []
        for p in self.client.get_all_positions():
            out.append({
                "symbol": p.symbol,
                "qty": int(float(p.qty)),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        return out

    # ----- orders ---------------------------------------------------------------
    def submit(self, symbol: str, qty: int, side: str):
        """Submit a market order for ``qty`` shares. ``side`` is 'BUY' or 'SELL'."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty <= 0:
            return None
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order)

    # ----- progression (best-effort; dashboard falls back to the local log) -----
    def portfolio_history(self):
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            req = GetPortfolioHistoryRequest(period="3M", timeframe="1D")
            return self.client.get_portfolio_history(req)
        except Exception:
            return None

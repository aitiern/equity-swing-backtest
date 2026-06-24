"""Auto paper-trader: compute signals -> reconcile vs positions -> submit paper orders.

    python -m src.live.trade --dry-run     # safe: prints intended orders, no keys, no submit
    python -m src.live.trade               # live PAPER trading (needs .env keys)

The sizing/reconciliation functions are pure and unit-tested; only main() touches the
network. Always run --dry-run first to see exactly what it would do.
"""

from __future__ import annotations

import argparse
import csv
import os

import pandas as pd

from ..universe import resolve
from .signals import desired_holdings, fetch_frames

LOG_PATH = "tracking/equity_log.csv"
LOG_FIELDS = ["timestamp", "equity", "cash", "n_positions", "holdings"]


def compute_target_shares(equity, holdings, symbols, prices, target_pct=0.95):
    """Equal-weight target share count per symbol (0 if not a desired holding).
    Mirrors the backtest's equal-weight sizing across the universe."""
    n = max(1, len(symbols))
    per_name = equity * target_pct / n
    targets = {}
    for s in symbols:
        price = prices.get(s)
        targets[s] = int(per_name // price) if (s in holdings and price and price > 0) else 0
    return targets


def reconcile(targets: dict[str, int], current: dict[str, int]):
    """Return the list of (symbol, qty, side) market orders to move current -> target."""
    orders = []
    for sym, tgt in targets.items():
        delta = tgt - current.get(sym, 0)
        if delta > 0:
            orders.append((sym, delta, "BUY"))
        elif delta < 0:
            orders.append((sym, -delta, "SELL"))
    # Defensively flatten anything held but not in the target universe.
    for sym, cur in current.items():
        if sym not in targets and cur > 0:
            orders.append((sym, cur, "SELL"))
    return orders


def _log_equity(snapshot: dict, holdings) -> None:
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "equity": snapshot["equity"],
            "cash": snapshot["cash"],
            "n_positions": len(holdings),
            "holdings": "|".join(sorted(holdings)),
        })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Alpaca paper auto-trader")
    p.add_argument("--dry-run", action="store_true", help="print intended orders, submit nothing")
    p.add_argument("--check", action="store_true", help="read-only: verify keys + show account, no orders")
    p.add_argument("--sector", default=os.getenv("TRADE_SECTOR", "tech"))
    p.add_argument("--strategy", default=os.getenv("TRADE_STRATEGY", "donchian"))
    p.add_argument("--capital", type=float, default=100_000.0, help="hypothetical equity for --dry-run")
    args = p.parse_args(argv)

    if args.check:
        from .broker import AlpacaBroker

        broker = AlpacaBroker()
        snap = broker.account_snapshot()
        print(f"CONNECTED to Alpaca paper. Equity ${snap['equity']:,.2f} | "
              f"Cash ${snap['cash']:,.2f} | Buying power ${snap['buying_power']:,.2f}")
        positions = broker.position_details()
        print(f"Open positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos['symbol']:5} {pos['qty']:>5} sh  P/L ${pos['unrealized_pl']:,.2f}")
        return 0

    symbols = resolve(args.sector)
    print(f"Universe: {args.sector} ({', '.join(symbols)}) | strategy: {args.strategy}")
    frames = fetch_frames(symbols)
    holdings, prices = desired_holdings(symbols, args.strategy, frames)
    print(f"Strategy wants to be LONG: {sorted(holdings) or '(nothing)'}")

    if args.dry_run:
        targets = compute_target_shares(args.capital, holdings, symbols, prices)
        orders = reconcile(targets, current={})  # assume flat
        print(f"\n[DRY RUN] hypothetical equity ${args.capital:,.0f} — intended orders:")
        for sym, qty, side in orders:
            price = prices.get(sym) or 0.0
            print(f"  {side:4} {qty:>5} {sym} @ ~${price:.2f}")
        if not orders:
            print("  (none)")
        print("\nNothing submitted. Remove --dry-run to trade on paper.")
        return 0

    # ---- live paper path ----
    from .broker import AlpacaBroker

    broker = AlpacaBroker()
    snapshot = broker.account_snapshot()
    current = broker.positions()
    targets = compute_target_shares(snapshot["equity"], holdings, symbols, prices)
    orders = reconcile(targets, current)

    print(f"\nPaper account equity: ${snapshot['equity']:,.2f} | submitting {len(orders)} orders")
    for sym, qty, side in orders:
        broker.submit(sym, qty, side)
        print(f"  {side:4} {qty:>5} {sym}")

    _log_equity(snapshot, holdings)
    print(f"\nLogged equity snapshot to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

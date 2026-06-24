"""Entry point: wires the components together and runs a backtest.

Run it:
    python -m src.backtest                  # runs on built-in DEMO data
    python -m src.backtest --csv-dir data --symbols AAPL MSFT

The DEMO path lets a reviewer clone and run immediately. Point ``--csv-dir`` at real
OHLCV CSVs (one <SYMBOL>.csv per symbol, first column a parseable date) to backtest
your actual data, then swap MovingAverageCrossStrategy for your real strategy.
"""

from __future__ import annotations

import argparse
from collections import deque

from .data import DataHandler
from .engine import Backtest
from .execution import SimulatedExecutionHandler
from .metrics import format_summary, summary
from .plotting import save_equity_curve
from .portfolio import Portfolio
from .strategy import MovingAverageCrossStrategy


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Event-driven backtest")
    p.add_argument("--csv-dir", default=None, help="directory of <SYMBOL>.csv OHLCV files")
    p.add_argument("--symbols", nargs="*", default=None, help="symbols to load from --csv-dir")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--commission-bps", type=float, default=1.0)
    p.add_argument("--short", type=int, default=20, help="demo strategy fast window")
    p.add_argument("--long", type=int, default=50, help="demo strategy slow window")
    p.add_argument("--plot", default="docs/equity-curve.png")
    p.add_argument("--no-plot", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    events: deque = deque()

    if args.csv_dir and args.symbols:
        data = DataHandler.from_csv_dir(args.csv_dir, args.symbols, events)
    else:
        print("No --csv-dir/--symbols given; running on synthetic DEMO data.")
        data = DataHandler.demo(events)

    strategy = MovingAverageCrossStrategy(data, events, short=args.short, long=args.long)
    portfolio = Portfolio(data, events, initial_capital=args.capital)
    execution = SimulatedExecutionHandler(
        data, events, slippage_bps=args.slippage_bps, commission_bps=args.commission_bps
    )

    result = Backtest(data, strategy, portfolio, execution, events).run()

    equity = result.equity_series()
    if equity.empty:
        print("No equity recorded — check that data loaded correctly.")
        return 1

    stats = summary(equity, result.closed_trades)
    print(format_summary(stats))

    if not args.no_plot:
        path = save_equity_curve(equity, args.plot)
        print(f"\nEquity curve written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

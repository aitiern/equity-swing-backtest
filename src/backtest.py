"""Entry point: wires the components together and runs a backtest.

Run it:
    python -m src.backtest                         # built-in DEMO data
    python -m src.backtest --config config.yaml    # reproducible run from a file
    python -m src.backtest --csv-dir data --symbols AAPL MSFT --oos-start 2023-01-01

Defaults come from --config (if given); any CLI flag overrides the file. The DEMO
path lets a reviewer clone and run immediately. Point --csv-dir at real OHLCV CSVs
(one <SYMBOL>.csv per symbol, first column a parseable date) and swap
MovingAverageCrossStrategy for your real signal to make it yours.
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import pandas as pd

from .data import DataHandler
from .engine import Backtest
from .execution import SimulatedExecutionHandler
from .metrics import format_summary, summary
from .plotting import save_equity_curve
from .portfolio import Portfolio
from .strategy import STRATEGIES, MovingAverageCrossStrategy


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Event-driven backtest")
    p.add_argument("--config", default=None, help="YAML file of defaults (CLI flags override it)")
    p.add_argument("--csv-dir", default=None, help="directory of <SYMBOL>.csv OHLCV files")
    p.add_argument("--symbols", nargs="*", default=None, help="symbols to load from --csv-dir")
    p.add_argument("--yf", nargs="*", default=None, help="ticker(s) to download from Yahoo Finance")
    p.add_argument("--start", default="2015-01-01", help="start date for --yf download")
    p.add_argument("--end", default=None, help="end date for --yf download (default: today)")
    p.add_argument(
        "--strategy", default="ma_cross", choices=list(STRATEGIES),
        help="which strategy to run (default: ma_cross)",
    )
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--commission-bps", type=float, default=1.0)
    p.add_argument("--short", type=int, default=20, help="demo strategy fast window")
    p.add_argument("--long", type=int, default=50, help="demo strategy slow window")
    p.add_argument(
        "--oos-start",
        default=None,
        help="date (YYYY-MM-DD) splitting in-sample from held-out out-of-sample reporting",
    )
    p.add_argument("--plot", default="docs/equity-curve.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--blotter", default="results/trades.csv", help="trade log output path")
    return p


def _apply_config(parser: argparse.ArgumentParser, argv) -> argparse.Namespace:
    """Two-pass parse: load --config as defaults, then let real CLI flags win."""
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        import yaml

        with open(pre.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        # Map config keys (underscored) onto argparse dests.
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return parser.parse_args(argv)


def _report(label: str, equity: pd.Series, closed_trades) -> None:
    print(f"\n=== {label} ({equity.index[0].date()} -> {equity.index[-1].date()}) ===")
    print(format_summary(summary(equity, closed_trades)))


def main(argv=None) -> int:
    parser = build_argparser()
    args = _apply_config(parser, argv)
    events: deque = deque()

    if args.yf:
        print(f"Downloading {args.yf} from Yahoo Finance ({args.start}..{args.end or 'today'})...")
        data = DataHandler.from_yfinance(args.yf, args.start, args.end, events)
    elif args.csv_dir and args.symbols:
        data = DataHandler.from_csv_dir(args.csv_dir, args.symbols, events)
    else:
        print("No data source given; running on synthetic DEMO data.")
        data = DataHandler.demo(events)

    if args.strategy == "ma_cross":
        strategy = MovingAverageCrossStrategy(data, events, short=args.short, long=args.long)
    else:
        strategy = STRATEGIES[args.strategy](data, events)
    portfolio = Portfolio(data, events, initial_capital=args.capital)
    execution = SimulatedExecutionHandler(
        data, events, slippage_bps=args.slippage_bps, commission_bps=args.commission_bps
    )

    result = Backtest(data, strategy, portfolio, execution, events).run()

    equity = result.equity_series()
    if equity.empty:
        print("No equity recorded — check that data loaded correctly.")
        return 1

    # Full-period stats, plus the equal-weight buy-and-hold benchmark.
    _report("FULL PERIOD", equity, result.closed_trades)
    benchmark = data.benchmark_equity(args.capital)
    if not benchmark.empty:
        _report("BENCHMARK (buy & hold)", benchmark, [])

    # Out-of-sample split: report the held-out segment separately. A strategy that
    # only looks good in-sample is overfit — this is the honesty check.
    oos_start = pd.Timestamp(args.oos_start) if args.oos_start else None
    if oos_start is not None:
        in_sample = equity[equity.index < oos_start]
        out_sample = equity[equity.index >= oos_start]
        if len(in_sample) > 1:
            _report("IN-SAMPLE", in_sample, [])
        if len(out_sample) > 1:
            _report("OUT-OF-SAMPLE", out_sample, [])

    # Auditable trade blotter.
    blotter = result.blotter()
    if not blotter.empty and args.blotter:
        os.makedirs(os.path.dirname(args.blotter) or ".", exist_ok=True)
        blotter.to_csv(args.blotter, index=False)
        print(f"\nTrade blotter written to {args.blotter} ({len(blotter)} fills)")

    if not args.no_plot:
        path = save_equity_curve(equity, args.plot, benchmark=benchmark, oos_start=oos_start)
        print(f"Equity curve written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

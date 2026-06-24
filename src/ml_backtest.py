"""Backtest the per-sector walk-forward ML strategy through the real engine.

    python -m src.ml_backtest --sector tech --start 2015-01-01
    python -m src.ml_backtest --yf AAPL MSFT NVDA --threshold 0.55

Pipeline: download data -> generate walk-forward signals (src/ml) -> execute them
with costs/sizing (the same engine the rule-based strategies use) -> report metrics
restricted to the out-of-sample period, against buy-and-hold.
"""

from __future__ import annotations

import argparse
from collections import deque

import pandas as pd

from .data import DataHandler
from .engine import Backtest
from .execution import SimulatedExecutionHandler
from .metrics import format_summary, summary
from .ml.walkforward import walk_forward_signals
from .plotting import save_equity_curve
from .portfolio import Portfolio
from .strategy import MLSignalStrategy
from .universe import resolve


def _load_frames(symbols, start, end):
    """Fetch + cache via the DataHandler, then read the cached CSVs as raw frames."""
    DataHandler.from_yfinance(symbols, start, end, deque(), cache_dir="data")
    return {
        sym: pd.read_csv(f"data/{sym}.csv", index_col=0, parse_dates=True)
        for sym in symbols
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Per-sector walk-forward ML backtest")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sector", help="named sector basket (tech, financials, ...)")
    src.add_argument("--yf", nargs="*", help="explicit ticker list")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--min-train", type=int, default=504, help="bars before first prediction")
    p.add_argument("--step", type=int, default=63, help="retrain frequency in bars")
    p.add_argument("--threshold", type=float, default=0.5, help="P(up) needed to go long")
    p.add_argument("--lookback", type=int, default=None, help="rolling train window (days); omit=expanding")
    p.add_argument("--plot", default=None, help="path to save the equity curve PNG")
    args = p.parse_args(argv)

    symbols = resolve(args.sector) if args.sector else args.yf
    label = args.sector or ", ".join(symbols)
    print(f"Sector model: {label} ({', '.join(symbols)})")
    print(f"Period {args.start}..{args.end or 'today'} | retrain every {args.step} bars "
          f"| threshold {args.threshold}\n")

    frames = _load_frames(symbols, args.start, args.end)
    wf = walk_forward_signals(
        frames, min_train=args.min_train, step=args.step,
        threshold=args.threshold, lookback=args.lookback,
    )

    print("--- Walk-forward model diagnostics ---")
    print(f"folds (retrains): {wf.folds}")
    print(f"OOS predictions:  {wf.n_predictions}")
    print(f"directional accuracy: {wf.accuracy * 100:.2f}%  (base rate / always-long: "
          f"{wf.base_rate * 100:.2f}%)")
    edge = (wf.accuracy - wf.base_rate) * 100
    print(f"edge over base rate:  {edge:+.2f} pts  "
          f"{'(model adds signal)' if edge > 0 else '(NO edge — model is not beating naive)'}")
    if wf.first_prediction is None:
        print("\nNo predictions produced — need more history (lower --min-train).")
        return 1

    # Execute the signals through the real engine.
    events: deque = deque()
    data = DataHandler.from_csv_dir("data", symbols, events)
    strategy = MLSignalStrategy(data, events, wf.signals)
    portfolio = Portfolio(data, events, initial_capital=args.capital)
    execution = SimulatedExecutionHandler(data, events)
    result = Backtest(data, strategy, portfolio, execution, events).run()

    equity = result.equity_series()
    benchmark = data.benchmark_equity(args.capital)
    # Report only the live (out-of-sample) window — before the first prediction the
    # book is flat, which would otherwise dilute the metrics.
    start_oos = wf.first_prediction
    eq_oos = equity[equity.index >= start_oos]
    bench_oos = benchmark[benchmark.index >= start_oos]

    print(f"\n=== ML STRATEGY (out-of-sample, {eq_oos.index[0].date()} -> "
          f"{eq_oos.index[-1].date()}) ===")
    print(format_summary(summary(eq_oos, result.closed_trades)))
    print("\n=== BENCHMARK (buy & hold, same window) ===")
    print(format_summary(summary(bench_oos, [])))

    if args.plot:
        path = save_equity_curve(eq_oos, args.plot, benchmark=bench_oos,
                                 title=f"Per-sector ML strategy — {label}",
                                 oos_start=start_oos)
        print(f"\nEquity curve written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

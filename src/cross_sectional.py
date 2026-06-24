"""Cross-sectional ranking strategy.

Instead of predicting market direction (which we showed the model can't do), this
predicts *relative* performance: will this name beat the sector's median over the
next ``horizon`` days? It then holds the top-k highest-ranked names, rebalanced
periodically. The benchmark is the equal-weight sector basket, so this isolates
stock-SELECTION skill from market direction — where real equity-ML edges usually
live, if they live anywhere.

    python -m src.cross_sectional --sector tech --horizon 5 --topk 2
    python -m src.cross_sectional --sector energy --horizon 10 --topk 2 --rebalance 5
"""

from __future__ import annotations

import argparse

from .metrics import format_summary, summary
from .ml.execute import backtest_signals, load_frames, precision_at_k, topk_signals
from .ml.walkforward import walk_forward_signals
from .universe import resolve


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-sectional ML ranking strategy")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sector", help="named sector basket")
    src.add_argument("--yf", nargs="*", help="explicit ticker list")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--horizon", type=int, default=5, help="forward window for outperformance label")
    p.add_argument("--topk", type=int, default=2, help="number of names to hold")
    p.add_argument("--rebalance", type=int, default=5, help="rebalance frequency in bars")
    p.add_argument("--min-train", type=int, default=504)
    p.add_argument("--step", type=int, default=63)
    args = p.parse_args(argv)

    symbols = resolve(args.sector) if args.sector else args.yf
    label = args.sector or ", ".join(symbols)
    k = min(args.topk, len(symbols))
    print(f"Cross-sectional ranking: {label} ({', '.join(symbols)})")
    print(f"horizon={args.horizon}d  hold top {k} of {len(symbols)}  "
          f"rebalance every {args.rebalance} bars\n")

    frames = load_frames(symbols, args.start, args.end)
    wf = walk_forward_signals(
        frames, min_train=args.min_train, step=args.step,
        horizon=args.horizon, relative=True,
    )
    if wf.first_prediction is None:
        print("\nNot enough history.")
        return 1

    # Honest selection metric: precision@k (did the top-ranked names actually
    # outperform?) vs the k/N base rate. Classification accuracy is reported too, but
    # against the MAJORITY-class baseline so it can't look good by predicting "no".
    prec = precision_at_k(wf.proba, wf.targets, k)
    base = k / len(symbols)
    print("--- Model diagnostics (predicting sector outperformance) ---")
    print(f"classification accuracy: {wf.accuracy * 100:.2f}%  "
          f"(majority-class baseline: {wf.majority_baseline * 100:.2f}%, "
          f"edge {wf.classification_edge:+.2f} pts)")
    print(f"precision@{k}: {prec * 100:.2f}%  (base rate {base * 100:.2f}%, "
          f"edge {(prec - base) * 100:+.2f} pts)  "
          f"{'<- ranks better than chance' if prec > base else '<- NO selection edge'}")

    signals = topk_signals(wf.proba, k, rebalance=args.rebalance)
    # Scale sizing so the k held names are ~fully invested (each ~1/k of equity).
    target_pct = 0.95 * len(symbols) / k
    result, data = backtest_signals(symbols, signals, args.capital, target_pct=target_pct)

    equity = result.equity_series()
    bench = data.benchmark_equity(args.capital)
    oos = wf.first_prediction
    eq = equity[equity.index >= oos]
    bench = bench[bench.index >= oos]

    print(f"\n=== CROSS-SECTIONAL STRATEGY (out-of-sample, {eq.index[0].date()} -> "
          f"{eq.index[-1].date()}) ===")
    print(format_summary(summary(eq, result.closed_trades)))
    print("\n=== BENCHMARK (equal-weight sector, same window) ===")
    print(format_summary(summary(bench, [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

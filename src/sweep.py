"""Sector x threshold sweep of the directional ML model.

Trains one walk-forward model per sector (training is independent of the trading
threshold, so we train ONCE per sector and apply every threshold to the cached
probabilities), then tabulates directional accuracy and risk-adjusted performance
vs buy-and-hold across the whole grid.

    python -m src.sweep --start 2015-01-01 --end 2024-01-01
    python -m src.sweep --sectors tech energy --thresholds 0.50 0.55 --horizon 5
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from .metrics import sharpe_ratio, summary
from .ml.execute import backtest_signals, load_frames, signals_from_proba
from .ml.walkforward import walk_forward_signals
from .universe import SECTORS


def run_sweep(sectors, thresholds, start, end, horizon, min_train, step, capital=100_000.0):
    rows = []
    for sector in sectors:
        symbols = SECTORS[sector]
        print(f"  [{sector}] downloading + training walk-forward model...")
        frames = load_frames(symbols, start, end)
        wf = walk_forward_signals(frames, min_train=min_train, step=step, horizon=horizon)
        if wf.first_prediction is None:
            print(f"  [{sector}] not enough history — skipped")
            continue
        oos = wf.first_prediction

        for thr in thresholds:
            signals = signals_from_proba(wf.proba, thr)
            result, data = backtest_signals(symbols, signals, capital)
            equity = result.equity_series()
            eq = equity[equity.index >= oos]
            bench = data.benchmark_equity(capital)
            bench = bench[bench.index >= oos]
            stats = summary(eq, result.closed_trades)
            rows.append({
                "sector": sector,
                "thr": thr,
                "accuracy": wf.accuracy,
                "maj_base": wf.majority_baseline,
                "edge_pts": wf.classification_edge,
                "sharpe": stats["sharpe"],
                "bench_sharpe": sharpe_ratio(bench),
                "max_dd": stats["max_drawdown"],
                "trades": stats["num_trades"],
            })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    out = df.copy()
    out["accuracy"] = (out["accuracy"] * 100).map(lambda v: f"{v:5.2f}%")
    out["maj_base"] = (out["maj_base"] * 100).map(lambda v: f"{v:5.2f}%")
    out["edge_pts"] = out["edge_pts"].map(lambda v: f"{v:+5.2f}")
    out["max_dd"] = (out["max_dd"] * 100).map(lambda v: f"{v:6.1f}%")
    out["sharpe"] = out["sharpe"].map(lambda v: f"{v:5.2f}")
    out["bench_sharpe"] = out["bench_sharpe"].map(lambda v: f"{v:5.2f}")
    out["beat?"] = df.apply(lambda r: "YES" if r["sharpe"] > r["bench_sharpe"] else "no", axis=1)
    return out.to_string(index=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sector x threshold ML sweep")
    p.add_argument("--sectors", nargs="*", default=list(SECTORS), choices=list(SECTORS))
    p.add_argument("--thresholds", nargs="*", type=float, default=[0.50, 0.52, 0.55])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--horizon", type=int, default=1, help="label horizon in days (1=daily, 5=weekly)")
    p.add_argument("--min-train", type=int, default=504)
    p.add_argument("--step", type=int, default=63)
    p.add_argument("--out", default="results/ml_sweep.csv")
    args = p.parse_args(argv)

    print(f"Sweep: sectors={args.sectors} thresholds={args.thresholds} "
          f"horizon={args.horizon} ({args.start}..{args.end or 'today'})")
    df = run_sweep(args.sectors, args.thresholds, args.start, args.end,
                   args.horizon, args.min_train, args.step)
    if df.empty:
        print("No results.")
        return 1

    print("\n" + format_table(df))
    won = (df["sharpe"] > df["bench_sharpe"]).sum()
    print(f"\nConfigs beating buy-and-hold on Sharpe: {won}/{len(df)}.")
    print("Directional 'edge_pts' is accuracy minus the always-long base rate — "
          "positive means the model actually predicts; near-zero/negative means it does not.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nFull grid saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run every registered strategy on the same universe and tabulate the results.

This is the baseline harness: it backtests each strategy on identical data with
identical costs, alongside the buy-and-hold benchmark, so they can be compared
fairly. When an out-of-sample date is given, it reports the held-out Sharpe too —
because in-sample ranking is meaningless; only out-of-sample survival counts.

    python -m src.compare --sector tech --start 2015-01-01 --oos-start 2021-01-01
    python -m src.compare --yf AAPL MSFT --start 2018-01-01
"""

from __future__ import annotations

import argparse
from collections import deque

import pandas as pd

from .data import DataHandler
from .engine import Backtest
from .execution import SimulatedExecutionHandler
from .metrics import sharpe_ratio, summary
from .portfolio import Portfolio
from .strategy import STRATEGIES
from .universe import resolve


def _build_data(events, symbols, start, end, cache_dir):
    return DataHandler.from_yfinance(symbols, start, end, events, cache_dir=cache_dir)


def run_comparison(symbols, start, end, capital=100_000.0, oos_start=None,
                   slippage_bps=5.0, commission_bps=1.0):
    """Backtest every strategy + benchmark on the same data. Returns a DataFrame."""
    # Fetch once into the cache, then rebuild a fresh (stateful) DataHandler per run.
    _build_data(deque(), symbols, start, end, cache_dir="data")

    oos = pd.Timestamp(oos_start) if oos_start else None
    rows = []

    def oos_sharpe(equity):
        if oos is None:
            return None
        held = equity[equity.index >= oos]
        return sharpe_ratio(held) if len(held) > 1 else None

    for name, cls in STRATEGIES.items():
        events = deque()
        data = DataHandler.from_csv_dir("data", symbols, events)
        strategy = cls(data, events)
        portfolio = Portfolio(data, events, initial_capital=capital)
        execution = SimulatedExecutionHandler(
            data, events, slippage_bps=slippage_bps, commission_bps=commission_bps
        )
        result = Backtest(data, strategy, portfolio, execution, events).run()
        equity = result.equity_series()
        stats = summary(equity, result.closed_trades)
        rows.append({
            "strategy": name,
            "total_return": stats["total_return"],
            "cagr": stats["cagr"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "cvar_95": stats["cvar_95"],
            "trades": stats["num_trades"],
            "oos_sharpe": oos_sharpe(equity),
        })

    # Buy-and-hold benchmark as the bar to clear.
    events = deque()
    data = DataHandler.from_csv_dir("data", symbols, events)
    bench = data.benchmark_equity(capital)
    bstats = summary(bench, [])
    rows.append({
        "strategy": "BENCHMARK",
        "total_return": bstats["total_return"],
        "cagr": bstats["cagr"],
        "sharpe": bstats["sharpe"],
        "max_drawdown": bstats["max_drawdown"],
        "cvar_95": bstats["cvar_95"],
        "trades": 0,
        "oos_sharpe": oos_sharpe(bench),
    })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ["total_return", "cagr", "max_drawdown", "cvar_95"]:
        out[col] = (out[col] * 100).map(lambda v: f"{v:6.1f}%")
    out["sharpe"] = out["sharpe"].map(lambda v: f"{v:5.2f}")
    out["oos_sharpe"] = out["oos_sharpe"].map(lambda v: "  -  " if v is None else f"{v:5.2f}")
    return out.to_string(index=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare all strategies on one universe")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sector", help="named sector basket (tech, financials, energy, ...)")
    src.add_argument("--yf", nargs="*", help="explicit list of tickers")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--oos-start", default=None, help="report out-of-sample Sharpe from this date")
    p.add_argument("--capital", type=float, default=100_000.0)
    args = p.parse_args(argv)

    symbols = resolve(args.sector) if args.sector else args.yf
    label = args.sector or ", ".join(symbols)
    print(f"Universe: {label} ({', '.join(symbols)})")
    print(f"Period: {args.start}..{args.end or 'today'}"
          + (f"  |  OOS from {args.oos_start}" if args.oos_start else ""))

    df = run_comparison(symbols, args.start, args.end, args.capital, args.oos_start)
    print("\n" + format_table(df))
    print("\nRead it this way: beat the BENCHMARK row on risk-adjusted return (Sharpe),"
          " and demand a positive OOS Sharpe before believing the edge is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

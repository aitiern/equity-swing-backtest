# [Strategy Name] — [one-line description, e.g. "Mean-reversion strategy on US large-cap equities"]

[![CI](https://github.com/aitiern/equity-swing-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/aitiern/equity-swing-backtest/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Headline result:** Backtested [START]–[END] on [universe]: Sharpe **[x.xx]**, max drawdown **[xx]%**,
> [n] trades. Benchmark: [e.g. SPY buy-and-hold, Sharpe x.xx].
>
> ⚠️ Replace every bracketed value with your real, verified figures before making this repo public.
> Do not publish placeholder numbers.

![Equity curve](docs/equity-curve.png)
<!-- Drop your real equity-curve / results chart at docs/equity-curve.png. A chart at the top is
     what gets a screener to keep reading. -->

## Overview
One short paragraph: what the strategy does, on what market, and why it's interesting. Written for a
reader who spends 30 seconds before deciding whether to read on.

## The idea
- **Hypothesis:** [what market behavior you're exploiting]
- **Signal:** [how you generate entry/exit signals]
- **Universe & timeframe:** [instruments, bar frequency, date range]

## Data
- Source: [e.g. yfinance, vendor, CSV] — note what's included in `data/` vs fetched at runtime.
- [Any cleaning / survivorship-bias / corporate-action handling worth mentioning.]

## Backtest methodology
- **Costs & slippage:** modelled explicitly (`--slippage-bps`, `--commission-bps`); slippage always
  works against the fill. [State your assumptions and why.]
- **Position sizing / risk controls:** fixed-fractional sizing against current equity (`Portfolio`).
  [Swap for vol-targeting / fixed-risk if you use it.]
- **Out-of-sample:** pass `--oos-start YYYY-MM-DD` to report held-out performance separately from the
  fitting period — the overfitting guard. [Say which dates you held out.]
- **Market risk:** historical & parametric VaR plus CVaR (expected shortfall) at 95% / 99% are reported
  every run. [Comment on tail behaviour.]

## Results
<!-- Fill from a real run: `python -m src.backtest --config config.yaml`. -->
| Metric | Strategy | Benchmark |
|---|---|---|
| CAGR | [ ] | [ ] |
| Sharpe | [ ] | [ ] |
| Max drawdown | [ ] | [ ] |
| 95% VaR (daily) | [ ] | [ ] |
| 95% CVaR (daily) | [ ] | [ ] |
| Win rate | [ ] | — |

## Limitations & next steps
Honest paragraph on what this doesn't capture and what you'd do with more time. Showing you know the
weaknesses reads as maturity — especially to a risk audience.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                 # runtime + test/lint tooling

python -m src.backtest                               # built-in synthetic demo data
python -m src.backtest --config config.yaml          # reproducible run from a config file
python -m src.backtest --csv-dir data --symbols AAPL --oos-start 2023-01-01

pytest          # run the test suite
ruff check .    # lint
```

Outputs: a performance summary (full / benchmark / in-sample / out-of-sample), an equity-curve chart
at `docs/equity-curve.png`, and an auditable trade blotter at `results/trades.csv`.

## Project layout
```
src/        events · data · strategy · portfolio · execution · engine · metrics · plotting · backtest
tests/      metric math, cash-conservation accounting, end-to-end + lookahead-bias guard
config.yaml reproducible run configuration
```
Replace the placeholder strategy in `src/strategy.py` with your real signal — that is the only file
you must edit. Everything else (accounting, costs, metrics, charting) is reusable.

## Tech
Python · pandas · NumPy · matplotlib · pytest · ruff

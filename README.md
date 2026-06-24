# [Strategy Name] — [one-line description, e.g. "Mean-reversion strategy on US large-cap equities"]

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
- Costs & slippage: [assumptions — be explicit; this is what separates serious work from toy backtests]
- Position sizing / risk controls: [e.g. fixed fractional, vol targeting, max position]
- Out-of-sample / walk-forward: [how you guarded against overfitting]

## Results
| Metric | Strategy | Benchmark |
|---|---|---|
| CAGR | [ ] | [ ] |
| Sharpe | [ ] | [ ] |
| Max drawdown | [ ] | [ ] |
| Win rate | [ ] | — |

## Limitations & next steps
Honest paragraph on what this doesn't capture and what you'd do with more time. Showing you know the
weaknesses reads as maturity — especially to a risk audience.

## How to run
```bash
pip install -r requirements.txt
python -m src.backtest
```

## Tech
Python · pandas · NumPy · matplotlib · [add what you actually use]

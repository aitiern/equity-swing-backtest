# Findings: can ML predict equity returns from technical features?

A research log of what this engine found. The headline is a **negative result**, reported
honestly — which for a risk audience is more useful than a polished "AI beats the market"
claim that wouldn't survive scrutiny.

> **TL;DR.** Across 6 sectors, gradient-boosted models built on technical features showed
> **no reliable edge** — neither at predicting next-day direction nor at ranking which names
> outperform. Simple rule-based strategies reduced risk but did not beat buy-and-hold on a
> risk-adjusted basis. Every result below is **walk-forward, out-of-sample, after costs**
> (5 bps slippage + 1 bps commission per trade).

## Method (why these numbers are trustworthy)
- **No lookahead:** features are strictly backward-looking; a unit test asserts a feature's
  value at day *d* is identical whether computed on the full history or only data up to *d*.
- **Walk-forward validation:** models retrain on a rolling/expanding past window and predict
  the next out-of-sample slice, with a 1-day embargo so no label leaks across the boundary.
- **Real execution:** signals run through the same event-driven engine the rule-based
  strategies use — identical cost and sizing assumptions for every comparison.

## Experiment 1 — rule-based baselines (tech sector, OOS from 2021)
| Strategy | Sharpe | Max DD | OOS Sharpe |
|---|---|---|---|
| MA crossover | 1.16 | −34% | 0.86 |
| Donchian breakout | 0.95 | −21% | 0.80 |
| Bollinger reversion | 0.56 | −21% | 0.75 |
| RSI mean-reversion | 0.38 | −23% | 0.54 |
| **Buy & hold** | **1.19** | **−58%** | **1.05** |

**Takeaway:** none beat buy-and-hold on Sharpe, in- or out-of-sample. But every strategy
roughly **halved the drawdown and tail risk** — they are risk reducers, not return enhancers,
over a megacap-tech bull run.

## Experiment 2 — directional ML (next-day up/down), 6 sectors × 3 thresholds
Walk-forward `HistGradientBoosting`, 18 configurations. **Classification edge over the
majority-class baseline was negative in every sector** (−0.7 to −2.2 pts). Only 1 of 18
configs beat buy-and-hold on Sharpe, and that was a side-effect of lower market exposure,
not predictive skill. Raising the conviction threshold consistently *hurt* — there was no
signal to concentrate on.

**Takeaway:** daily direction in liquid large-caps is not predictable from these features.
This is the expected, academically consistent result — and the rigorous setup refused to
let the model pretend otherwise.

## Experiment 3 — cross-sectional ranking (predict sector outperformance)
Reframed to predict *relative* performance (beat the sector median over 5 days) and hold the
top-2 of 5 names, rebalanced weekly. Benchmark = equal-weight sector.

| Sector | Strategy Sharpe | Benchmark Sharpe | precision@2 vs 40% base |
|---|---|---|---|
| energy | 0.51 | 0.42 | +2.1 pts |
| consumer | 0.87 | 0.85 | ~ |
| financials | 0.45 | 0.46 | ~ |
| tech | 0.89 | 1.00 | +2.0 pts |
| healthcare | 0.46 | 0.68 | ~ |
| industrials | 0.10 | 0.46 | +0.1 pts |

**Beat the benchmark in 2 of 6 sectors, tied 1, lost 3 — indistinguishable from chance.**
The ranking signal (precision@2) sat just **+0.1 to +2.1 points above the base rate** — tiny
and inconsistent.

## The metric trap (the most important lesson)
The first cut of the cross-sectional diagnostic looked spectacular: *"+17 points of edge over
the base rate."* It was an illusion. With 2-of-5 names beating the median, positive labels are
only **40%** of the data, so the bar a classifier must clear is the **60% majority class**, not
40%. Measured correctly, accuracy was *below* 60%, and the honest ranking metric — **precision@k
vs the k/N base rate** — showed almost nothing. Picking the right benchmark for a metric is the
difference between a real finding and fooling yourself.

## Conclusions
1. **No tradable edge** was found from price/volume technical features via gradient boosting.
2. **Rule-based timing reduces risk** but does not add risk-adjusted return here.
3. The infrastructure is sound: any future signal can be tested the same rigorous way.

## What a real edge would likely require (next steps)
- **Better information,** not just better models: fundamentals, estimate revisions, alternative
  data — technicals alone are largely arbitraged in large-caps.
- **Longer horizons** (monthly) where return/risk is more forecastable than daily noise.
- **Proper cross-sectional factor construction** (size/value/momentum/quality) with a
  long-short, sector- and beta-neutral book — rather than long-only top-k.
- **Wider universe and more history** so a 2-of-6 result can't masquerade as signal.

*Reproduce:* `python -m src.sweep`, `python -m src.cross_sectional --sector energy`,
`python -m src.compare --sector tech --oos-start 2021-01-01`.

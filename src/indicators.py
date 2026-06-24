"""Technical indicators, computed on trailing windows of bars.

Each function takes a list of past values (oldest first) and returns the latest
indicator value, or None if there isn't enough history yet. Operating on the
trailing window only — never the full series — keeps strategies free of lookahead
bias by construction.
"""

from __future__ import annotations

import statistics


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def rolling_std(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return statistics.pstdev(values[-n:])


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI in [0, 100]. Needs period + 1 closes."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):  # Wilder smoothing over the rest
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger(closes: list[float], n: int = 20, k: float = 2.0):
    """Return (lower, mid, upper) bands, or None if insufficient history."""
    mid = sma(closes, n)
    if mid is None:
        return None
    sd = rolling_std(closes, n)
    return mid - k * sd, mid, mid + k * sd


def highest(values: list[float], n: int) -> float | None:
    return max(values[-n:]) if len(values) >= n else None


def lowest(values: list[float], n: int) -> float | None:
    return min(values[-n:]) if len(values) >= n else None

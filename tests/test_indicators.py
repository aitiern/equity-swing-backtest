"""Tests for the technical indicators the strategies rely on."""

from src import indicators


def test_sma_and_insufficient_history():
    assert indicators.sma([1, 2, 3, 4], 2) == 3.5
    assert indicators.sma([1], 5) is None


def test_rsi_all_gains_is_100():
    closes = list(range(1, 20))  # strictly rising -> no losses
    assert indicators.rsi(closes, 14) == 100.0


def test_rsi_bounds_and_midpoint():
    # Alternating up/down by equal amounts -> RSI near 50.
    closes = [10 + (1 if i % 2 else -1) * 0.5 for i in range(40)]
    value = indicators.rsi(closes, 14)
    assert 0.0 <= value <= 100.0
    assert abs(value - 50.0) < 15.0


def test_bollinger_orders_bands():
    closes = [10, 11, 9, 10, 12, 8, 10, 11, 9, 10] * 3
    lower, mid, upper = indicators.bollinger(closes, 20, 2.0)
    assert lower < mid < upper


def test_donchian_high_low():
    highs = [1, 5, 3, 7, 2]
    lows = [0, 2, 1, 4, 1]
    assert indicators.highest(highs, 3) == 7
    assert indicators.lowest(lows, 3) == 1

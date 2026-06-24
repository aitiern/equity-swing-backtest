"""Sector baskets for backtesting across a diversified set of names.

Liquid large-caps grouped by GICS-style sector. Testing a strategy on a whole
sector basket (not a single hand-picked name) is a basic guard against the
cherry-picking that makes a backtest look better than the idea really is.
"""

from __future__ import annotations

SECTORS: dict[str, list[str]] = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "industrials": ["CAT", "BA", "HON", "UNP", "GE"],
}


def resolve(name: str) -> list[str]:
    """Look up a sector basket by name (case-insensitive)."""
    key = name.lower()
    if key not in SECTORS:
        raise KeyError(f"unknown sector '{name}'. Known: {', '.join(SECTORS)}")
    return SECTORS[key]

"""Machine-learning signal generation (per-sector, walk-forward).

Kept separate from the backtest engine on purpose: this package produces *signals*
with strict no-lookahead discipline; the engine in ``src/`` then executes them with
real costs and sizing. Research and execution stay decoupled and independently
checkable.
"""

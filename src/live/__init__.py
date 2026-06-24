"""Live (paper) trading + tracking.

Connects the backtested strategies to an Alpaca PAPER account: compute today's
signals, reconcile against current positions, submit paper orders, and log equity
progression for the dashboard. Paper-only by hard guard — see config.py.
"""

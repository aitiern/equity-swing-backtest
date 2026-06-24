"""End-to-end smoke test + a lookahead-bias guard.

If these pass, the whole event pipeline is wired correctly and the strategy cannot
see the future — the two things most likely to silently break a backtest.
"""

from collections import deque

from src.data import DataHandler
from src.engine import Backtest
from src.execution import SimulatedExecutionHandler
from src.portfolio import Portfolio
from src.strategy import MovingAverageCrossStrategy


def _run(n=300):
    events = deque()
    data = DataHandler.demo(events, n=n)
    strategy = MovingAverageCrossStrategy(data, events, short=10, long=30)
    portfolio = Portfolio(data, events, initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(data, events)
    return Backtest(data, strategy, portfolio, execution, events).run(), data


def test_backtest_produces_equity_curve():
    result, _ = _run()
    equity = result.equity_series()
    assert not equity.empty
    assert equity.index.is_monotonic_increasing
    assert (equity > 0).all()  # never goes bankrupt on the gentle demo series


def test_blotter_matches_closed_trades_count():
    result, _ = _run()
    blotter = result.blotter()
    # Every realised (closed) trade must correspond to at least one recorded fill.
    if result.closed_trades:
        assert not blotter.empty
        assert len(blotter) >= len(result.closed_trades)


def test_no_lookahead_in_data_handler():
    """get_latest_bars must never expose more bars than have been streamed."""
    events = deque()
    data = DataHandler.demo(events, n=50)
    seen = 0
    while True:
        data.update_bars()
        if not data.continue_backtest:
            break
        seen += 1
        bars = data.get_latest_bars("DEMO", 9999)
        assert len(bars) <= seen  # cannot see into the future
        # The most recent bar's timestamp must equal the current simulation time.
        assert bars[-1][0] == data.current_time


def test_benchmark_starts_at_capital():
    _, data = _run()
    bench = data.benchmark_equity(100_000.0)
    assert abs(bench.iloc[0] - 100_000.0) < 1e-6

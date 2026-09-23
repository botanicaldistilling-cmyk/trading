import numpy as np
import pandas as pd
import pytest

from tradelab.backtest import BacktestConfig, run_backtest
from tradelab.config import load_instrument
from tests.conftest import FixedSignals, make_m1

CFG = BacktestConfig(initial_balance=5000, risk_pct=1.0, respect_session=False)


def flat_then(path, n_before=15):
    """One flat 15-minute bar at 1.1000, then `path` minute closes."""
    return [1.1000] * n_before + list(path)


def run(m1, plan, inst, exits=None, cfg=CFG):
    strat = FixedSignals(symbol="EURUSD", timeframe="15min", plan=plan, exits=exits or {})
    return run_backtest(strat, m1, inst, cfg)


def test_long_stop_fill_uses_bid_and_slippage(eurusd):
    # enter long at bar 1 open (ask = 1.1000 + spread), price falls through stop
    path = np.linspace(1.1000, 1.0950, 15)
    m1 = make_m1("2024-01-02 10:00", flat_then(path), spread=0.0001)
    res = run(m1, {0: (1, 1.0980, np.nan)}, eurusd)
    t = res.trades.iloc[0]
    assert t.entry == pytest.approx(1.1000 + 0.0001 + eurusd.slip_entry)
    assert t.exit_reason == "stop"
    # minute where low first <= 1.0980 opens above it, so fill is stop - slippage
    assert t.exit == pytest.approx(1.0980 - eurusd.slip_stop)
    # R is a bit worse than -1 because of stop slippage and commission
    assert -1.2 < t.r < -1.0


def test_stop_and_target_same_minute_counts_as_stop(eurusd):
    m1 = make_m1("2024-01-02 10:00", flat_then([1.1000] * 15), wick=0.0)
    # one wide minute that spans both stop and target
    m1.iloc[20, m1.columns.get_loc("high")] = 1.1100
    m1.iloc[20, m1.columns.get_loc("low")] = 1.0900
    res = run(m1, {0: (1, 1.0950, 1.1050)}, eurusd)
    assert res.trades.iloc[0].exit_reason == "stop"


def test_target_before_stop_within_bar(eurusd):
    up = list(np.linspace(1.1000, 1.1060, 6))
    down = list(np.linspace(1.1060, 1.0900, 9))
    m1 = make_m1("2024-01-02 10:00", flat_then(up + down))
    res = run(m1, {0: (1, 1.0950, 1.1050)}, eurusd)
    t = res.trades.iloc[0]
    assert t.exit_reason == "target"
    assert t.exit == pytest.approx(1.1050)


def test_gap_through_stop_fills_at_open(eurusd):
    closes = flat_then([1.1000] * 5 + [1.0900] * 10)
    m1 = make_m1("2024-01-02 10:00", closes)
    # make the gap: minute opens far below the stop
    j = 15 + 5
    m1.iloc[j, m1.columns.get_loc("open")] = 1.0900
    m1.iloc[j, m1.columns.get_loc("high")] = 1.0900
    res = run(m1, {0: (1, 1.0980, np.nan)}, eurusd)
    t = res.trades.iloc[0]
    assert t.exit == pytest.approx(1.0900 - eurusd.slip_stop)
    assert t.r < -4.5   # ~4.8R loss: the gap is far beyond the 1R stop


def test_short_stop_triggers_on_ask(eurusd):
    # bid rises to 1.1015; ask = bid + 0.0010 reaches 1.1025 >= stop 1.1020
    m1 = make_m1("2024-01-02 10:00", flat_then(list(np.linspace(1.1000, 1.1015, 15))), spread=0.0010)
    res = run(m1, {0: (-1, 1.1020, np.nan)}, eurusd)
    assert res.trades.iloc[0].exit_reason == "stop"


def test_signal_exit_at_next_open(eurusd):
    m1 = make_m1("2024-01-02 10:00", flat_then([1.1010] * 30))
    res = run(m1, {0: (1, 1.0900, np.nan)}, eurusd, exits={1: "long"})
    t = res.trades.iloc[0]
    assert t.exit_reason == "signal"
    assert t.exit_time == pd.Timestamp("2024-01-02 10:30")


def test_position_size_rounds_down_and_risk_is_capped(eurusd):
    m1 = make_m1("2024-01-02 10:00", flat_then([1.1000] * 15))
    res = run(m1, {0: (1, 1.0973, np.nan)}, eurusd)
    t = res.trades.iloc[0]
    # $50 risk over ~0.0028 distance -> 0.17857 lots -> 0.17
    assert t.lots == pytest.approx(0.17)
    assert t.risk_usd <= 50.0


def test_trade_skipped_when_min_lot_exceeds_risk(eurusd):
    m1 = make_m1("2024-01-02 10:00", flat_then([1.1000] * 15))
    cfg = BacktestConfig(initial_balance=5000, risk_pct=0.01, respect_session=False)  # $0.50 risk
    res = run(m1, {0: (1, 1.0900, np.nan)}, eurusd, cfg=cfg)
    assert res.trades.empty
    assert res.skipped["min_lot"] == 1


def test_session_blocks_entry(eurusd):
    m1 = make_m1("2024-01-02 23:45", flat_then([1.1000] * 15))
    cfg = BacktestConfig(respect_session=True)
    res = run(m1, {0: (1, 1.0900, np.nan)}, eurusd, cfg=cfg)  # would enter at 00:00 < 00:05
    assert res.trades.empty and res.skipped["session_or_news"] == 1


def test_daily_low_includes_losing_trade(eurusd):
    path = np.linspace(1.1000, 1.0950, 15)
    m1 = make_m1("2024-01-02 10:00", flat_then(path))
    res = run(m1, {0: (1, 1.0980, np.nan)}, eurusd)
    day = res.daily.loc["2024-01-02"]
    assert day.pnl_usd == pytest.approx(res.trades.iloc[0].pnl_usd)
    assert day.daily_low_usd <= day.pnl_usd + 1e-9

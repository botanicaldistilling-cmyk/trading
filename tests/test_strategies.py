"""Every candidate must be causal: signals at bar k may only use bars <= k."""
import numpy as np
import pandas as pd
import pytest

from tradelab.backtest import BacktestConfig, run_backtest
from tradelab.config import load_instrument
from tradelab.data.store import resample
from tradelab.data.synthetic import random_walk_m1
from tradelab.splits import param_grid
from tradelab.strategies import CANDIDATES

M1 = random_walk_m1(start="2021-01-04", days=40, price=100.0, vol_per_min=0.0004, spread=0.01, seed=3)


@pytest.mark.parametrize("cls,symbol", CANDIDATES)
def test_no_lookahead(cls, symbol):
    strat = cls(symbol=symbol)
    bars = resample(M1, strat.timeframe)
    full = strat.signals(bars)
    for k in (len(bars) // 3, len(bars) // 2, len(bars) - 5):
        part = strat.signals(bars.iloc[:k])
        np.testing.assert_array_equal(part["entry"].to_numpy(), full["entry"].to_numpy()[:k])
        on = full["entry"].to_numpy()[:k] != 0
        np.testing.assert_allclose(part["stop"].to_numpy()[on], full["stop"].to_numpy()[:k][on])


@pytest.mark.parametrize("cls,symbol", CANDIDATES)
def test_every_entry_has_a_valid_stop_and_grid_runs(cls, symbol):
    inst = load_instrument("EURUSD")
    for p in param_grid(cls.GRID):
        strat = cls(symbol=symbol, params=p)
        sig = strat.signals(resample(M1, strat.timeframe))
        e = sig["entry"] != 0
        assert sig.loc[e, "stop"].notna().all()
        # stop on the losing side of the signal close
        c = resample(M1, strat.timeframe)["close"]
        assert ((c[e] - sig.loc[e, "stop"]) * sig.loc[e, "entry"] > 0).all()
    res = run_backtest(cls(symbol=symbol), M1, inst, BacktestConfig(respect_session=False))
    assert res.trades.empty or res.trades["r"].notna().all()

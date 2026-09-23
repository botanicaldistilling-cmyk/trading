from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from tradelab.backtest.strategy import Strategy
from tradelab.config import load_instrument


@pytest.fixture
def eurusd():
    return load_instrument("EURUSD")


def make_m1(start: str, closes, spread=0.0001, wick=0.0):
    """M1 bars whose open is the previous close; optional symmetric wick."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + wick,
        "low": np.minimum(opens, closes) - wick,
        "close": closes,
        "spread": spread,
    }, index=idx)


@dataclass
class FixedSignals(Strategy):
    """Emit given signals at given bar numbers: {bar: (side, stop, target)}."""
    plan: dict = field(default_factory=dict)
    exits: dict = field(default_factory=dict)   # {bar: "long"|"short"}
    name: str = "fixed"

    def signals(self, bars):
        n = len(bars)
        df = pd.DataFrame({"entry": np.zeros(n, int), "stop": np.nan, "target": np.nan,
                           "exit_long": False, "exit_short": False}, index=bars.index)
        for k, (side, stop, target) in self.plan.items():
            df.iloc[k, df.columns.get_loc("entry")] = side
            df.iloc[k, df.columns.get_loc("stop")] = stop
            df.iloc[k, df.columns.get_loc("target")] = target
        for k, which in self.exits.items():
            df.iloc[k, df.columns.get_loc(f"exit_{which}")] = True
        return df

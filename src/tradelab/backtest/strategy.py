"""Strategy interface.

A strategy looks only at completed bars of its own timeframe and returns a
frame aligned to those bars. Everything is decided at the bar's close and
acted on at the next bar's open, so there is no look-ahead.

Columns returned by `signals(bars)`:
    entry        +1 long, -1 short, 0 nothing
    stop         absolute stop price for that entry          (required when entry != 0)
    target       absolute target price, NaN for none
    exit_long    True -> close an open long at next bar open
    exit_short   True -> close an open short at next bar open
    trail_long   level the long stop may ratchet up to (NaN = no change)
    trail_short  level the short stop may ratchet down to
Missing optional columns are treated as empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    symbol: str
    timeframe: str = "1h"                 # pandas offset alias: "15min", "1h"
    params: dict = field(default_factory=dict)
    max_hold_bars: int | None = None      # time exit
    name: str = "strategy"

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    prev = bars["close"].shift()
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


@dataclass
class RandomEntry(Strategy):
    """Baseline with no edge: random side, ATR stop, fixed-R target.

    Not a candidate strategy. Its result shows what costs alone do to a
    zero-edge system, which is the bar every real strategy must clear.
    """
    name: str = "random_baseline"

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = {"prob": 0.02, "atr_n": 14, "stop_atr": 1.5, "target_r": 1.5, "seed": 1} | self.params
        rng = np.random.default_rng(p["seed"])
        a = atr(bars, p["atr_n"])
        fire = rng.random(len(bars)) < p["prob"]
        side = np.where(rng.random(len(bars)) < 0.5, 1, -1) * fire
        side = np.where(a.notna(), side, 0)
        stop = bars["close"] - side * p["stop_atr"] * a
        target = bars["close"] + side * p["stop_atr"] * p["target_r"] * a
        return pd.DataFrame({"entry": side, "stop": stop, "target": target}, index=bars.index)

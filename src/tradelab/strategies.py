"""Stage 2 candidate strategies, pre-registered.

These definitions and their parameter grids were written and committed BEFORE
any real market data was loaded. They will not be changed after seeing
results; a strategy that fails the out-of-sample rules is dropped.

All times are server time (EET/EEST, New York + 7h). Every entry has a stop.
Each strategy has 2-3 parameters, each with 2-3 values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest.strategy import Strategy, atr


def _empty(bars: pd.DataFrame) -> pd.DataFrame:
    n = len(bars)
    return pd.DataFrame({
        "entry": np.zeros(n, dtype=int), "stop": np.nan, "target": np.nan,
        "exit_long": False, "exit_short": False, "trail_long": np.nan, "trail_short": np.nan,
    }, index=bars.index)


def _first_per_day(mask: pd.Series) -> pd.Series:
    """Keep only the first True of each server day."""
    day = mask.index.normalize()
    first = mask & (mask.astype(int).groupby(day).cumsum() == 1)
    return first


def _hhmm(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour * 100 + idx.minute


# --------------------------------------------------------------------------
# A. Trend following: Donchian breakout with trend filter (XAUUSD, H1)
# --------------------------------------------------------------------------
@dataclass
class DonchianTrend(Strategy):
    """Buy a close above the N-bar high while above the 200 EMA (shorts mirrored).

    Stop: stop_atr x ATR(14) from the signal close. Exit: trailing stop at the
    M-bar low (long) / high (short). No fixed target: let trends run.
    """
    name: str = "A_donchian_trend"
    timeframe: str = "1h"

    GRID = {"entry_n": [20, 40, 55], "exit_n": [10, 20], "stop_atr": [2.0, 3.0]}

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = {"entry_n": 40, "exit_n": 20, "stop_atr": 2.0, "ema_n": 200} | self.params
        c = bars["close"]
        hi = bars["high"].rolling(p["entry_n"]).max().shift(1)
        lo = bars["low"].rolling(p["entry_n"]).min().shift(1)
        ema = c.ewm(span=p["ema_n"], adjust=False, min_periods=p["ema_n"]).mean()
        a = atr(bars, 14)
        out = _empty(bars)
        long_ = (c > hi) & (c > ema)
        short = (c < lo) & (c < ema)
        out["entry"] = np.where(long_, 1, np.where(short, -1, 0))
        out["stop"] = c - out["entry"] * p["stop_atr"] * a
        out["trail_long"] = bars["low"].rolling(p["exit_n"]).min()
        out["trail_short"] = bars["high"].rolling(p["exit_n"]).max()
        out.loc[a.isna() | ema.isna(), "entry"] = 0
        return out


# --------------------------------------------------------------------------
# B. Mean reversion: Bollinger fade in the quiet Asian session (EURUSD, M15)
# --------------------------------------------------------------------------
@dataclass
class AsianBollingerFade(Strategy):
    """Fade closes outside Bollinger(20, k) during 02:00-08:45 server time.

    Target: the 20-bar mean at signal time. Stop: stop_atr x ATR(14) beyond
    the entry. Time exit after 16 bars (4 hours), and flat by 10:00 server
    (London open) whatever happens.
    """
    name: str = "B_asian_bollinger_fade"
    timeframe: str = "15min"
    max_hold_bars: int | None = 16

    GRID = {"band_k": [2.0, 2.5], "stop_atr": [1.0, 1.5, 2.0]}

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = {"band_k": 2.0, "stop_atr": 1.5, "n": 20} | self.params
        c = bars["close"]
        mid = c.rolling(p["n"]).mean()
        sd = c.rolling(p["n"]).std()
        a = atr(bars, 14)
        t = _hhmm(bars.index)
        window = (t >= 200) & (t <= 845)
        out = _empty(bars)
        long_ = window & (c < mid - p["band_k"] * sd)
        short = window & (c > mid + p["band_k"] * sd)
        out["entry"] = np.where(long_, 1, np.where(short, -1, 0))
        out["stop"] = c - out["entry"] * p["stop_atr"] * a
        out["target"] = mid
        flat = t >= 945                      # exit at the 10:00 bar open
        out["exit_long"] = flat
        out["exit_short"] = flat
        out.loc[a.isna() | sd.isna(), "entry"] = 0
        return out


# --------------------------------------------------------------------------
# C. Opening-range breakout at the New York cash open (NAS100 / US30, M15)
# --------------------------------------------------------------------------
@dataclass
class NYOpeningRange(Strategy):
    """Range = first `range_bars` M15 bars from 16:30 server (09:30 New York).

    First M15 close outside the range before 20:00 server -> enter that way.
    Stop: the other side of the range. Target: target_r x risk (0 = none).
    One trade per day, flat at 22:45 server. Uses the range, not ATR, so the
    stop adapts to each day's opening volatility.
    """
    name: str = "C_ny_opening_range"
    timeframe: str = "15min"

    GRID = {"range_bars": [2, 4], "target_r": [0.0, 1.5, 2.0]}

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = {"range_bars": 2, "target_r": 1.5} | self.params
        t = _hhmm(bars.index)
        day = bars.index.normalize()
        rb = int(p["range_bars"])
        end_min = 16 * 60 + 30 + rb * 15
        end_hhmm = (end_min // 60) * 100 + end_min % 60
        in_range = (t >= 1630) & (t < end_hhmm)
        rng_hi = bars["high"].where(in_range).groupby(day).transform("max")
        rng_lo = bars["low"].where(in_range).groupby(day).transform("min")
        n_range = pd.Series(in_range, index=bars.index).groupby(day).transform("sum")
        c = bars["close"]
        window = (t >= end_hhmm) & (t < 2000) & (n_range == rb).to_numpy()
        brk_up = window & (c > rng_hi)
        brk_dn = window & (c < rng_lo)
        first = _first_per_day(brk_up | brk_dn)
        out = _empty(bars)
        out["entry"] = np.where(first & brk_up, 1, np.where(first & brk_dn, -1, 0))
        out["stop"] = np.where(out["entry"] > 0, rng_lo, np.where(out["entry"] < 0, rng_hi, np.nan))
        if p["target_r"] > 0:
            risk = (c - out["stop"]).abs()
            out["target"] = c + out["entry"] * p["target_r"] * risk
        flat = t >= 2230                     # exit at the 22:45 bar open
        out["exit_long"] = flat
        out["exit_short"] = flat
        return out


# --------------------------------------------------------------------------
# D. London breakout of the Asian range (GBPJPY, M15)
# --------------------------------------------------------------------------
@dataclass
class LondonBreakout(Strategy):
    """Asian range = 02:00-09:45 server. First M15 close outside it between
    10:00 and 13:45 server -> enter. Stop: opposite side of the range, or the
    range midpoint. Target: target_r x risk. One trade per day, flat at 22:00.
    Days whose Asian range is wider than 1.5x ATR(96 M15 bars) are skipped
    (the move has already happened).
    """
    name: str = "D_london_breakout"
    timeframe: str = "15min"

    GRID = {"stop_at": ["opposite", "mid"], "target_r": [1.0, 1.5, 2.0]}

    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = {"stop_at": "opposite", "target_r": 1.5} | self.params
        t = _hhmm(bars.index)
        day = bars.index.normalize()
        asia = (t >= 200) & (t <= 945)
        hi = bars["high"].where(asia).groupby(day).transform("max")
        lo = bars["low"].where(asia).groupby(day).transform("min")
        width = hi - lo
        daily_atr = atr(bars, 96) * np.sqrt(96)  # rough day-scale volatility
        ok_width = width <= 1.5 * daily_atr
        c = bars["close"]
        window = (t >= 1000) & (t <= 1345)
        brk_up = window & (c > hi) & ok_width
        brk_dn = window & (c < lo) & ok_width
        first = _first_per_day(brk_up | brk_dn)
        out = _empty(bars)
        side = np.where(first & brk_up, 1, np.where(first & brk_dn, -1, 0))
        out["entry"] = side
        if p["stop_at"] == "mid":
            stop = (hi + lo) / 2
        else:
            stop = pd.Series(np.where(side > 0, lo, hi), index=bars.index)
        out["stop"] = np.where(side != 0, stop, np.nan)
        risk = (c - out["stop"]).abs()
        out["target"] = c + side * p["target_r"] * risk
        flat = t >= 2145
        out["exit_long"] = flat
        out["exit_short"] = flat
        out.loc[daily_atr.isna(), "entry"] = 0
        return out


# Which market each candidate is tested on (fixed in advance).
CANDIDATES = [
    (DonchianTrend, "XAUUSD"),
    (AsianBollingerFade, "EURUSD"),
    (NYOpeningRange, "NAS100"),
    (NYOpeningRange, "US30"),
    (LondonBreakout, "GBPJPY"),
]

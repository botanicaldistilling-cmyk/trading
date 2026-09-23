"""High-impact news blackout.

The calendar CSV (from the MT5 export script, or any other source) needs:
    time      server time of the release
    currency  e.g. USD, EUR
    impact    "high" rows are used; others ignored
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Which currencies' news affects which instrument.
CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPJPY": ["GBP", "JPY"],
    "XAUUSD": ["USD"],
    "NAS100": ["USD"],
    "USOIL": ["USD"],
    "BTCUSD": ["USD"],
}


class NewsCalendar:
    def __init__(self, events: pd.DataFrame):
        ev = events.copy()
        ev["time"] = pd.to_datetime(ev["time"])
        ev = ev[ev["impact"].str.lower() == "high"]
        self._times = {
            ccy: np.sort(g["time"].to_numpy(dtype="datetime64[ns]")) for ccy, g in ev.groupby("currency")
        }

    @classmethod
    def from_csv(cls, path: Path) -> "NewsCalendar":
        return cls(pd.read_csv(path))

    @classmethod
    def empty(cls) -> "NewsCalendar":
        return cls(pd.DataFrame(columns=["time", "currency", "impact"]))

    def blocked(self, times: pd.DatetimeIndex, symbol: str, before_min: float, after_min: float) -> np.ndarray:
        """True where an order at `times` would fall inside [event - before, event + after]."""
        t = np.asarray(times, dtype="datetime64[ns]")
        out = np.zeros(len(t), dtype=bool)
        before = np.timedelta64(int(before_min * 60e9), "ns")
        after = np.timedelta64(int(after_min * 60e9), "ns")
        for ccy in CURRENCIES.get(symbol, ["USD"]):
            ev = self._times.get(ccy)
            if ev is None or len(ev) == 0:
                continue
            # nearest event at or after t, and nearest at or before t
            i = np.searchsorted(ev, t, side="left")
            nxt = ev[np.minimum(i, len(ev) - 1)]
            prv = ev[np.maximum(i - 1, 0)]
            out |= (i < len(ev)) & (nxt - t <= before)
            out |= (i > 0) & (t - prv <= after)
            out |= (i < len(ev)) & (nxt == t)
        return out

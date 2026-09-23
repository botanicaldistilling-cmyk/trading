"""Time zone handling.

MT5 servers like The5ers' run on EET/EEST, arranged so that 00:00 server time
is 17:00 New York (the FX daily rollover). That makes server time exactly
New York time + 7 hours all year round, which is how we compute it.
The firm's daily loss reset happens at server midnight.
"""
from __future__ import annotations

import pandas as pd

SERVER_OFFSET = pd.Timedelta(hours=7)


def utc_to_server(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC (aware or naive) -> naive server-time index."""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ny = idx.tz_convert("America/New_York").tz_localize(None)
    return ny + SERVER_OFFSET


def server_to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive server-time index -> aware UTC index."""
    ny = (idx - SERVER_OFFSET).tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    )
    return ny.tz_convert("UTC")


def minutes_of_day(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

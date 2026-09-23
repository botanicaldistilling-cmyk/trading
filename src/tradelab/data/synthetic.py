"""Synthetic M1 data for tests and plumbing checks (no edge by construction)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_m1(start: str = "2020-01-01", days: int = 60, price: float = 1.10,
                   vol_per_min: float = 0.00008, spread: float = 0.00008, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min")
    idx = idx[idx.dayofweek < 5]  # server-time weekdays only
    n = len(idx)
    rets = rng.normal(0, vol_per_min, n)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol_per_min * price * 0.6, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "spread": np.full(n, spread)},
        index=idx,
    )

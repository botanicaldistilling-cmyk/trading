"""Statistical character of a market at a given horizon.

Used as a sanity check on a strategy's premise (does this market actually
mean-revert or trend at the horizon the strategy trades?), never to pick
parameters. Run on IN-SAMPLE data only.

  variance ratio VR(k) = Var(k-bar return) / (k * Var(1-bar return))
      VR < 1: returns partly cancel (mean reversion); VR > 1: they persist (trend).
      z-score uses the Lo-MacKinlay homoskedastic standard error.
  Hurst exponent: slope of log std(x[t+lag] - x[t]) vs log lag. 0.5 = random walk.
  half-life: from regressing the change on the previous level (AR(1)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(log_price: pd.Series, k: int) -> tuple[float, float]:
    """(VR, z). Uses only moves inside the given bars (gaps across NaN are skipped)."""
    x = log_price.to_numpy(dtype=float)
    r1 = np.diff(x)
    r1 = r1[np.isfinite(r1)]
    rk = x[k:] - x[:-k]
    rk = rk[np.isfinite(rk)]
    n = len(r1)
    if n < 10 * k or len(rk) < 10:
        return float("nan"), float("nan")
    vr = np.var(rk, ddof=1) / (k * np.var(r1, ddof=1))
    se = np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * n))
    return float(vr), float((vr - 1) / se)


def hurst(log_price: pd.Series, max_lag: int = 100) -> float:
    x = log_price.dropna().to_numpy(dtype=float)
    lags = np.arange(2, min(max_lag, len(x) // 4))
    if len(lags) < 5:
        return float("nan")
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def half_life(series: pd.Series) -> float:
    y = series.dropna()
    ylag, dy = y.shift(1).iloc[1:].to_numpy(), y.diff().iloc[1:].to_numpy()
    lam = np.polyfit(ylag, dy, 1)[0]
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def session_variance_ratios(bars: pd.DataFrame, sessions: dict[str, tuple[int, int]],
                            ks=(2, 4, 8)) -> pd.DataFrame:
    """VR of log closes inside each session window (hhmm server time), per day.

    Each day's session is treated as its own segment so overnight gaps and
    other sessions don't leak in.
    """
    t = bars.index.hour * 100 + bars.index.minute
    lp = np.log(bars["close"])
    rows = []
    for name, (start, end) in sessions.items():
        mask = np.asarray((t >= start) & (t <= end))
        day = bars.index.normalize()
        row = {"session": name, "bars": int(mask.sum())}
        for k in ks:
            vr, z = _segmented_vr(lp, mask, day, k)
            row[f"VR({k})"], row[f"z({k})"] = vr, z
        rows.append(row)
    return pd.DataFrame(rows)


def _segmented_vr(lp: pd.Series, mask: np.ndarray, day, k: int) -> tuple[float, float]:
    r1s, rks = [], []
    df = pd.DataFrame({"lp": lp.to_numpy(), "day": day, "m": mask})
    for _, g in df[df["m"]].groupby("day"):
        x = g["lp"].to_numpy()
        if len(x) > k:
            r1s.append(np.diff(x))
            rks.append(x[k:] - x[:-k])
    if not r1s:
        return float("nan"), float("nan")
    r1, rk = np.concatenate(r1s), np.concatenate(rks)
    n = len(r1)
    if n < 10 * k:
        return float("nan"), float("nan")
    vr = np.var(rk, ddof=1) / (k * np.var(r1, ddof=1))
    se = np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * n))
    return float(vr), float((vr - 1) / se)

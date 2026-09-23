"""Performance statistics reported for every strategy and every data split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest.engine import BacktestResult


def summarize(res: BacktestResult, start=None, end=None) -> dict:
    """Stats on trades entered in [start, end]. R-multiples include all costs."""
    t = res.trades
    d = res.daily
    if start is not None:
        t = t[t["entry_time"] >= pd.Timestamp(start)] if not t.empty else t
        d = d[d.index >= pd.Timestamp(start)]
    if end is not None:
        t = t[t["entry_time"] < pd.Timestamp(end) + pd.Timedelta(days=1)] if not t.empty else t
        d = d[d.index <= pd.Timestamp(end)]
    initial = res.config.initial_balance if res.config else 5000.0
    if t.empty:
        return {"trades": 0}
    r = t["r"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    eq = np.cumsum(r)
    dd_r = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    bal = initial + d["pnl_usd"].cumsum()
    peak = np.maximum.accumulate(np.r_[initial, bal.to_numpy()])[1:]
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9) if len(d) else np.nan
    return {
        "trades": int(len(r)),
        "trades_per_month": len(r) / (years * 12) if years == years else np.nan,
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "avg_r_pre_comm": float(t["r_pre_comm"].mean()),
        "median_r": float(np.median(r)),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "total_r": float(r.sum()),
        "max_dd_r": float(dd_r),
        "max_dd_pct": float(((peak - bal.to_numpy()) / initial).max() * 100) if len(bal) else 0.0,
        "worst_day_pct": float(d["pnl_pct"].min()) if len(d) else 0.0,
        "worst_intraday_pct": float(d["daily_low_pct"].min()) if len(d) else 0.0,
        "max_losing_streak": int(_longest_run(r <= 0)),
        "avg_bars_held": float(t["bars_held"].mean()),
        "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 1 and r.std() > 0 else 0.0,
    }


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def compare_table(rows: dict[str, dict]) -> pd.DataFrame:
    """Side-by-side table, e.g. {"in-sample": summarize(...), "out-of-sample": ...}."""
    cols = ["trades", "trades_per_month", "win_rate", "avg_r", "avg_r_pre_comm", "profit_factor",
            "total_r", "max_dd_r", "max_dd_pct", "worst_day_pct", "worst_intraday_pct",
            "max_losing_streak", "t_stat"]
    df = pd.DataFrame(rows).T
    return df[[c for c in cols if c in df.columns]]


def format_table(df: pd.DataFrame) -> str:
    fmt = {
        "trades": "{:.0f}", "trades_per_month": "{:.1f}", "win_rate": "{:.1%}", "avg_r": "{:+.3f}",
        "avg_r_pre_comm": "{:+.3f}", "profit_factor": "{:.2f}", "total_r": "{:+.1f}", "max_dd_r": "{:.1f}",
        "max_dd_pct": "{:.1f}%", "worst_day_pct": "{:.2f}%", "worst_intraday_pct": "{:.2f}%",
        "max_losing_streak": "{:.0f}", "t_stat": "{:+.2f}",
    }
    out = df.copy().astype(object)
    for c, f in fmt.items():
        if c in out:
            out[c] = [f.format(v) if pd.notna(v) else "-" for v in df[c]]
    return out.to_string()

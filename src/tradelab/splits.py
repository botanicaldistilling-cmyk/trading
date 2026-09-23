"""In-sample / out-of-sample / holdout periods and walk-forward testing.

Rules (fixed before looking at any results):
  * Parameters are chosen on IN-SAMPLE data only.
  * OUT-OF-SAMPLE is used to accept or reject a strategy, once.
  * HOLDOUT is untouched until the final portfolio check at the end.
  * Walk-forward re-picks parameters on a rolling training window and
    stitches together the following, never-seen test windows.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import pandas as pd

from .backtest.engine import BacktestConfig, BacktestResult, run_backtest
from .backtest.strategy import Strategy
from .config import Instrument
from .metrics import summarize


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str


DEFAULT_PERIODS = (
    Period("in_sample", "2018-01-01", "2022-12-31"),
    Period("out_of_sample", "2023-01-01", "2025-12-31"),
    Period("holdout", "2026-01-01", "2026-12-31"),
)

# Acceptance rules for a strategy, applied to OUT-OF-SAMPLE results.
ACCEPTANCE = {
    "min_trades": 100,
    "min_profit_factor": 1.15,
    "min_oos_to_is_avg_r": 0.5,   # OOS avg R must be at least half the IS avg R
}


def walk_forward_windows(start: str, end: str, train_years: int = 3, test_years: int = 1):
    """Yield (train_start, train_end, test_start, test_end) as Timestamps."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    k = 0
    while True:
        tr_s = s + pd.DateOffset(years=k * test_years)
        tr_e = tr_s + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        te_s = tr_e + pd.Timedelta(days=1)
        te_e = te_s + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if te_s > e:
            break
        yield tr_s, tr_e, te_s, min(te_e, e)
        k += 1


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def objective(stats: dict, min_trades: int = 30) -> float:
    """Score used to pick parameters in training windows: avg R, penalised for few trades."""
    if stats.get("trades", 0) < min_trades:
        return float("-inf")
    return stats["avg_r"] * min(1.0, stats["trades"] / 100) ** 0.5


def walk_forward(strategy: Strategy, grid: dict[str, list], m1: pd.DataFrame, inst: Instrument,
                 cfg: BacktestConfig, start: str, end: str, train_years: int = 3,
                 test_years: int = 1, results: dict | None = None) -> dict:
    """Pick the best params per training window, record the next test window's trades.

    Every parameter set is backtested once over the full range; windows are then
    scored by slicing the trade list, which is equivalent because signals only
    use past bars (indicator warm-up apart).
    """
    combos = param_grid(grid)
    if results is None:  # {combo index: BacktestResult}, may be passed in to avoid re-running
        results = {}
        for k, p in enumerate(combos):
            strat = replace(strategy, params=strategy.params | p)
            results[k] = run_backtest(strat, m1, inst, cfg)

    windows, oos_trades = [], []
    for tr_s, tr_e, te_s, te_e in walk_forward_windows(start, end, train_years, test_years):
        scores = {k: objective(summarize(r, tr_s, tr_e)) for k, r in results.items()}
        best = max(scores, key=scores.get)
        test = summarize(results[best], te_s, te_e)
        windows.append({"train": f"{tr_s:%Y-%m-%d}..{tr_e:%Y-%m-%d}",
                        "test": f"{te_s:%Y-%m-%d}..{te_e:%Y-%m-%d}",
                        "params": combos[best], "train_score": scores[best],
                        "test_trades": test.get("trades", 0), "test_avg_r": test.get("avg_r"),
                        "test_pf": test.get("profit_factor")})
        t = results[best].trades
        if not t.empty:
            oos_trades.append(t[(t["entry_time"] >= te_s) & (t["entry_time"] < te_e + pd.Timedelta(days=1))])
    stitched = pd.concat(oos_trades) if oos_trades else pd.DataFrame()
    return {"windows": pd.DataFrame(windows), "oos_trades": stitched, "all_results": results,
            "combos": combos}


def accept(is_stats: dict, oos_stats: dict, rules: dict = ACCEPTANCE) -> tuple[bool, list[str]]:
    """Apply the pre-registered acceptance rules. Returns (passed, reasons for failure)."""
    why = []
    if oos_stats.get("trades", 0) < rules["min_trades"]:
        why.append(f"only {oos_stats.get('trades', 0)} OOS trades (< {rules['min_trades']})")
    pf = oos_stats.get("profit_factor", 0)
    if pf < rules["min_profit_factor"]:
        why.append(f"OOS profit factor {pf:.2f} < {rules['min_profit_factor']}")
    is_r, oos_r = is_stats.get("avg_r", 0), oos_stats.get("avg_r", 0)
    if oos_r <= 0:
        why.append(f"OOS avg R {oos_r:+.3f} is not positive")
    elif is_r > 0 and oos_r < rules["min_oos_to_is_avg_r"] * is_r:
        why.append(f"OOS avg R {oos_r:+.3f} < {rules['min_oos_to_is_avg_r']:.0%} of IS {is_r:+.3f}")
    return (not why), why

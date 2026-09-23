"""Plumbing check on synthetic random-walk data (no market data needed).

Two checks:
  * zero-cost run: avg R must be about 0 (t-stat within about +/-2). A clearly
    positive result would mean the engine leaks future information.
  * realistic-cost run: avg R is negative by the cost drag. That drag, in R,
    is what any real strategy must beat.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradelab.backtest import BacktestConfig, run_backtest  # noqa: E402
from tradelab.backtest.strategy import RandomEntry  # noqa: E402
from tradelab.config import load_instrument  # noqa: E402
from tradelab.data.synthetic import random_walk_m1  # noqa: E402
from tradelab.metrics import compare_table, format_table, summarize  # noqa: E402


def main():
    inst = load_instrument("EURUSD")
    m1 = random_walk_m1(start="2020-01-01", days=730, spread=0.00008, seed=7)
    t0 = time.time()
    free = replace(inst, commission_value=0.0, slip_entry_points=0, slip_stop_points=0)
    m1_free = m1.assign(spread=0.0)
    rows = {}
    for seed in range(3):
        strat = RandomEntry(symbol="EURUSD", timeframe="15min", params={"seed": seed})
        rows[f"no costs, seed {seed}"] = summarize(run_backtest(strat, m1_free, free, BacktestConfig()))
        rows[f"with costs, seed {seed}"] = summarize(run_backtest(strat, m1, inst, BacktestConfig()))
    print(f"{len(m1):,} M1 bars, 6 runs in {time.time() - t0:.1f}s")
    print(format_table(compare_table(rows)))


if __name__ == "__main__":
    main()

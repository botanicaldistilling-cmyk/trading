"""Run the zero-edge random baseline on real data and print IS vs OOS stats.

    python scripts/run_baseline.py --symbol EURUSD --timeframe 15min

Shows how much the cost model alone costs per trade (avg R gross vs net).
A real strategy has to beat this by a clear margin.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradelab.backtest import BacktestConfig, run_backtest  # noqa: E402
from tradelab.backtest.strategy import RandomEntry  # noqa: E402
from tradelab.config import load_instrument  # noqa: E402
from tradelab.data.store import load_m1  # noqa: E402
from tradelab.metrics import compare_table, format_table, summarize  # noqa: E402
from tradelab.splits import DEFAULT_PERIODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="15min")
    ap.add_argument("--risk", type=float, default=1.0)
    args = ap.parse_args()

    inst = load_instrument(args.symbol)
    m1 = load_m1(inst)
    res = run_backtest(RandomEntry(symbol=inst.name, timeframe=args.timeframe), m1, inst,
                       BacktestConfig(risk_pct=args.risk))
    rows = {p.name: summarize(res, p.start, p.end) for p in DEFAULT_PERIODS}
    print(format_table(compare_table(rows)))
    print("skipped:", res.skipped)


if __name__ == "__main__":
    main()

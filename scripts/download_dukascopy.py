"""Download M1 bid/ask history from Dukascopy into data/m1/.

    python scripts/download_dukascopy.py --symbols EURUSD XAUUSD NAS100 --start 2018 --end 2026

Re-running resumes: raw daily files are cached under data/raw/.
Also downloads USDJPY automatically when GBPJPY is requested (JPY -> USD conversion).
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradelab.config import Instrument, load_instruments  # noqa: E402
from tradelab.data.dukascopy import download_year  # noqa: E402

USDJPY = Instrument(
    name="USDJPY", mt5_symbol="USDJPY", contract_size=100000, quote_ccy="JPY", point=0.001,
    min_lot=0.01, lot_step=0.01, commission_type="per_lot_rt", commission_value=4.0,
    spread_markup_points=0, slip_entry_points=0, slip_stop_points=0, swap_long=0, swap_short=0,
    swap_triple_day="wednesday", swap_weekend_multiplier=1, session_open="00:05", session_close="23:55",
    dukascopy_code="USDJPY", dukascopy_divisor=1000, sane_range=(70, 200),
)


def main():
    insts = load_instruments()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPJPY", "XAUUSD", "NAS100"])
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    todo = [insts[s] for s in args.symbols]
    if "GBPJPY" in args.symbols:
        todo.append(USDJPY)
    for inst in todo:
        for year in range(args.start, args.end + 1):
            t0 = time.time()
            path = download_year(inst, year, workers=args.workers)
            print(f"{inst.name} {year}: {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

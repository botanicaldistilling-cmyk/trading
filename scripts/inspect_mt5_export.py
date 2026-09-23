"""Summarise the files written by mql5/Scripts/ExportResearchData.mq5.

    python scripts/inspect_mt5_export.py path/to/MQL5/Files/research

Prints the broker's specs for our instruments (to fill config/instruments.yaml),
the typical spread by server hour from the MT5 M1 bars, and, when Dukascopy
data exists too, the same spread from Dukascopy so the markup can be set.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradelab.config import load_instruments  # noqa: E402
from tradelab.data.store import load_m1, load_mt5_csv  # noqa: E402


def main():
    folder = Path(sys.argv[1])
    insts = load_instruments()
    specs = pd.read_csv(folder / "symbols.csv")
    for name, inst in insts.items():
        key = name.rstrip("0123456789")[:3] if name != "NAS100" else "NAS"
        hits = specs[specs["symbol"].str.upper().str.contains(key[:3])]
        print(f"\n=== {name}: candidate broker symbols ===")
        cols = ["symbol", "digits", "contract_size", "volume_min", "volume_step", "swap_mode",
                "swap_long", "swap_short", "swap_3day", "margin_1lot_buy", "spread_now"]
        print(hits[cols].head(8).to_string(index=False) if len(hits) else "  none found")

        bars_file = folder / f"bars_{inst.mt5_symbol}_M1.csv"
        if bars_file.exists():
            mt5 = load_mt5_csv(bars_file, inst)
            print(f"MT5 M1 history: {mt5.index[0]} .. {mt5.index[-1]} ({len(mt5):,} bars)")
            by_hour = (mt5["spread"] / inst.point).groupby(mt5.index.hour).median()
            out = pd.DataFrame({"mt5_pts": by_hour})
            try:
                duk = load_m1(inst, start=str(mt5.index[0].date()))
                out["dukascopy_pts"] = (duk["spread"] / inst.point).groupby(duk.index.hour).median()
            except FileNotFoundError:
                pass
            print("median spread by server hour (points):")
            print(out.round(1).T.to_string())

    cal = folder / "calendar_high.csv"
    if cal.exists():
        ev = pd.read_csv(cal, parse_dates=["time"])
        print(f"\ncalendar: {len(ev)} high-impact events {ev.time.min()} .. {ev.time.max()}")
        print(ev.groupby("currency").size().to_string())


if __name__ == "__main__":
    main()

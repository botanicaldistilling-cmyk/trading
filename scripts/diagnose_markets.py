"""Check each strategy's premise on IN-SAMPLE data (2018-2022) only.

    python scripts/diagnose_markets.py

For each market: variance ratios of log closes inside the session the
strategy trades, and over the whole day. VR < 1 with z < -2 means moves
tend to reverse at that horizon (supports mean reversion); VR > 1 with
z > 2 means they tend to continue (supports breakout/trend).
This does not choose any parameter. Output: reports/diagnostics.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tradelab.config import load_instrument  # noqa: E402
from tradelab.data.store import load_m1, resample  # noqa: E402
from tradelab.diagnostics import hurst, session_variance_ratios  # noqa: E402
from tradelab.splits import DEFAULT_PERIODS  # noqa: E402

IS = DEFAULT_PERIODS[0]
# market -> (timeframe, sessions to test in server time hhmm, premise)
PLAN = {
    "EURUSD": ("15min", {"asian 02:00-08:45 (B)": (200, 845), "london 10:00-16:00": (1000, 1600),
                         "all day": (0, 2359)}, "B expects reversion (VR<1) in the Asian session"),
    "NAS100": ("15min", {"NY open 16:30-20:00 (C)": (1630, 2000), "all day": (0, 2359)},
               "C expects continuation (VR>1) after the NY open"),
    "US30": ("15min", {"NY open 16:30-20:00 (C)": (1630, 2000), "all day": (0, 2359)},
             "C expects continuation (VR>1) after the NY open"),
    "GBPJPY": ("15min", {"london 10:00-13:45 (D)": (1000, 1345), "all day": (0, 2359)},
               "D expects continuation (VR>1) at the London open"),
    "XAUUSD": ("1h", {"all day (A)": (0, 2359)}, "A expects trending (VR>1, Hurst>0.5) on H1"),
}


def main():
    lines = ["# Market diagnostics (in-sample only)", "",
             f"Period {IS.start}..{IS.end}. VR(k) compares k-bar moves with k single bars; "
             "|z| > 2 is roughly significant. Overlapping returns make z somewhat optimistic.", ""]
    for sym, (tf, sessions, premise) in PLAN.items():
        try:
            m1 = load_m1(load_instrument(sym), start=IS.start, end=IS.end)
        except FileNotFoundError:
            print(f"{sym}: no data, skipped")
            continue
        bars = resample(m1, tf)
        vr = session_variance_ratios(bars, sessions)
        h = hurst(np.log(bars["close"]))
        lines += [f"## {sym} ({tf})", "", f"Premise: {premise}. Hurst (whole series): {h:.3f}", "",
                  "```", vr.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", ""]
        print(f"{sym} done")
    out = ROOT / "reports" / "diagnostics.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

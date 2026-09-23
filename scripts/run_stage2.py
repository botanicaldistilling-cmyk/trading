"""Stage 2: test every pre-registered candidate and write reports/stage2.md.

    python scripts/run_stage2.py              # all candidates with data present
    python scripts/run_stage2.py --only NAS100

Procedure per candidate (fixed in advance, see src/tradelab/splits.py):
  1. Backtest every parameter set in the strategy's small GRID on 2018-2025.
     The 2026 holdout is not loaded at all.
  2. Pick ONE parameter set using in-sample (2018-2022) results only.
  3. Report that set's out-of-sample (2023-2025) results and apply the
     acceptance rules. No re-tuning after this point.
  4. Walk-forward (3y train / 1y test) as a second opinion, plus the spread
     of OOS results across the whole grid (is the edge a plateau or a spike?).
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradelab.backtest import BacktestConfig, run_backtest  # noqa: E402
from tradelab.config import load_challenge, load_instrument  # noqa: E402
from tradelab.data.dukascopy import load_compact  # noqa: E402
from tradelab.data.store import load_m1  # noqa: E402
from tradelab.metrics import compare_table, format_table, summarize  # noqa: E402
from tradelab.news import NewsCalendar  # noqa: E402
from tradelab.splits import DEFAULT_PERIODS, accept, objective, param_grid, walk_forward  # noqa: E402
from tradelab.strategies import CANDIDATES  # noqa: E402
from tradelab.timeutil import utc_to_server  # noqa: E402

IS, OOS, _HOLDOUT = DEFAULT_PERIODS
RESEARCH_END = OOS.end                      # holdout never loaded here
OUT = ROOT / "reports" / "stage2"


def usd_rate_for(inst):
    """Quote->USD conversion series (only needed for non-USD quotes)."""
    if inst.quote_ccy == "USD" or not inst.usd_conversion_symbol:
        return None
    files = sorted((ROOT / "data" / "m1" / inst.usd_conversion_symbol).glob("*.parquet"))
    if not files:
        print(f"  ! no {inst.usd_conversion_symbol} data, using fixed rate {inst.usd_fallback_rate}")
        return None
    px = pd.concat(load_compact(f)["close"] for f in files).sort_index()
    px.index = utc_to_server(pd.DatetimeIndex(px.index))
    px = px.resample("1h").last().dropna()
    return 1.0 / px if inst.usd_conversion_invert else px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="symbols to run")
    ap.add_argument("--risk", type=float, default=0.5, help="risk %% per trade used for sizing")
    args = ap.parse_args()

    ch = load_challenge()
    cal_file = ROOT / "data" / "mt5" / "calendar_high.csv"
    news = NewsCalendar.from_csv(cal_file) if cal_file.exists() and ch["news"]["enabled"] else None
    OUT.mkdir(parents=True, exist_ok=True)

    lines = ["# Stage 2 results", "",
             f"Risk per trade for sizing: {args.risk}% of $5,000. All R figures are after costs.",
             f"News blackout: {'on (MT5 calendar)' if news else 'OFF (no calendar file yet)'}.",
             f"In-sample {IS.start}..{IS.end}, out-of-sample {OOS.start}..{OOS.end}. Holdout not loaded.", ""]
    verdicts = []

    for cls, symbol in CANDIDATES:
        if args.only and symbol not in args.only:
            continue
        inst = load_instrument(symbol)
        try:
            m1 = load_m1(inst, start=IS.start, end=RESEARCH_END)
        except FileNotFoundError:
            print(f"{cls.name} {symbol}: no data, skipped")
            continue
        t0 = time.time()
        cfg = BacktestConfig(risk_pct=args.risk, news=news, usd_rate=usd_rate_for(inst),
                             news_before_min=ch["news"]["safety_minutes_before"],
                             news_after_min=ch["news"]["safety_minutes_after"])
        base = cls(symbol=symbol)
        combos = param_grid(cls.GRID)
        results = {k: run_backtest(replace(base, params=p), m1, inst, cfg) for k, p in enumerate(combos)}

        grid_rows = []
        for k, p in enumerate(combos):
            s_is, s_oos = summarize(results[k], IS.start, IS.end), summarize(results[k], OOS.start, OOS.end)
            grid_rows.append({**p, "is_trades": s_is.get("trades", 0), "is_avg_r": s_is.get("avg_r"),
                              "is_pf": s_is.get("profit_factor"), "oos_trades": s_oos.get("trades", 0),
                              "oos_avg_r": s_oos.get("avg_r"), "oos_pf": s_oos.get("profit_factor"),
                              "score": objective(s_is), "skipped_min_lot": results[k].skipped["min_lot"]})
        grid = pd.DataFrame(grid_rows)
        best = int(grid["score"].idxmax())
        chosen = results[best]
        s_is, s_oos = summarize(chosen, IS.start, IS.end), summarize(chosen, OOS.start, OOS.end)
        passed, why = accept(s_is, s_oos)
        wf = walk_forward(base, cls.GRID, m1, inst, cfg, IS.start, RESEARCH_END, results=results)
        wf_t = wf["oos_trades"]
        wf_line = (f"{len(wf_t)} trades, avg R {wf_t['r'].mean():+.3f}, "
                   f"PF {wf_t.loc[wf_t.r > 0, 'r'].sum() / -wf_t.loc[wf_t.r <= 0, 'r'].sum():.2f}"
                   if len(wf_t) and (wf_t.r <= 0).any() else "no trades")

        tag = f"{cls.name}_{symbol}"
        chosen.trades.to_csv(OUT / f"{tag}_trades.csv", index=False)
        chosen.daily.to_csv(OUT / f"{tag}_daily.csv")
        verdict = "KEEP" if passed else "DROP"
        verdicts.append((tag, verdict, "; ".join(why) or "meets all rules"))

        lines += [f"## {cls.name} on {symbol}: **{verdict}**", "",
                  (cls.__doc__ or "").strip().splitlines()[0], "",
                  f"Chosen on in-sample only: `{combos[best]}`", "",
                  "```", format_table(compare_table({"in-sample": s_is, "out-of-sample": s_oos})), "```", "",
                  f"Walk-forward stitched out-of-sample: {wf_line}", "",
                  "Whole grid (robustness: are neighbours similar?):", "",
                  "```", grid.drop(columns="score").to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", "",
                  f"Skipped trades (chosen set): {chosen.skipped}", "",
                  f"Verdict: {'; '.join(why) if why else 'meets all acceptance rules'}", ""]
        print(f"{tag}: {verdict} ({time.time() - t0:.0f}s) {'; '.join(why)}")

    lines[6:6] = ["| Strategy | Verdict | Reason |", "|---|---|---|",
                  *[f"| {t} | {v} | {w} |" for t, v, w in verdicts], ""]
    (ROOT / "reports" / "stage2.md").write_text("\n".join(lines))
    print("wrote reports/stage2.md")


if __name__ == "__main__":
    main()

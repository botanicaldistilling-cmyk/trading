# tradelab: prop-challenge strategy research

A small, honest pipeline for building a few simple, uncorrelated strategies,
testing them out-of-sample, simulating a The5ers challenge, and porting the
survivors to one MT5 Expert Advisor.

| Stage | What | Status |
|---|---|---|
| 1 | Data + backtester + IS/OOS/walk-forward framework | **done: waiting for real data** |
| 2 | 3–4 strategies, judged by pre-set acceptance rules | next |
| 3 | Portfolio, correlation, Monte Carlo of the challenge | |
| 4 | MQL5 EA with shared risk manager | |

## Challenge rules used (`config/challenge.yaml`)

5K account, 2 steps, +10% target each step, **daily loss 3%** of the higher
of start-of-day balance/equity, **max loss $300 static**, leverage 1:30,
no orders ±2 min around high-impact news. Our own internal stops are tighter:
stop for the day at -2%, stop entirely at -$250. Change them in the YAML.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/pytest                                 # engine tests
.venv/bin/python scripts/demo_synthetic.py       # plumbing / look-ahead check
```

## Getting data

**1. Dukascopy (free, many years of M1 bid+ask).**

```bash
.venv/bin/python scripts/download_dukascopy.py --symbols EURUSD GBPJPY XAUUSD NAS100 US30 --start 2018
```

Writes `data/m1/<SYMBOL>/<YEAR>.parquet` (UTC), about 10–20 MB per symbol per year.
Re-running resumes from the cache in `data/raw/`. Expect roughly 10–30
minutes per symbol.

**2. Your MT5 terminal (exact broker specs, spreads, news calendar).**

1. Copy `mql5/Scripts/ExportResearchData.mq5` into your terminal's
   `MQL5/Scripts/` folder (File > Open Data Folder), open it in MetaEditor and compile (F7).
2. Tools > Options > Charts > "Max bars in chart" = Unlimited.
3. Drag the script onto any chart. Adjust the symbol list to your broker's
   exact names if they differ. It lists every server symbol in `symbols.csv`,
   so you can look them up there first.
4. Output lands in `MQL5/Files/research/`. Then:

```bash
.venv/bin/python scripts/inspect_mt5_export.py "<path to>/MQL5/Files/research"
```

This prints the broker specs for each instrument (for `config/instruments.yaml`)
and compares MT5 spreads by hour with Dukascopy's.
Please commit `symbols.csv`, `sessions.csv` and `calendar_high.csv` to
`data/mt5/` (they are small). The M1 bar CSVs are too big for git.

## How the backtester models reality (`src/tradelab/backtest/engine.py`)

- Signals on bar close (M15/H1), market entry at next bar open. No look-ahead.
- Longs buy the ask and sell the bid; shorts the reverse. Spread is the real
  historical Dukascopy bid/ask spread per minute, plus a configurable markup.
- Stops/targets are resolved on **M1 data**: the minute that touched first
  wins; if one minute touches both, the **stop** is assumed first.
- Stop fills get slippage; gaps fill at the gap price, not the stop.
- Commission ($4/lot round trip for FX, % of notional for others), swaps with
  triple/weekend multipliers, broker session hours, news blackout.
- Position size = risk % / stop distance, **rounded down** to 0.01 lots; a
  trade that can't be sized at the minimum lot is skipped and counted.
- Daily table includes a conservative intraday low for daily-loss checks.

## Research rules (fixed before seeing any results: `src/tradelab/splits.py`)

- In-sample 2018–2022 (choose parameters), out-of-sample 2023–2025 (accept /
  reject once), holdout 2026 (untouched until the final portfolio check).
- Walk-forward: 3-year train, 1-year test, rolled yearly.
- A strategy is kept only if OOS has ≥ 100 trades, profit factor ≥ 1.15
  after costs, positive avg R, and OOS avg R ≥ 50% of in-sample.
  Failures are reported and dropped, not re-tuned.

## Layout

```
config/            challenge rules, instrument specs
src/tradelab/      data, costs, news, backtest engine, metrics, splits
scripts/           download, inspect MT5 export, baseline, synthetic demo
mql5/Scripts/      ExportResearchData.mq5
tests/             engine and cost-model tests
```

"""Load stored M1 data and build research bars in server time.

Every frame used by the backtester has a naive server-time index and columns:
    open, high, low, close   bid prices
    spread                   ask - bid at the bar open, in price units
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import DATA_DIR
from ..config import Instrument
from ..timeutil import utc_to_server


def load_m1(inst: Instrument, start: str | None = None, end: str | None = None,
            root: Path | None = None) -> pd.DataFrame:
    root = root or DATA_DIR / "m1"
    files = sorted((root / inst.name).glob("*.parquet"))
    if start:
        files = [f for f in files if int(f.stem) >= pd.Timestamp(start).year]
    if end:
        files = [f for f in files if int(f.stem) <= pd.Timestamp(end).year]
    if not files:
        raise FileNotFoundError(f"No M1 data for {inst.name} under {root / inst.name}")
    from .dukascopy import load_compact
    df = pd.concat(load_compact(f) for f in files).sort_index()
    df.index = utc_to_server(pd.DatetimeIndex(df.index))
    df = df[~df.index.duplicated()]
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return prepare_m1(df, inst)


def prepare_m1(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    """Apply the broker spread markup and fill missing spreads."""
    df = df[["open", "high", "low", "close", "spread"]].astype(float).copy()
    df["spread"] = df["spread"].fillna(df["spread"].median()) + inst.spread_markup
    return df


def resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """M1 -> M15/H1/D1 bars. Bars are labelled by their open time (server time)."""
    agg = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "first"}
    )
    return agg.dropna(subset=["open"])


def load_mt5_csv(path: Path, inst: Instrument) -> pd.DataFrame:
    """Load bars written by mql5/Scripts/ExportResearchData.mq5 (server time, spread in points)."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    df = df.rename(columns=str.lower)
    df["spread"] = df["spread"] * inst.point
    return df[["open", "high", "low", "close", "spread"]]

"""Download free historical M1 bid/ask candles from Dukascopy.

Dukascopy serves one LZMA-compressed file per instrument, side and day:
    https://datafeed.dukascopy.com/datafeed/{CODE}/{YYYY}/{MM0}/{DD}/{SIDE}_candles_min_1.bi5
where MM0 is the zero-based month. Each record is 24 bytes, big-endian:
    uint32 seconds-from-midnight-UTC, uint32 open, close, low, high, float32 volume
Prices are integers; divide by the instrument's divisor (e.g. 100000 for EURUSD).

We keep bid OHLC plus the bid/ask spread at each bar's open, because entries
happen at bar opens. Output: data/m1/{SYMBOL}/{YEAR}.parquet, UTC timestamps.
"""
from __future__ import annotations

import datetime as dt
import lzma
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DATA_DIR
from ..config import Instrument

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">5I f")
RECORD_DTYPE = np.dtype(
    [("t", ">u4"), ("open", ">u4"), ("close", ">u4"), ("low", ">u4"), ("high", ">u4"), ("vol", ">f4")]
)


def day_url(code: str, day: dt.date, side: str) -> str:
    return f"{BASE_URL}/{code}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def parse_bi5(blob: bytes, day: dt.date, divisor: float) -> pd.DataFrame:
    """Decode one day of candles. Empty blob -> empty frame (weekends, holidays)."""
    if not blob:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
    raw = lzma.decompress(blob)
    arr = np.frombuffer(raw, dtype=RECORD_DTYPE)
    start = pd.Timestamp(day, tz="UTC")
    idx = start + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    df = pd.DataFrame(
        {
            "open": arr["open"] / divisor,
            "high": arr["high"] / divisor,
            "low": arr["low"] / divisor,
            "close": arr["close"] / divisor,
            "volume": arr["vol"].astype(float),
        },
        index=idx,
    )
    # Dukascopy fills closed-market minutes with flat zero-volume candles.
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])
    return df[~flat]


def encode_bi5(df: pd.DataFrame, day: dt.date, divisor: float) -> bytes:
    """Inverse of parse_bi5; used by tests."""
    start = pd.Timestamp(day, tz="UTC")
    out = bytearray()
    for ts, r in df.iterrows():
        out += RECORD.pack(
            int((ts - start).total_seconds()),
            round(r.open * divisor), round(r.close * divisor),
            round(r.low * divisor), round(r.high * divisor), float(r.volume),
        )
    return lzma.compress(bytes(out), format=lzma.FORMAT_ALONE)


def _fetch(url: str, cache: Path, retries: int = 5) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tradelab"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                blob = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                blob = b""
                break
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(blob)
    return blob


def download_day(inst: Instrument, day: dt.date, raw_dir: Path) -> pd.DataFrame:
    frames = {}
    for side in ("BID", "ASK"):
        url = day_url(inst.dukascopy_code, day, side)
        cache = raw_dir / inst.dukascopy_code / f"{day:%Y/%m/%d}_{side}.bi5"
        frames[side] = parse_bi5(_fetch(url, cache), day, inst.dukascopy_divisor)
    bid, ask = frames["BID"], frames["ASK"]
    if bid.empty:
        return bid.assign(spread=pd.Series(dtype=float))
    spread = (ask["open"] - bid["open"]).reindex(bid.index)
    bid = bid.assign(spread=spread.clip(lower=0))
    return bid


def validate(df: pd.DataFrame, inst: Instrument) -> None:
    """Catch a wrong divisor or field order before it poisons a backtest."""
    if df.empty:
        return
    bad = (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12) | (
        df["high"] < df[["open", "close"]].max(axis=1) - 1e-12
    )
    if bad.mean() > 0.001:
        raise ValueError(f"{inst.name}: {bad.mean():.1%} of candles have inconsistent OHLC")
    if inst.sane_range:
        lo, hi = inst.sane_range
        med = df["close"].median()
        if not lo <= med <= hi:
            raise ValueError(f"{inst.name}: median price {med} outside {inst.sane_range}; check divisor")


def download_year(inst: Instrument, year: int, out_dir: Path | None = None,
                  raw_dir: Path | None = None, workers: int = 8, end: dt.date | None = None) -> Path:
    out_dir = out_dir or DATA_DIR / "m1"
    raw_dir = raw_dir or DATA_DIR / "raw"
    last = min(dt.date(year, 12, 31), end or dt.date.today() - dt.timedelta(days=1))
    days = [d.date() for d in pd.date_range(dt.date(year, 1, 1), last) if d.weekday() != 5]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda d: download_day(inst, d, raw_dir), days))
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    df = df[~df.index.duplicated()]
    validate(df, inst)
    path = out_dir / inst.name / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("float64").to_parquet(path)
    return path

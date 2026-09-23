"""Load challenge rules and instrument specs from config/*.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import CONFIG_DIR


@dataclass(frozen=True)
class Instrument:
    name: str
    mt5_symbol: str
    contract_size: float
    quote_ccy: str
    point: float
    min_lot: float
    lot_step: float
    commission_type: str          # "per_lot_rt" | "pct_notional_rt"
    commission_value: float
    spread_markup_points: float
    slip_entry_points: float
    slip_stop_points: float
    swap_long: float              # USD per lot per night
    swap_short: float
    swap_triple_day: str          # weekday name or "none"
    swap_weekend_multiplier: float
    session_open: str             # "HH:MM" server time
    session_close: str
    dukascopy_code: str | None = None
    dukascopy_divisor: float | None = None
    sane_range: tuple[float, float] | None = None
    usd_fallback_rate: float = 1.0  # quote currency -> USD
    usd_conversion_symbol: str | None = None
    usd_conversion_invert: bool = False
    margin_per_lot_usd: float | None = None
    verified: bool = False
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def slip_entry(self) -> float:
        return self.slip_entry_points * self.point

    @property
    def slip_stop(self) -> float:
        return self.slip_stop_points * self.point

    @property
    def spread_markup(self) -> float:
        return self.spread_markup_points * self.point


def _instrument(name: str, d: dict) -> Instrument:
    duk = d.get("dukascopy") or {}
    conv = d.get("usd_conversion") or {}
    swap = d.get("swap_per_lot") or {}
    slip = d.get("slippage_points") or {}
    sess = d.get("session") or {}
    return Instrument(
        name=name,
        mt5_symbol=d.get("mt5_symbol", name),
        contract_size=float(d["contract_size"]),
        quote_ccy=d.get("quote_ccy", "USD"),
        point=float(d["point"]),
        min_lot=float(d.get("min_lot", 0.01)),
        lot_step=float(d.get("lot_step", 0.01)),
        commission_type=d["commission"]["type"],
        commission_value=float(d["commission"]["value"]),
        spread_markup_points=float(d.get("spread_markup_points", 0)),
        slip_entry_points=float(slip.get("entry", 0)),
        slip_stop_points=float(slip.get("stop", 0)),
        swap_long=float(swap.get("long", 0.0)),
        swap_short=float(swap.get("short", 0.0)),
        swap_triple_day=str(swap.get("triple_day", "wednesday")).lower(),
        swap_weekend_multiplier=float(swap.get("weekend_multiplier", 1)),
        session_open=str(sess.get("open", "00:00")),
        session_close=str(sess.get("close", "23:59")),
        dukascopy_code=duk.get("code"),
        dukascopy_divisor=float(duk["divisor"]) if duk.get("divisor") else None,
        sane_range=tuple(duk["sane_range"]) if duk.get("sane_range") else None,
        usd_fallback_rate=float(conv.get("fallback_rate", 1.0)),
        usd_conversion_symbol=conv.get("symbol"),
        usd_conversion_invert=bool(conv.get("invert", False)),
        margin_per_lot_usd=d.get("margin_per_lot_usd"),
        verified=bool(d.get("verified", False)),
        raw=d,
    )


def load_instruments(path: Path | None = None) -> dict[str, Instrument]:
    path = path or CONFIG_DIR / "instruments.yaml"
    data = yaml.safe_load(Path(path).read_text())
    return {name: _instrument(name, d) for name, d in data.items()}


def load_instrument(name: str, path: Path | None = None) -> Instrument:
    return load_instruments(path)[name]


def load_challenge(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "challenge.yaml"
    return yaml.safe_load(Path(path).read_text())

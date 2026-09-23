"""Position sizing and trading costs for one instrument."""
from __future__ import annotations

import math

import pandas as pd

from .config import Instrument

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def floor_lots(lots: float, inst: Instrument) -> float:
    steps = math.floor(lots / inst.lot_step + 1e-9)
    return round(steps * inst.lot_step, 8)


def size_position(risk_usd: float, stop_dist: float, inst: Instrument, usd_rate: float,
                  price: float, max_notional_usd: float | None = None) -> float:
    """Lots so that hitting the stop loses at most risk_usd (before costs). 0 if below min lot."""
    if stop_dist <= 0 or risk_usd <= 0:
        return 0.0
    loss_per_lot = stop_dist * inst.contract_size * usd_rate
    lots = floor_lots(risk_usd / loss_per_lot, inst)
    if max_notional_usd is not None:
        notional_per_lot = price * inst.contract_size * usd_rate
        if inst.margin_per_lot_usd:
            # margin-based cap: keep 10% free margin
            lots = min(lots, floor_lots(max_notional_usd * 0.9 / inst.margin_per_lot_usd, inst))
        elif notional_per_lot > 0:
            lots = min(lots, floor_lots(max_notional_usd / notional_per_lot, inst))
    return lots if lots >= inst.min_lot - 1e-9 else 0.0


def commission_usd(inst: Instrument, lots: float, price: float, usd_rate: float) -> float:
    if inst.commission_type == "per_lot_rt":
        return inst.commission_value * lots
    if inst.commission_type == "pct_notional_rt":
        return inst.commission_value / 100.0 * lots * inst.contract_size * price * usd_rate
    raise ValueError(f"unknown commission type {inst.commission_type}")


def swap_nights(inst: Instrument, entry: pd.Timestamp, exit: pd.Timestamp) -> float:
    """Swap-night count (with triple/weekend multipliers) for a position held entry->exit.

    A rollover happens at each server midnight; the one that ends trading day D
    is charged if D is a trading day. On `swap_triple_day` it counts 3 nights,
    or `swap_weekend_multiplier` when that is set above 1 (NAS100 x3, oil x10).
    """
    first = entry.normalize() + pd.Timedelta(days=1)
    last = exit.normalize()
    if last < first:
        return 0.0
    weekend_trading = inst.swap_triple_day == "none"
    nights = 0.0
    for midnight in pd.date_range(first, last, freq="D"):
        day = WEEKDAYS[(midnight - pd.Timedelta(days=1)).dayofweek]
        if not weekend_trading and day in ("saturday", "sunday"):
            continue
        if day == inst.swap_triple_day:
            nights += inst.swap_weekend_multiplier if inst.swap_weekend_multiplier > 1 else 3
        else:
            nights += 1
    return nights


def swap_usd(inst: Instrument, side: int, lots: float, entry: pd.Timestamp, exit: pd.Timestamp) -> float:
    per_lot = inst.swap_long if side > 0 else inst.swap_short
    return per_lot * lots * swap_nights(inst, entry, exit)

"""Event-driven bar backtester with M1 fill resolution.

How a trade is simulated:
  * Signals are computed on the strategy's timeframe (M15/H1) at bar close.
  * Entry is a market order at the next bar's open: longs pay the ask
    (bid + spread) plus entry slippage, shorts sell the bid minus slippage.
  * Stops and targets are checked minute by minute on M1 data, so the order in
    which they are touched inside a bar is known. If both are touched in the
    same minute, the stop is assumed hit first (conservative).
  * Stop fills get stop slippage; if a minute opens beyond the stop (a gap),
    the fill is that minute's open price minus slippage, not the stop price.
  * Long positions exit on the bid, shorts on the ask.
  * Commission and swap are charged per the instrument config.
  * Size is fixed-fractional: risk_pct of the balance at the stop distance,
    rounded DOWN to the lot step. Trades below the minimum lot are skipped
    and counted, never rounded up.
  * One open position per strategy.

The result is a trade list with R-multiples and a daily table with realized
P&L and a conservative intraday low (see `daily_low_usd`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Instrument
from ..costs import commission_usd, size_position, swap_usd
from ..data.store import resample
from ..news import NewsCalendar
from ..timeutil import minutes_of_day
from .strategy import Strategy


@dataclass
class BacktestConfig:
    initial_balance: float = 5000.0
    risk_pct: float = 1.0                  # % of balance risked per trade
    compounding: bool = False              # False: risk is % of initial balance
    leverage: float = 30.0
    respect_session: bool = True
    news: NewsCalendar | None = None
    news_before_min: float = 5.0
    news_after_min: float = 5.0
    usd_rate: pd.Series | None = None      # quote->USD rate over time (e.g. 1/USDJPY)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    skipped: dict = field(default_factory=dict)
    config: BacktestConfig | None = None
    strategy: str = ""
    symbol: str = ""


@dataclass
class _Pos:
    side: int
    entry_idx: int                         # tf bar index of entry
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    initial_stop: float
    target: float
    lots: float
    units: float                           # lots * contract size
    usd_rate: float
    risk_usd: float
    commission: float
    mae_price: float                       # worst adverse price seen
    day_ref: float = 0.0                   # floating P&L at start of current day
    day_worst: dict = field(default_factory=dict)  # date -> worst change that day


def _session_ok(times: pd.DatetimeIndex, inst: Instrument) -> np.ndarray:
    mins = times.hour * 60 + times.minute
    return (mins >= minutes_of_day(inst.session_open)) & (mins <= minutes_of_day(inst.session_close))


def run_backtest(strategy: Strategy, m1: pd.DataFrame, inst: Instrument,
                 cfg: BacktestConfig | None = None, bars: pd.DataFrame | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    bars = bars if bars is not None else resample(m1, strategy.timeframe)
    sig = strategy.signals(bars).reindex(bars.index)

    n = len(bars)
    entry = sig["entry"].fillna(0).to_numpy(dtype=int)
    stop_s = sig["stop"].to_numpy(dtype=float) if "stop" in sig else np.full(n, np.nan)
    tgt_s = sig["target"].to_numpy(dtype=float) if "target" in sig else np.full(n, np.nan)
    ex_l = sig["exit_long"].fillna(False).to_numpy(dtype=bool) if "exit_long" in sig else np.zeros(n, bool)
    ex_s = sig["exit_short"].fillna(False).to_numpy(dtype=bool) if "exit_short" in sig else np.zeros(n, bool)
    tr_l = sig["trail_long"].to_numpy(dtype=float) if "trail_long" in sig else np.full(n, np.nan)
    tr_s = sig["trail_short"].to_numpy(dtype=float) if "trail_short" in sig else np.full(n, np.nan)

    # M1 arrays and each bar's M1 slice [lo[i], lo[i+1])
    t1 = m1.index.to_numpy(dtype="datetime64[ns]")
    o1, h1, l1 = (m1[c].to_numpy(dtype=float) for c in ("open", "high", "low"))
    sp1 = m1["spread"].to_numpy(dtype=float)
    lo = np.searchsorted(t1, bars.index.to_numpy(dtype="datetime64[ns]"), side="left")
    hi = np.r_[lo[1:], len(t1)]
    btimes = bars.index
    bdates = btimes.normalize()

    allowed = np.ones(n, dtype=bool)
    if cfg.respect_session:
        allowed &= _session_ok(btimes, inst)
    if cfg.news is not None:
        allowed &= ~cfg.news.blocked(btimes, inst.name, cfg.news_before_min, cfg.news_after_min)

    rate_t = rate_v = None
    if cfg.usd_rate is not None and len(cfg.usd_rate):
        rate_t = cfg.usd_rate.index.to_numpy(dtype="datetime64[ns]")
        rate_v = cfg.usd_rate.to_numpy(dtype=float)

    def usd_rate_at(ts) -> float:
        if rate_t is None:
            return inst.usd_fallback_rate
        k = np.searchsorted(rate_t, np.datetime64(ts, "ns"), side="right") - 1
        return float(rate_v[max(k, 0)])

    trades: list[dict] = []
    skipped = {"min_lot": 0, "session_or_news": 0, "stop_already_hit": 0}
    balance = cfg.initial_balance
    pos: _Pos | None = None
    pending_entry: tuple[int, float, float] | None = None   # side, stop, target
    pending_exit: str | None = None

    def floating(p: _Pos, price: float) -> float:
        return (price - p.entry_price) * p.side * p.units * p.usd_rate - p.commission

    def close(p: _Pos, j: int, price: float, reason: str) -> None:
        nonlocal balance
        exit_time = pd.Timestamp(t1[j]) if j < len(t1) else btimes[-1]
        gross = (price - p.entry_price) * p.side * p.units * p.usd_rate
        swp = swap_usd(inst, p.side, p.lots, p.entry_time, exit_time)
        pnl = gross - p.commission + swp
        balance += pnl
        mae_usd = (p.mae_price - p.entry_price) * p.side * p.units * p.usd_rate
        trades.append({
            "entry_time": p.entry_time, "exit_time": exit_time, "side": p.side,
            "entry": p.entry_price, "exit": price, "stop": p.initial_stop, "target": p.target,
            "lots": p.lots, "risk_usd": p.risk_usd, "commission": p.commission, "swap": swp,
            "pnl_usd": pnl, "r": pnl / p.risk_usd, "r_pre_comm": gross / p.risk_usd,
            "mae_r": mae_usd / p.risk_usd, "bars_held": None, "exit_reason": reason,
            "day_worst": dict(p.day_worst),
        })

    for i in range(n):
        a, b = lo[i], hi[i]
        if a >= b:
            continue
        day = bdates[i]

        # 1) orders queued at the previous bar's close, filled at this bar's open
        if pos is not None and pending_exit is not None:
            px = _mark(pos, o1[a], sp1[a]) - pos.side * inst.slip_entry
            _touch_day(pos, day, i, floating(pos, _mark(pos, o1[a], sp1[a])))
            pos.day_worst[day] = min(pos.day_worst[day], floating(pos, px) - pos.day_ref)
            close(pos, a, px, pending_exit)
            trades[-1]["bars_held"] = i - pos.entry_idx
            pos = None
        pending_exit = None

        if pos is None and pending_entry is not None:
            side, stop, target = pending_entry
            if not allowed[i]:
                skipped["session_or_news"] += 1
            else:
                fill = o1[a] + (sp1[a] if side > 0 else 0.0) + side * inst.slip_entry
                dist = (fill - stop) * side
                if dist <= 0 or (not np.isnan(target) and (target - fill) * side <= 0):
                    skipped["stop_already_hit"] += 1
                else:
                    rate = usd_rate_at(btimes[i])
                    base = balance if cfg.compounding else cfg.initial_balance
                    risk = base * cfg.risk_pct / 100.0
                    lots = size_position(risk, dist, inst, rate, fill, balance * cfg.leverage)
                    if lots <= 0:
                        skipped["min_lot"] += 1
                    else:
                        units = lots * inst.contract_size
                        pos = _Pos(side, i, btimes[i], fill, stop, stop, target, lots, units, rate,
                                   dist * units * rate, commission_usd(inst, lots, fill, rate), fill)
        pending_entry = None

        # 2) walk this bar's minutes for stop / target
        if pos is not None:
            _touch_day(pos, day, i, floating(pos, _mark(pos, o1[a], sp1[a])))
            hs, ls, sps, os_ = h1[a:b], l1[a:b], sp1[a:b], o1[a:b]
            if pos.side > 0:
                stop_hit = ls <= pos.stop
                tgt_hit = hs >= pos.target if not np.isnan(pos.target) else np.zeros(b - a, bool)
            else:
                stop_hit = hs + sps >= pos.stop
                tgt_hit = ls + sps <= pos.target if not np.isnan(pos.target) else np.zeros(b - a, bool)
            js = int(np.argmax(stop_hit)) if stop_hit.any() else b - a
            jt = int(np.argmax(tgt_hit)) if tgt_hit.any() else b - a
            k = min(js, jt)
            upto = min(k + 1, b - a)
            # worst adverse price reached while open in this bar
            worst = ls[:upto].min() if pos.side > 0 else (hs[:upto] + sps[:upto]).max()
            if js <= jt and js < b - a:
                worst = min(worst, pos.stop) if pos.side > 0 else max(worst, pos.stop)
            pos.mae_price = min(pos.mae_price, worst) if pos.side > 0 else max(pos.mae_price, worst)
            pos.day_worst[day] = min(pos.day_worst[day], floating(pos, worst) - pos.day_ref)

            if js < b - a and js <= jt:
                if pos.side > 0:
                    px = min(os_[js], pos.stop) - inst.slip_stop
                else:
                    px = max(os_[js] + sps[js], pos.stop) + inst.slip_stop
                pos.day_worst[day] = min(pos.day_worst[day], floating(pos, px) - pos.day_ref)
                close(pos, a + js, px, "stop")
                trades[-1]["bars_held"] = i - pos.entry_idx
                pos = None
            elif jt < b - a:
                close(pos, a + jt, pos.target, "target")
                trades[-1]["bars_held"] = i - pos.entry_idx
                pos = None

        # 3) decisions at this bar's close
        if pos is not None:
            if pos.side > 0 and not np.isnan(tr_l[i]) and tr_l[i] > pos.stop:
                pos.stop = tr_l[i]
            if pos.side < 0 and not np.isnan(tr_s[i]) and tr_s[i] < pos.stop:
                pos.stop = tr_s[i]
            if (pos.side > 0 and ex_l[i]) or (pos.side < 0 and ex_s[i]):
                pending_exit = "signal"
            elif strategy.max_hold_bars and i + 1 - pos.entry_idx >= strategy.max_hold_bars:
                pending_exit = "time"
        elif entry[i] != 0 and i + 1 < n:
            pending_entry = (int(entry[i]), float(stop_s[i]), float(tgt_s[i]))

    if pos is not None:
        j = len(t1) - 1
        close(pos, j, _mark(pos, float(m1["close"].iloc[-1]), sp1[j]), "end_of_data")
        trades[-1]["bars_held"] = n - 1 - pos.entry_idx

    tdf = pd.DataFrame(trades)
    daily = _daily_table(tdf, bars, cfg.initial_balance)
    if not tdf.empty:
        tdf = tdf.drop(columns=["day_worst"])
    return BacktestResult(tdf, daily, skipped, cfg, strategy.name, inst.name)


def _touch_day(pos: _Pos, day, i: int, float_now: float) -> None:
    """Start tracking a new server day: remember the position's value at day start."""
    if day not in pos.day_worst:
        # On the entry bar the reference is 0, so entry commission counts as that day's loss.
        pos.day_ref = 0.0 if pos.entry_idx == i else float_now
        pos.day_worst[day] = 0.0


def _mark(pos: _Pos, bid: float, spread: float) -> float:
    """Price a position would close at: bid for longs, ask for shorts."""
    return bid if pos.side > 0 else bid + spread


def _daily_table(trades: pd.DataFrame, bars: pd.DataFrame, initial: float) -> pd.DataFrame:
    """Per server day: realized P&L and a conservative intraday low.

    daily_low_usd sums, over every trade active that day, the worst change in
    that trade's value during the day (a closed loser counts its full loss).
    Summing each trade's worst point is never better than the real equity low,
    so this errs on the side of caution for daily-loss checks.
    """
    days = pd.DatetimeIndex(bars.index.normalize().unique())
    out = pd.DataFrame(index=days)
    out.index.name = "date"
    if trades.empty:
        out["pnl_usd"] = 0.0
        out["daily_low_usd"] = 0.0
    else:
        out["pnl_usd"] = trades.groupby(trades["exit_time"].dt.normalize())["pnl_usd"].sum().reindex(days).fillna(0.0)
        low = pd.Series(0.0, index=days)
        for dw in trades["day_worst"]:
            for d, v in dw.items():
                if d in low.index:
                    low[d] += min(v, 0.0)
        out["daily_low_usd"] = np.minimum(low, np.minimum(out["pnl_usd"], 0.0))
    out["balance"] = initial + out["pnl_usd"].cumsum()
    out["pnl_pct"] = out["pnl_usd"] / initial * 100
    out["daily_low_pct"] = out["daily_low_usd"] / initial * 100
    return out

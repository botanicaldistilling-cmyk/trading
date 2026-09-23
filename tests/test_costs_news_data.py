import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tradelab.config import load_instrument, load_instruments, load_challenge
from tradelab.costs import commission_usd, floor_lots, size_position, swap_nights
from tradelab.data.dukascopy import encode_bi5, parse_bi5, validate
from tradelab.news import NewsCalendar
from tradelab.splits import walk_forward_windows
from tradelab.timeutil import utc_to_server


def test_configs_load():
    insts = load_instruments()
    assert {"EURUSD", "XAUUSD", "NAS100"} <= set(insts)
    ch = load_challenge()
    assert ch["max_loss"]["limit_usd"] == 300
    assert ch["risk_manager"]["daily_stop_pct"] < ch["daily_loss"]["limit_pct"]


def test_floor_lots_and_min_lot(eurusd):
    assert floor_lots(0.179, eurusd) == pytest.approx(0.17)
    assert size_position(0.5, 0.01, eurusd, 1.0, 1.1) == 0.0


def test_commission():
    eu, xau = load_instrument("EURUSD"), load_instrument("XAUUSD")
    assert commission_usd(eu, 0.5, 1.1, 1.0) == pytest.approx(2.0)
    # 1 lot gold at 2000 = 200k notional * 0.002% = $4
    assert commission_usd(xau, 1.0, 2000.0, 1.0) == pytest.approx(4.0)


def test_swap_nights():
    eu, nas = load_instrument("EURUSD"), load_instrument("NAS100")
    T = pd.Timestamp
    assert swap_nights(eu, T("2024-01-02 10:00"), T("2024-01-02 20:00")) == 0   # intraday
    assert swap_nights(eu, T("2024-01-03 10:00"), T("2024-01-04 10:00")) == 3   # Wednesday night
    assert swap_nights(eu, T("2024-01-05 10:00"), T("2024-01-08 10:00")) == 1   # Fri -> Mon
    assert swap_nights(nas, T("2024-01-05 10:00"), T("2024-01-08 10:00")) == 3  # weekend x3


def test_server_time_midnight_is_ny_5pm():
    winter = utc_to_server(pd.DatetimeIndex([pd.Timestamp("2024-01-10 22:00", tz="UTC")]))
    summer = utc_to_server(pd.DatetimeIndex([pd.Timestamp("2024-07-10 21:00", tz="UTC")]))
    assert winter[0] == pd.Timestamp("2024-01-11 00:00")
    assert summer[0] == pd.Timestamp("2024-07-11 00:00")


def test_news_blackout():
    cal = NewsCalendar(pd.DataFrame({"time": ["2024-01-05 15:30"], "currency": ["USD"], "impact": ["High"]}))
    times = pd.DatetimeIndex(["2024-01-05 15:24", "2024-01-05 15:26", "2024-01-05 15:30",
                              "2024-01-05 15:35", "2024-01-05 15:36"])
    assert list(cal.blocked(times, "EURUSD", 5, 5)) == [False, True, True, True, False]
    assert not cal.blocked(times, "GBPJPY", 5, 5).any()


def test_dukascopy_roundtrip():
    inst = load_instrument("EURUSD")
    day = dt.date(2024, 1, 2)
    idx = pd.date_range("2024-01-02", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": [1.1, 1.10002, 1.10004], "high": [1.10005, 1.10006, 1.10008],
                       "low": [1.09995, 1.1, 1.10001], "close": [1.10002, 1.10004, 1.10005],
                       "volume": [10.0, 12.0, 8.0]}, index=idx)
    out = parse_bi5(encode_bi5(df, day, inst.dukascopy_divisor), day, inst.dukascopy_divisor)
    out.index = out.index.as_unit("us")
    df.index = df.index.as_unit("us")
    pd.testing.assert_frame_equal(out, df, check_freq=False, atol=1e-9)
    validate(out, inst)
    with pytest.raises(ValueError):
        validate(out * 1000, inst)   # wrong divisor is caught


def test_walk_forward_windows():
    w = list(walk_forward_windows("2018-01-01", "2025-12-31", 3, 1))
    assert len(w) == 5
    assert w[0][2] == pd.Timestamp("2021-01-01") and w[-1][3] == pd.Timestamp("2025-12-31")

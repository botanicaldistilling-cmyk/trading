import numpy as np
import pandas as pd

from tradelab.diagnostics import half_life, hurst, session_variance_ratios, variance_ratio


def _series(rets):
    return pd.Series(np.cumsum(rets))


def test_variance_ratio_detects_character():
    rng = np.random.default_rng(0)
    e = rng.normal(size=20000)
    rw = _series(e)
    rev = _series(e[1:] - 0.5 * e[:-1])          # negatively autocorrelated returns
    tr = np.zeros_like(e)
    for i in range(1, len(e)):
        tr[i] = 0.4 * tr[i - 1] + e[i]           # positively autocorrelated returns
    vr_rw, z_rw = variance_ratio(rw, 4)
    assert abs(vr_rw - 1) < 0.05 and abs(z_rw) < 3
    assert variance_ratio(rev, 4)[0] < 0.8
    assert variance_ratio(_series(tr), 4)[0] > 1.3


def test_hurst_and_half_life():
    rng = np.random.default_rng(1)
    assert 0.45 < hurst(_series(rng.normal(size=20000))) < 0.55
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = x[i - 1] * (1 - np.log(2) / 10) + rng.normal()
    assert 8 < half_life(pd.Series(x)) < 12


def test_session_vr_runs_and_splits_days():
    idx = pd.date_range("2021-01-04", periods=96 * 60, freq="15min")
    rng = np.random.default_rng(2)
    bars = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))}, index=idx)
    out = session_variance_ratios(bars, {"asia": (200, 845), "all": (0, 2359)})
    assert list(out["session"]) == ["asia", "all"]
    assert out["VR(4)"].between(0.8, 1.2).all()

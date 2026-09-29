import numpy as np
import pytest

from crucible.model import Drivers, WACCInputs, checks, dcf, forecast, sensitivity
from crucible.synthetic import synthetic_history


@pytest.fixture(scope="module")
def hist():
    return synthetic_history()


def test_history_is_internally_consistent(hist):
    assert (hist.loc["total_assets"] - hist.loc["total_liabilities_and_equity"]).abs().max() < 1e-6
    assert (hist.loc["cfo"] + hist.loc["cfi"] + hist.loc["cff"] - hist.loc["net_change_cash"]).abs().max() < 1e-6


def test_forecast_balances_and_cash_rolls(hist):
    drv = Drivers.from_history(hist, years=5)
    fc = forecast(hist, drv)
    chk = checks(fc, drv.years)
    assert chk["ok"] == 1.0, chk
    # equity roll-forward: prior + NI + SBC - div - buybacks
    y0, y1 = hist.columns[-1], drv.years[0]
    expected = fc.loc["total_equity", y0] + fc.loc["net_income", y1] + fc.loc["sbc", y1] - drv.dividends[0] - drv.buybacks[0]
    assert abs(fc.loc["total_equity", y1] - expected) < 1e-6


def test_driver_override_changes_only_that_driver(hist):
    drv = Drivers.from_history(hist, years=5)
    d2 = drv.with_override("revenue_growth", 0.30)
    assert d2.revenue_growth == [0.30] * 5 and d2.gross_margin == drv.gross_margin
    fc1, fc2 = forecast(hist, drv), forecast(hist, d2)
    assert fc2.loc["revenue", d2.years[-1]] > fc1.loc["revenue", drv.years[-1]]
    assert checks(fc2, d2.years)["ok"] == 1.0


def test_dcf_invariants(hist):
    drv = Drivers.from_history(hist, years=5)
    fc = forecast(hist, drv)
    w = WACCInputs()
    r = dcf(fc, drv, w.wacc, 0.025)
    assert r.enterprise_value == pytest.approx(r.pv_ufcf + r.pv_terminal)
    assert r.equity_value == pytest.approx(r.enterprise_value - r.net_debt)
    assert r.per_share == pytest.approx(r.equity_value / r.diluted_shares)
    with pytest.raises(ValueError):
        dcf(fc, drv, 0.02, 0.03)
    grid = sensitivity(fc, drv, [0.07, 0.08, 0.09], [0.02, 0.025, 0.03])
    vals = grid.values.astype(float)
    assert np.all(np.diff(vals, axis=0) < 0), "value must fall as WACC rises"
    assert np.all(np.diff(vals, axis=1) > 0), "value must rise with terminal growth"
    mid = dcf(fc, drv, w.wacc, 0.025, mid_year=True)
    assert mid.enterprise_value > r.enterprise_value

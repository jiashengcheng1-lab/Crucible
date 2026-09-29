"""Offline tests for the market-data caches and the analyst-inputs builder (fetchers injected)."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from crucible import marketdata as md
from crucible.inputs import derive_inputs, flatten, is_fresh, summary
from crucible.evidence import build_packet
from crucible.synthetic import synthetic_history


def fake_fred(series_id, start, end):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "value": np.linspace(4.0, 4.3, len(days))})


def _prices(symbol, start, end):
    """One shared market path (fixed seed) plus symbol-specific noise, so betas are recoverable."""
    weeks = pd.date_range(start, end, freq="W-FRI")
    mkt = np.cumsum(np.random.default_rng(7).normal(0.002, 0.02, len(weeks)))
    if symbol == "^GSPC":
        lvl = 4000 * np.exp(mkt)
    else:
        rng = np.random.default_rng(sum(map(ord, symbol)))
        beta = 1.5 if symbol == "VRT" else 1.0
        lvl = 100 * np.exp(beta * mkt + np.cumsum(rng.normal(0, 0.01, len(weeks))))
    return pd.DataFrame({"date": weeks, "close": lvl})


def fake_html():
    return "<table><tr><th>Year</th><th>Implied Premium (DDM)</th><th>Implied ERP (FCFE)</th></tr><tr><td>2024</td><td>4.10%</td><td>4.33%</td></tr><tr><td>2025</td><td>4.20%</td><td>4.60%</td></tr></table>"


def test_fred_cache_and_risk_free(tmp_path):
    rf = md.risk_free_rate("2026-09-24", data_dir=tmp_path, fetch=fake_fred)
    assert 0.04 < rf["value"] < 0.045 and rf["date"] <= "2026-09-24" and "DGS10" in rf["source"]
    calls = []
    def counting(series_id, start, end):
        calls.append(1)
        return fake_fred(series_id, start, end)
    md.risk_free_rate("2026-09-24", data_dir=tmp_path, fetch=counting)
    assert calls == []  # cache hit, no refetch


def test_weekly_beta_recovers_the_simulated_beta(tmp_path):
    b = md.weekly_beta("VRT", "^GSPC", "2026-09-24", data_dir=tmp_path, fetch=_prices)
    assert 1.2 < b["value"] < 1.8 and b["n_weeks"] >= 90 and 0 < b["r2"] <= 1
    assert abs(b["adjusted"] - (2 / 3 * b["value"] + 1 / 3)) < 1e-6


def test_damodaran_parser_takes_latest_fcfe_row():
    erp = md.damodaran_implied_erp(fetch=fake_html)
    assert erp["year"] == 2025 and abs(erp["value"] - 0.046) < 1e-6


def test_derive_flatten_and_packet(tmp_path):
    hist = synthetic_history()
    inp = derive_inputs("VRT", tmp_path, hist, peers=["ETN"], as_of="2026-09-24", fetch_fred=fake_fred, fetch_prices=_prices, fetch_erp=fake_html)
    f = inp["fields"]
    assert {"risk_free", "beta", "erp", "tax_rate", "cost_of_debt", "debt_weight"} <= set(f) and "ETN" in f["peer_betas"]
    assert f["debt_weight"]["source"].startswith("book debt")  # no facts in tmp -> book weight with a warning
    flat = flatten(inp)
    assert flat["risk_free"] == f["risk_free"]["value"] and "FRED" in flat["risk_free_source"]
    inp["overrides"] = {"beta": {"value": 1.1, "reason": "bottom-up peer beta"}}
    flat2 = flatten(inp)
    assert flat2["beta"] == 1.1 and flat2["beta_source"].startswith("analyst override")
    assert is_fresh(inp, "2026-09-26") and not is_fresh(inp, "2026-11-01")
    assert "cost_of_debt" in summary(inp)
    pk = build_packet("wacc", "Test Co", hist, analyst=flat2)
    srcs = " ".join(i.source for i in pk.items)
    assert "FRED" in srcs and "analyst override" in srcs and "Damodaran" in srcs and "expected market return" in srcs
    assert (tmp_path / "market" / "erp_series.csv").exists() and f["erp"]["method"] == "hist30y" and "hist10y" in f["erp"]["alternatives"]
    assert any(i.source == "mechanical WACC from the items above" for i in pk.items)

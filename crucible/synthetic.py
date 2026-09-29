"""Synthetic company history for offline demos and tests.

Built by running the forecast engine from a seed balance sheet, so the history
is internally consistent (balance sheet balances, cash rolls). The numbers are
invented and are labeled as such everywhere they appear.
"""
from __future__ import annotations

import pandas as pd

from .mapping import add_residuals
from .model import Drivers, forecast
from .schema import MappingReport, items_for

SEED_YEAR = 2020


def _seed_history() -> pd.DataFrame:
    y = SEED_YEAR
    rows = {
        "revenue": 4000.0, "cogs": 2880.0, "gross_profit": 1120.0, "sga": 620.0, "rnd": 160.0, "operating_income": 260.0,
        "interest_expense": 110.0, "pretax_income": 150.0, "income_tax": 33.0, "net_income": 117.0,
        "shares_diluted": 380.0, "eps_diluted": 117.0 / 380.0,
        "cash": 500.0, "receivables": 800.0, "inventory": 600.0, "total_current_assets": 2100.0, "ppe_net": 900.0,
        "goodwill": 1200.0, "intangibles": 400.0, "total_assets": 4900.0, "payables": 700.0, "short_term_debt": 100.0,
        "total_current_liabilities": 1300.0, "long_term_debt": 2000.0, "total_liabilities": 3600.0, "total_equity": 1300.0,
        "total_liabilities_and_equity": 4900.0,
        "cf_net_income": 117.0, "d_and_a": 120.0, "sbc": 40.0, "cfo": 250.0, "capex": 140.0, "cfi": -140.0,
        "dividends": 40.0, "buybacks": 0.0, "cff": -140.0, "net_change_cash": -30.0,
    }
    df = pd.DataFrame({y: rows})
    reports = {"IS": MappingReport("IS"), "BS": MappingReport("BS"), "CF": MappingReport("CF")}
    is_df = df.reindex([it.key for it in items_for("IS")])
    bs_df = df.reindex([it.key for it in items_for("BS")])
    cf_df = df.reindex([it.key for it in items_for("CF")])
    return add_residuals(is_df, bs_df, cf_df, reports)


def synthetic_history(years: int = 5) -> pd.DataFrame:
    """Five 'historical' years FY2021..FY2025 generated from the seed year."""
    seed = _seed_history()
    drv = Drivers(
        years=[SEED_YEAR + i for i in range(1, years + 1)],
        revenue_growth=[0.10, 0.14, 0.20, 0.17, 0.15][:years],
        gross_margin=[0.30, 0.31, 0.33, 0.35, 0.36][:years],
        sga_pct=[0.15] * years, rnd_pct=[0.04] * years, other_opex_pct=[0.02] * years,
        da_pct=[0.03] * years, sbc_pct=[0.01] * years, capex_pct=[0.035] * years, tax_rate=[0.22] * years,
        dso=[70.0] * years, dio=[60.0] * years, dpo=[65.0] * years, other_income=[0.0] * years,
        dividends=[40.0] * years, buybacks=[0.0, 0.0, 50.0, 80.0, 100.0][:years],
        net_debt_issuance=[-100.0, -100.0, -150.0, -100.0, -100.0][:years],
        interest_rate_debt=0.05, interest_rate_cash=0.01,
    )
    fc = forecast(seed, drv)
    raw_keys = [it.key for st in ("IS", "BS", "CF") for it in items_for(st)]
    raw = fc.reindex(raw_keys)[drv.years]
    raw.loc["d_and_a"] = fc.loc["cf_da", drv.years]
    raw.loc["sbc"] = fc.loc["cf_sbc", drv.years]
    reports = {"IS": MappingReport("IS"), "BS": MappingReport("BS"), "CF": MappingReport("CF")}
    return add_residuals(raw.reindex([it.key for it in items_for("IS")]),
                         raw.reindex([it.key for it in items_for("BS")]),
                         raw.reindex([it.key for it in items_for("CF")]), reports)


SYNTHETIC_META = {
    "ticker": "SYNTH",
    "name": "Synthetic Industrial Co. (invented data for offline demo)",
    "units": "USD millions",
    "sources": {str(SEED_YEAR + i): "synthetic" for i in range(1, 6)},
}

"""Deterministic three-statement model and DCF.

Everything numeric is computed here from (a) the mapped historical table and
(b) explicit driver assumptions. Interest is charged on beginning balances, so
the model has no circular reference and the Excel export can mirror it cell
for cell (see ``excel.py``; ``tests/test_excel.py`` asserts they agree).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd

DAYS = 365.0

PER_YEAR_DRIVERS = [
    ("revenue_growth", "Revenue growth", "pct"),
    ("gross_margin", "Gross margin", "pct"),
    ("sga_pct", "SG&A % revenue", "pct"),
    ("rnd_pct", "R&D % revenue", "pct"),
    ("other_opex_pct", "Other opex % revenue", "pct"),
    ("da_pct", "D&A % revenue", "pct"),
    ("sbc_pct", "SBC % revenue", "pct"),
    ("capex_pct", "Capex % revenue", "pct"),
    ("tax_rate", "Effective tax rate", "pct"),
    ("dso", "Receivable days (DSO)", "days"),
    ("dio", "Inventory days (DIO)", "days"),
    ("dpo", "Payable days (DPO)", "days"),
    ("other_income", "Other income/(expense), net", "amount"),
    ("dividends", "Dividends paid", "amount"),
    ("buybacks", "Buybacks", "amount"),
    ("net_debt_issuance", "Net debt issued/(repaid)", "amount"),
]
SCALAR_DRIVERS = [
    ("interest_rate_debt", "Interest rate on debt", "pct"),
    ("interest_rate_cash", "Yield on cash", "pct"),
]


@dataclass
class Drivers:
    years: list[int]
    revenue_growth: list[float]
    gross_margin: list[float]
    sga_pct: list[float]
    rnd_pct: list[float]
    other_opex_pct: list[float]
    da_pct: list[float]
    sbc_pct: list[float]
    capex_pct: list[float]
    tax_rate: list[float]
    dso: list[float]
    dio: list[float]
    dpo: list[float]
    other_income: list[float]
    dividends: list[float]
    buybacks: list[float]
    net_debt_issuance: list[float]
    interest_rate_debt: float = 0.05
    interest_rate_cash: float = 0.02
    nwc_method: str = "days"          # "days" (DSO/DIO/DPO) or "pct_revenue" (ratios held at the last historical year)
    shares_override: float | None = None  # diluted shares (millions) at the as-of price from the treasury stock method; None = last 10-K diluted count
    non_operating_assets: float = 0.0     # USD m added to equity value (a bitcoin treasury, investments); not in the cash flows
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.years)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Drivers":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def with_override(self, key: str, value) -> "Drivers":
        """Return a copy with one driver replaced (scalar or full per-year list)."""
        d = self.to_dict()
        if key in dict((k, 1) for k, _, _ in PER_YEAR_DRIVERS) and not isinstance(value, (list, tuple)):
            value = [float(value)] * self.n
        d[key] = value
        return Drivers.from_dict(d)

    @classmethod
    def from_history(cls, hist: pd.DataFrame, years: int = 5, lookback: int = 3) -> "Drivers":
        """Default drivers from trailing averages. These are starting points the
        debate and the analyst are expected to overwrite."""
        h = hist
        cols = list(h.columns)
        last = cols[-1]
        lb = cols[-lookback:]
        rev = h.loc["revenue"]

        def pct(key: str) -> float:
            return float((h.loc[key, lb] / rev[lb].replace(0, np.nan)).mean())

        growth = float((rev[lb].pct_change().dropna()).mean()) if len(lb) > 1 else 0.05
        gm = float(((rev[lb] - h.loc["cogs", lb]) / rev[lb].replace(0, np.nan)).mean())
        pretax = h.loc["pretax_income", lb]
        tax = h.loc["income_tax", lb]
        tax_rate = float(np.clip((tax.sum() / pretax.sum()) if pretax.sum() > 0 else 0.21, 0.0, 0.35))
        cogs_last = h.loc["cogs", last] or np.nan
        dso = float(h.loc["receivables", last] / (rev[last] or np.nan) * DAYS) if rev[last] else 0.0
        dio = float(h.loc["inventory", last] / cogs_last * DAYS) if cogs_last and not np.isnan(cogs_last) else 0.0
        dpo = float(h.loc["payables", last] / cogs_last * DAYS) if cogs_last and not np.isnan(cogs_last) else 0.0
        debt = h.loc["short_term_debt"] + h.loc["long_term_debt"]
        avg_debt = float(debt[cols[-2:]].mean()) if len(cols) >= 2 else float(debt[last])
        rate = float(np.clip(h.loc["interest_expense", last] / avg_debt, 0.0, 0.15)) if avg_debt > 0 else 0.05
        fy = [int(last) + i for i in range(1, years + 1)]

        def rep(x: float) -> list[float]:
            x = 0.0 if (x is None or np.isnan(x)) else x
            return [float(x)] * years

        notes = {"basis": f"trailing {lookback}-year averages ending FY{last}; working-capital days from FY{last}"}
        other = pct("other_opex")
        if not np.isnan(other) and other < 0:
            # a negative residual is net gains inside operating income (fair-value gains on digital assets, disposals):
            # forecasting them as recurring would carry the gains into value; the default is zero and the analyst decides
            notes["other_opex"] = f"historical other operating items were net gains ({other:+.0%} of revenue on average); not forecast as recurring (0%); override with --drivers"
            other = 0.0
        da, cx = pct("d_and_a"), pct("capex")
        if not np.isnan(da) and not np.isnan(cx) and growth > 0 and cx < da:
            # capex below depreciation while revenue grows means the asset base shrinks as output rises
            notes["capex"] = f"capex/revenue ({cx:.0%}) was below D&A/revenue ({da:.0%}) with revenue growing; capex floored at D&A so the asset base is maintained; override with --drivers"
            cx = da

        return cls(
            years=fy,
            revenue_growth=rep(growth),
            gross_margin=rep(gm),
            sga_pct=rep(pct("sga")),
            rnd_pct=rep(pct("rnd")),
            other_opex_pct=rep(other),
            da_pct=rep(da),
            sbc_pct=rep(pct("sbc")),
            capex_pct=rep(cx),
            tax_rate=rep(tax_rate),
            dso=rep(dso),
            dio=rep(dio),
            dpo=rep(dpo),
            other_income=rep(0.0),
            dividends=rep(float(h.loc["dividends", last])),
            buybacks=rep(float(h.loc["buybacks", last])),
            net_debt_issuance=rep(0.0),
            interest_rate_debt=rate,
            interest_rate_cash=0.02,
            notes=notes,
        )


FORECAST_ROWS = [
    # IS
    "revenue", "cogs", "gross_profit", "sga", "rnd", "other_opex", "operating_income", "d_and_a", "ebitda", "sbc",
    "interest_expense", "other_income", "pretax_income", "income_tax", "net_income", "shares_diluted", "eps_diluted",
    # BS
    "cash", "receivables", "inventory", "other_current_assets", "total_current_assets", "ppe_net", "goodwill",
    "intangibles", "other_noncurrent_assets", "total_assets", "payables", "other_current_liabilities", "short_term_debt",
    "total_current_liabilities", "long_term_debt", "other_noncurrent_liabilities", "total_liabilities", "total_equity",
    "total_liabilities_and_equity", "balance_check",
    # CF
    "cf_net_income", "cf_da", "cf_sbc", "d_nwc", "cfo", "capex", "other_investing", "cfi", "dividends", "buybacks",
    "net_debt_issuance", "cff", "net_change_cash",
]


def forecast(hist: pd.DataFrame, drv: Drivers) -> pd.DataFrame:
    """Return a DataFrame (rows = FORECAST_ROWS, cols = hist years + forecast years)."""
    h = hist.copy()
    h.loc["cf_da"] = h.loc["d_and_a"]
    h.loc["cf_sbc"] = h.loc["sbc"]
    h.loc["balance_check"] = h.loc["total_assets"] - h.loc["total_liabilities_and_equity"]
    out = h.reindex(FORECAST_ROWS).fillna(0.0).astype(float)
    prev = out[out.columns[-1]].copy()
    for i, y in enumerate(drv.years):
        c = pd.Series(0.0, index=FORECAST_ROWS)
        rev = prev["revenue"] * (1 + drv.revenue_growth[i])
        cogs = rev * (1 - drv.gross_margin[i])
        gp = rev - cogs
        sga, rnd, oox = rev * drv.sga_pct[i], rev * drv.rnd_pct[i], rev * drv.other_opex_pct[i]
        ebit = gp - sga - rnd - oox
        da = rev * drv.da_pct[i]
        sbc = rev * drv.sbc_pct[i]
        interest = (prev["short_term_debt"] + prev["long_term_debt"]) * drv.interest_rate_debt - prev["cash"] * drv.interest_rate_cash
        oi = drv.other_income[i]
        pretax = ebit - interest + oi
        tax = pretax * drv.tax_rate[i]
        ni = pretax - tax
        if drv.nwc_method == "pct_revenue":
            last_rev = float(hist.loc["revenue"].iloc[-1]) or 1.0
            ar = rev * float(hist.loc["receivables"].iloc[-1]) / last_rev
            inv = rev * float(hist.loc["inventory"].iloc[-1]) / last_rev
            ap = rev * float(hist.loc["payables"].iloc[-1]) / last_rev
        else:
            ar = rev * drv.dso[i] / DAYS
            inv = cogs * drv.dio[i] / DAYS
            ap = cogs * drv.dpo[i] / DAYS
        capex = rev * drv.capex_pct[i]
        ppe = prev["ppe_net"] + capex - da
        ltd = prev["long_term_debt"] + drv.net_debt_issuance[i]
        equity = prev["total_equity"] + ni + sbc - drv.dividends[i] - drv.buybacks[i]
        d_nwc = -((ar - prev["receivables"]) + (inv - prev["inventory"]) - (ap - prev["payables"]))
        cfo = ni + da + sbc + d_nwc
        cfi = -capex
        cff = -drv.dividends[i] - drv.buybacks[i] + drv.net_debt_issuance[i]
        dcash = cfo + cfi + cff
        cash = prev["cash"] + dcash
        c.update(pd.Series({
            "revenue": rev, "cogs": cogs, "gross_profit": gp, "sga": sga, "rnd": rnd, "other_opex": oox,
            "operating_income": ebit, "d_and_a": da, "ebitda": ebit + da, "sbc": sbc, "interest_expense": interest,
            "other_income": oi, "pretax_income": pretax, "income_tax": tax, "net_income": ni,
            "shares_diluted": prev["shares_diluted"],
            "eps_diluted": ni / prev["shares_diluted"] if prev["shares_diluted"] else 0.0,
            "cash": cash, "receivables": ar, "inventory": inv, "other_current_assets": prev["other_current_assets"],
            "ppe_net": ppe, "goodwill": prev["goodwill"], "intangibles": prev["intangibles"],
            "other_noncurrent_assets": prev["other_noncurrent_assets"], "payables": ap,
            "other_current_liabilities": prev["other_current_liabilities"], "short_term_debt": prev["short_term_debt"],
            "long_term_debt": ltd, "other_noncurrent_liabilities": prev["other_noncurrent_liabilities"],
            "total_equity": equity, "cf_net_income": ni, "cf_da": da, "cf_sbc": sbc, "d_nwc": d_nwc, "cfo": cfo,
            "capex": capex, "other_investing": 0.0, "cfi": cfi, "dividends": drv.dividends[i],
            "buybacks": drv.buybacks[i], "net_debt_issuance": drv.net_debt_issuance[i], "cff": cff,
            "net_change_cash": dcash,
        }))
        c["total_current_assets"] = c["cash"] + c["receivables"] + c["inventory"] + c["other_current_assets"]
        c["total_assets"] = c["total_current_assets"] + c["ppe_net"] + c["goodwill"] + c["intangibles"] + c["other_noncurrent_assets"]
        c["total_current_liabilities"] = c["payables"] + c["other_current_liabilities"] + c["short_term_debt"]
        c["total_liabilities"] = c["total_current_liabilities"] + c["long_term_debt"] + c["other_noncurrent_liabilities"]
        c["total_liabilities_and_equity"] = c["total_liabilities"] + c["total_equity"]
        c["balance_check"] = c["total_assets"] - c["total_liabilities_and_equity"]
        out[y] = c
        prev = c
    return out


@dataclass
class WACCInputs:
    risk_free: float = 0.042
    beta: float = 1.2
    erp: float = 0.05
    size_premium: float = 0.0
    pretax_cost_of_debt: float = 0.055
    tax_rate: float = 0.21
    debt_weight: float = 0.20  # D / (D + E)

    @property
    def cost_of_equity(self) -> float:
        return self.risk_free + self.beta * self.erp + self.size_premium

    @property
    def wacc(self) -> float:
        return (1 - self.debt_weight) * self.cost_of_equity + self.debt_weight * self.pretax_cost_of_debt * (1 - self.tax_rate)


@dataclass
class DCFResult:
    wacc: float
    terminal_growth: float
    ufcf: pd.Series
    discount_factors: pd.Series
    pv_ufcf: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    diluted_shares: float
    per_share: float

    def as_dict(self) -> dict:
        return {
            "wacc": self.wacc, "terminal_growth": self.terminal_growth, "pv_ufcf": self.pv_ufcf,
            "terminal_value": self.terminal_value, "pv_terminal": self.pv_terminal,
            "enterprise_value": self.enterprise_value, "net_debt": self.net_debt,
            "equity_value": self.equity_value, "diluted_shares": self.diluted_shares, "per_share": self.per_share,
            "tv_share_of_ev": (self.pv_terminal / self.enterprise_value) if self.enterprise_value else float("nan"),
        }


def dcf(fc: pd.DataFrame, drv: Drivers, wacc: float, terminal_growth: float, mid_year: bool = False,
        last_hist_year: int | None = None) -> DCFResult:
    years = drv.years
    if last_hist_year is None:
        last_hist_year = int([c for c in fc.columns if c not in years][-1])
    ebit = fc.loc["operating_income", years]
    tax = pd.Series(drv.tax_rate, index=years)
    ufcf = ebit * (1 - tax) + fc.loc["d_and_a", years] - fc.loc["capex", years] + fc.loc["d_nwc", years]
    t = pd.Series([i + 1 - (0.5 if mid_year else 0.0) for i in range(len(years))], index=years)
    dfac = 1.0 / (1.0 + wacc) ** t
    pv = float((ufcf * dfac).sum())
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth")
    tv = float(ufcf.iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth))
    pv_tv = tv * float(dfac.iloc[-1])
    ev = pv + pv_tv
    net_debt = float(fc.loc["short_term_debt", last_hist_year] + fc.loc["long_term_debt", last_hist_year] - fc.loc["cash", last_hist_year])
    eq = ev - net_debt + float(getattr(drv, "non_operating_assets", 0.0) or 0.0)
    shares = float(drv.shares_override) if drv.shares_override else float(fc.loc["shares_diluted", last_hist_year])
    return DCFResult(wacc, terminal_growth, ufcf, dfac, pv, tv, pv_tv, ev, net_debt, eq, shares, eq / shares if shares else float("nan"))


def sensitivity(fc: pd.DataFrame, drv: Drivers, waccs: list[float], growths: list[float], mid_year: bool = False) -> pd.DataFrame:
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in waccs], columns=[f"{g:.1%}" for g in growths], dtype=float)
    for w in waccs:
        for g in growths:
            try:
                grid.at[f"{w:.1%}", f"{g:.1%}"] = dcf(fc, drv, w, g, mid_year).per_share
            except ValueError:
                grid.at[f"{w:.1%}", f"{g:.1%}"] = np.nan
    grid.index.name = "WACC \\ g"
    return grid


def checks(fc: pd.DataFrame, forecast_years: list[int], tol: float = 1e-6) -> dict[str, float]:
    """Integrity checks. Pass/fail applies to forecast years only; historical
    gaps (FX effects, restricted cash, restatements) are reported as diagnostics."""
    fy = [y for y in forecast_years if y in fc.columns]
    hy = [c for c in fc.columns if c not in fy]
    scale = max(1.0, float(fc.loc["total_assets"].abs().max()))
    out = {}
    out["forecast_max_abs_balance_check"] = float(fc.loc["balance_check", fy].abs().max()) if fy else 0.0
    out["forecast_max_abs_cf_sum_gap"] = float((fc.loc["cfo", fy] + fc.loc["cfi", fy] + fc.loc["cff", fy] - fc.loc["net_change_cash", fy]).abs().max()) if fy else 0.0
    cols = list(fc.columns)
    roll = [abs(fc.loc["cash", cols[i]] - fc.loc["cash", cols[i - 1]] - fc.loc["net_change_cash", cols[i]]) for i in range(1, len(cols)) if cols[i] in fy]
    out["forecast_max_abs_cash_roll_gap"] = float(max(roll)) if roll else 0.0
    out["hist_max_abs_balance_check"] = float(fc.loc["balance_check", hy].abs().max()) if hy else 0.0
    hroll = [abs(fc.loc["cash", cols[i]] - fc.loc["cash", cols[i - 1]] - fc.loc["net_change_cash", cols[i]]) for i in range(1, len(cols)) if cols[i] in hy]
    out["hist_max_abs_cash_roll_gap"] = float(max(hroll)) if hroll else 0.0
    out["ok"] = float(all(v <= tol * scale for v in (out["forecast_max_abs_balance_check"], out["forecast_max_abs_cf_sum_gap"], out["forecast_max_abs_cash_roll_gap"])))
    return out


# ----------------------------------------------------------------------------- scenarios and elasticity

import copy as _copy


def shifted_drivers(drv: Drivers, growth_delta: float = 0.0, margin_delta: float = 0.0, capex_delta: float = 0.0, taper: float = 1.0) -> Drivers:
    """A copy of the drivers with per-year shifts (decimal). ``taper`` < 1 shrinks the shift each year after the first."""
    d = _copy.deepcopy(drv)
    for i in range(len(d.years)):
        f = taper ** i
        d.revenue_growth[i] = d.revenue_growth[i] + growth_delta * f
        d.gross_margin[i] = min(max(d.gross_margin[i] + margin_delta * f, 0.0), 0.95)
        d.capex_pct[i] = max(d.capex_pct[i] + capex_delta * f, 0.0)
    return d


def scenario_values(hist: pd.DataFrame, drv: Drivers, wacc: float, terminal_growth: float, scenarios: dict[str, dict], mid_year: bool = False) -> dict:
    """Per-share value for each scenario. ``scenarios``: name -> {growth_delta, margin_delta, capex_delta, wacc_delta, tg_delta, prob}."""
    out = {}
    for name, sc in scenarios.items():
        d = shifted_drivers(drv, sc.get("growth_delta", 0.0), sc.get("margin_delta", 0.0), sc.get("capex_delta", 0.0), sc.get("taper", 1.0))
        fc = forecast(hist, d)
        v = dcf(fc, d, wacc + sc.get("wacc_delta", 0.0), terminal_growth + sc.get("tg_delta", 0.0), mid_year=mid_year).as_dict()
        out[name] = {"per_share": v["per_share"], "enterprise_value": v["enterprise_value"], "prob": sc.get("prob", 0.0),
                     "revenue_last": float(fc.loc["revenue"].iloc[-1]), "ebit_margin_last": float(fc.loc["operating_income"].iloc[-1] / fc.loc["revenue"].iloc[-1]),
                     "drivers": {"growth": [round(x, 4) for x in d.revenue_growth], "gross_margin": [round(x, 4) for x in d.gross_margin], "capex_pct": [round(x, 4) for x in d.capex_pct]},
                     **{k: sc.get(k, 0.0) for k in ("growth_delta", "margin_delta", "capex_delta", "wacc_delta", "tg_delta")}}
    probs = sum(v["prob"] for v in out.values())
    if probs > 0:
        out["_probability_weighted"] = {"per_share": sum(v["per_share"] * v["prob"] for v in out.values() if v["prob"]) / probs}
    return out


def elasticity(hist: pd.DataFrame, drv: Drivers, wacc: float, terminal_growth: float, mid_year: bool = False) -> list[dict]:
    """Per-driver sensitivity of per-share value: each driver shocked alone by a standard amount, all years."""
    base = dcf(forecast(hist, drv), drv, wacc, terminal_growth, mid_year=mid_year).as_dict()["per_share"]
    shocks = [("revenue_growth", 0.01, "pp all years"), ("gross_margin", 0.01, "pp all years"), ("sga_pct", -0.01, "pp all years"),
              ("capex_pct", -0.01, "pp all years"), ("dso", -5.0, "days"), ("dio", -5.0, "days"), ("dpo", 5.0, "days"), ("tax_rate", -0.01, "pp all years")]
    rows = []
    for key, shock, unit in shocks:
        d = _copy.deepcopy(drv)
        setattr(d, key, [x + shock for x in getattr(d, key)])
        v = dcf(forecast(hist, d), d, wacc, terminal_growth, mid_year=mid_year).as_dict()["per_share"]
        rows.append({"driver": key, "shock": shock, "unit": unit, "per_share": round(v, 3), "delta": round(v - base, 3),
                     "pct_change_in_value": round((v / base - 1) if base else float("nan"), 4)})
    for key, shock, unit in (("wacc", 0.01, "pp"), ("terminal_growth", 0.005, "pp")):
        w, g = (wacc + shock, terminal_growth) if key == "wacc" else (wacc, terminal_growth + shock)
        v = dcf(forecast(hist, drv), drv, w, g, mid_year=mid_year).as_dict()["per_share"]
        rows.append({"driver": key, "shock": shock, "unit": unit, "per_share": round(v, 3), "delta": round(v - base, 3),
                     "pct_change_in_value": round((v / base - 1) if base else float("nan"), 4)})
    rows.sort(key=lambda r: -abs(r["delta"]))
    return [{"base_per_share": round(base, 3), **r} for r in rows]

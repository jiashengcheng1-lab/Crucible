"""Write the model to an Excel workbook with live formulas.

Conventions (financial-model standard): blue = hardcoded input, black = formula,
green = link to another sheet, yellow fill = key assumption. Historical actuals
are inputs (blue) with their filing source on the Sources sheet; every forecast
cell is a formula that references the Inputs sheet, so the workbook recomputes
when an analyst changes an assumption. ``tests/test_excel.py`` recalculates the
file with LibreOffice and checks it agrees with ``model.forecast``/``model.dcf``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .model import PER_YEAR_DRIVERS, SCALAR_DRIVERS, Drivers, WACCInputs

BLUE = Font(name="Arial", color="0000FF")
BLACK = Font(name="Arial", color="000000")
GREEN = Font(name="Arial", color="008000")
BOLD = Font(name="Arial", bold=True)
TITLE = Font(name="Arial", bold=True, size=13)
YELLOW = PatternFill("solid", fgColor="FFFF00")
GREY = PatternFill("solid", fgColor="EEEEEE")
FMT_AMT = "#,##0;(#,##0);-"
FMT_PCT = "0.0%;(0.0%);-"
FMT_DAYS = "0.0"
FMT_PS = "0.00;(0.00);-"

MODEL_ROWS: list[tuple[str, str, str]] = [
    # key, label, kind (section|amount|pct|days|shares|ps)
    ("_IS", "INCOME STATEMENT", "section"),
    ("revenue", "Revenue", "amount"), ("cogs", "Cost of revenue", "amount"), ("gross_profit", "Gross profit", "amount"),
    ("sga", "SG&A", "amount"), ("rnd", "R&D", "amount"), ("other_opex", "Other operating expense (residual)", "amount"),
    ("operating_income", "Operating income (EBIT)", "amount"), ("d_and_a", "D&A (memo, from cash flow)", "amount"),
    ("ebitda", "EBITDA (memo)", "amount"), ("sbc", "Stock-based comp (memo)", "amount"),
    ("interest_expense", "Interest expense, net", "amount"), ("other_income", "Other income/(expense), net (residual)", "amount"),
    ("pretax_income", "Pre-tax income", "amount"), ("income_tax", "Income tax", "amount"), ("net_income", "Net income", "amount"),
    ("shares_diluted", "Diluted shares (mm)", "shares"), ("eps_diluted", "Diluted EPS", "ps"),
    ("_BS", "BALANCE SHEET", "section"),
    ("cash", "Cash & equivalents", "amount"), ("receivables", "Accounts receivable", "amount"), ("inventory", "Inventory", "amount"),
    ("other_current_assets", "Other current assets (residual)", "amount"), ("total_current_assets", "Total current assets", "amount"),
    ("ppe_net", "PP&E, net", "amount"), ("goodwill", "Goodwill", "amount"), ("intangibles", "Intangibles, net", "amount"),
    ("other_noncurrent_assets", "Other non-current assets (residual)", "amount"), ("total_assets", "Total assets", "amount"),
    ("payables", "Accounts payable", "amount"), ("other_current_liabilities", "Other current liabilities (residual)", "amount"),
    ("short_term_debt", "Short-term debt", "amount"), ("total_current_liabilities", "Total current liabilities", "amount"),
    ("long_term_debt", "Long-term debt", "amount"), ("other_noncurrent_liabilities", "Other non-current liabilities (residual)", "amount"),
    ("total_liabilities", "Total liabilities", "amount"), ("total_equity", "Total equity (incl. NCI)", "amount"),
    ("total_liabilities_and_equity", "Total liabilities & equity", "amount"), ("balance_check", "Balance check (should be 0)", "amount"),
    ("_CF", "CASH FLOW", "section"),
    ("cf_net_income", "Net income", "amount"), ("cf_da", "D&A", "amount"), ("cf_sbc", "Stock-based comp", "amount"),
    ("d_nwc", "Change in NWC & other (residual)", "amount"), ("cfo", "Cash from operations", "amount"),
    ("capex", "Capex (positive = outflow)", "amount"), ("other_investing", "Other investing (residual)", "amount"),
    ("cfi", "Cash from investing", "amount"), ("dividends", "Dividends paid (positive = outflow)", "amount"),
    ("buybacks", "Buybacks (positive = outflow)", "amount"), ("net_debt_issuance", "Net debt issued/(repaid) & other (residual)", "amount"),
    ("cff", "Cash from financing", "amount"), ("net_change_cash", "Net change in cash", "amount"),
    ("cash_roll_check", "Cash roll check (history: FX/restricted cash)", "amount"),
]

# Historical rows that are formulas (residuals/memos); everything else historical is a hardcoded actual.
HIST_FORMULA_ROWS = {"other_opex", "ebitda", "other_income", "other_current_assets", "other_noncurrent_assets",
                     "other_current_liabilities", "other_noncurrent_liabilities", "balance_check", "cf_da", "cf_sbc",
                     "d_nwc", "other_investing", "net_debt_issuance", "cash_roll_check"}


def _fmt(kind: str) -> str:
    return {"amount": FMT_AMT, "pct": FMT_PCT, "days": FMT_DAYS, "shares": FMT_AMT, "ps": FMT_PS}.get(kind, FMT_AMT)


def write_model(hist: pd.DataFrame, drv: Drivers, wacc_in: WACCInputs, terminal_growth: float, path: Path,
                company: str = "Company", units: str = "USD millions", sources: dict | None = None,
                mapping_notes: list[str] | None = None, mid_year: bool = False, scenarios: dict | None = None,
                elasticity_rows: list[dict] | None = None, audit: bool = False, template: dict | None = None) -> Path:
    """``template``: {"key", "inputs": driver paths, "build": DataFrame, "basis"} from crucible.templates; when given, a Drivers
    sheet carries the driver inputs (blue) and the build as formulas, and the model's growth, margin, D&A and capex inputs
    link to it, so the three statements run off the operating drivers."""
    hist_years = [int(c) for c in hist.columns]
    fc_years = list(drv.years)
    all_years = hist_years + fc_years
    first_col = 3  # column C
    col = {y: get_column_letter(first_col + i) for i, y in enumerate(all_years)}
    prev = {all_years[i]: all_years[i - 1] for i in range(1, len(all_years))}
    wb = Workbook()

    DRV_ROW: dict[str, int] = {}
    if template:
        keys = ["hashrate_eh", "network_hashrate_eh", "btc_price", "fee_share", "efficiency_j_th", "power_price_mwh", "other_cash_cogs_pct", "other_revenue_pct",
                "capex_per_eh", "btc_sold_pct", "issuance_btc", "network_share", "btc_mined", "mining_revenue", "revenue", "energy_mwh", "energy_cost",
                "other_cash_cogs", "capex", "d_and_a", "cogs", "gross_margin", "capex_pct", "da_pct", "revenue_growth", "btc_held_end", "treasury_value"]
        for i, k in enumerate(keys):
            DRV_ROW[k] = 5 + i
        DRV_ROW["fleet_life_years"] = 5 + len(keys) + 1
        DRV_ROW["btc_held"] = DRV_ROW["fleet_life_years"] + 1
        DRV_ROW["ppe0"] = DRV_ROW["fleet_life_years"] + 2
    SCN_ROW: dict[tuple[str, str], int] = {}
    if scenarios:
        names = [n for n in scenarios if not n.startswith("_")]
        r0 = 6
        for n in names:
            for key in ("revenue_growth", "gross_margin", "capex_pct"):
                SCN_ROW[(n, key)] = r0
                r0 += 1
            r0 += 1
    # ------------------------------------------------------------------ Inputs
    ws_in = wb.active
    ws_in.title = "Inputs"
    ws_in["A1"], ws_in["A1"].font = f"{company}: driver assumptions", TITLE
    if scenarios:
        names = [n for n in scenarios if not n.startswith("_")]
        ws_in["A1"] = f"{company}: driver assumptions   |   scenario selector in B1: " + ", ".join(f"{i + 1} = {n}" for i, n in enumerate(names))
        ws_in["B1"], ws_in["B1"].font, ws_in["B1"].fill = 1, BLUE, YELLOW
    ws_in["A2"] = f"Units: {units}. Blue = input, green = historical ratio linked from Model, yellow = key assumption."
    ws_in["A3"], ws_in["A3"].font = "Driver", BOLD
    ws_in["B3"], ws_in["B3"].font = "Unit", BOLD
    for y in all_years:
        c = ws_in[f"{col[y]}3"]
        c.value, c.font, c.alignment = (f"{y}A" if y in hist_years else f"{y}E"), BOLD, Alignment(horizontal="right")
    IR: dict[str, int] = {}
    r = 4
    for key, label, unit in PER_YEAR_DRIVERS:
        IR[key] = r
        ws_in[f"A{r}"], ws_in[f"B{r}"] = label, unit
        r += 1
    r += 1
    ws_in[f"A{r}"], ws_in[f"A{r}"].font = "Scalar drivers", BOLD
    r += 1
    for key, label, unit in SCALAR_DRIVERS:
        IR[key] = r
        ws_in[f"A{r}"], ws_in[f"B{r}"] = label, unit
        cell = ws_in[f"C{r}"]
        cell.value, cell.font, cell.number_format = float(getattr(drv, key)), BLUE, FMT_PCT
        r += 1
    r += 1
    ws_in[f"A{r}"] = "Basis for defaults: " + drv.notes.get("basis", "n/a")
    if drv.notes.get("shares"):
        r += 1
        ws_in[f"A{r}"] = "Share count: " + drv.notes["shares"]
    for k in ("other_opex", "capex"):
        if drv.notes.get(k):
            r += 1
            ws_in[f"A{r}"] = f"Default driver note ({k}): " + drv.notes[k]
    r += 1
    ws_in[f"A{r}"] = f"Working capital method: {drv.nwc_method} (days = DSO/DIO/DPO drivers; pct_revenue = ratios to revenue held at the last actual year)"
    IR["_end"] = r

    # ------------------------------------------------------------------ Model rows
    ws = wb.create_sheet("Model")
    ws["A1"], ws["A1"].font = f"{company}: three-statement model", TITLE
    ws["A2"] = f"Units: {units}. Historical actuals in blue (sources on the Sources sheet); every forecast cell is a formula."
    ws["A3"], ws["A3"].font = "Line item", BOLD
    for y in all_years:
        c = ws[f"{col[y]}3"]
        c.value, c.font, c.alignment = (f"{y}A" if y in hist_years else f"{y}E"), BOLD, Alignment(horizontal="right")
    MR: dict[str, int] = {}
    r = 4
    for key, label, kind in MODEL_ROWS:
        MR[key] = r
        ws[f"A{r}"] = label
        if kind == "section":
            ws[f"A{r}"].font = BOLD
            for y in all_years:
                ws[f"{col[y]}{r}"].fill = GREY
            ws[f"A{r}"].fill = GREY
        r += 1

    def M(key: str, y: int) -> str:  # same-sheet reference
        return f"{col[y]}{MR[key]}"

    def I(key: str, y: int | None = None) -> str:  # cross-sheet reference to Inputs
        return f"Inputs!$C${IR[key]}" if y is None else f"Inputs!{col[y]}{IR[key]}"

    # Historical values / formulas
    for y in hist_years:
        p = prev.get(y)
        for key, _, kind in MODEL_ROWS:
            if kind == "section":
                continue
            cell = ws[M(key, y)]
            cell.number_format = _fmt(kind)
            if key in HIST_FORMULA_ROWS:
                cell.font = BLACK
                cell.value = {
                    "other_opex": f"={M('gross_profit', y)}-{M('sga', y)}-{M('rnd', y)}-{M('operating_income', y)}",
                    "ebitda": f"={M('operating_income', y)}+{M('d_and_a', y)}",
                    "other_income": f"={M('pretax_income', y)}-{M('operating_income', y)}+{M('interest_expense', y)}",
                    "other_current_assets": f"={M('total_current_assets', y)}-{M('cash', y)}-{M('receivables', y)}-{M('inventory', y)}",
                    "other_noncurrent_assets": f"={M('total_assets', y)}-{M('total_current_assets', y)}-{M('ppe_net', y)}-{M('goodwill', y)}-{M('intangibles', y)}",
                    "other_current_liabilities": f"={M('total_current_liabilities', y)}-{M('payables', y)}-{M('short_term_debt', y)}",
                    "other_noncurrent_liabilities": f"={M('total_liabilities', y)}-{M('total_current_liabilities', y)}-{M('long_term_debt', y)}",
                    "balance_check": f"={M('total_assets', y)}-{M('total_liabilities_and_equity', y)}",
                    "cf_da": f"={M('d_and_a', y)}",
                    "cf_sbc": f"={M('sbc', y)}",
                    "d_nwc": f"={M('cfo', y)}-{M('cf_net_income', y)}-{M('cf_da', y)}-{M('cf_sbc', y)}",
                    "other_investing": f"={M('cfi', y)}+{M('capex', y)}",
                    "net_debt_issuance": f"={M('cff', y)}+{M('dividends', y)}+{M('buybacks', y)}",
                    "cash_roll_check": (f"={M('cash', p)}+{M('net_change_cash', y)}-{M('cash', y)}" if p else "=0"),
                }[key]
            else:
                cell.font = BLUE
                cell.value = float(hist.loc[key, y]) if key in hist.index else 0.0

    # Forecast formulas
    for y in fc_years:
        p = prev[y]
        f = {
            "revenue": f"={M('revenue', p)}*(1+{I('revenue_growth', y)})",
            "cogs": f"={M('revenue', y)}*(1-{I('gross_margin', y)})",
            "gross_profit": f"={M('revenue', y)}-{M('cogs', y)}",
            "sga": f"={M('revenue', y)}*{I('sga_pct', y)}",
            "rnd": f"={M('revenue', y)}*{I('rnd_pct', y)}",
            "other_opex": f"={M('revenue', y)}*{I('other_opex_pct', y)}",
            "operating_income": f"={M('gross_profit', y)}-{M('sga', y)}-{M('rnd', y)}-{M('other_opex', y)}",
            "d_and_a": f"={M('revenue', y)}*{I('da_pct', y)}",
            "ebitda": f"={M('operating_income', y)}+{M('d_and_a', y)}",
            "sbc": f"={M('revenue', y)}*{I('sbc_pct', y)}",
            "interest_expense": f"=({M('short_term_debt', p)}+{M('long_term_debt', p)})*{I('interest_rate_debt')}-{M('cash', p)}*{I('interest_rate_cash')}",
            "other_income": f"={I('other_income', y)}",
            "pretax_income": f"={M('operating_income', y)}-{M('interest_expense', y)}+{M('other_income', y)}",
            "income_tax": f"={M('pretax_income', y)}*{I('tax_rate', y)}",
            "net_income": f"={M('pretax_income', y)}-{M('income_tax', y)}",
            "shares_diluted": f"={M('shares_diluted', p)}",
            "eps_diluted": f"=IF({M('shares_diluted', y)}=0,0,{M('net_income', y)}/{M('shares_diluted', y)})",
            "cash": f"={M('cash', p)}+{M('net_change_cash', y)}",
            "receivables": f"={M('revenue', y)}*{I('dso', y)}/365",
            "inventory": f"={M('cogs', y)}*{I('dio', y)}/365",
            "other_current_assets": f"={M('other_current_assets', p)}",
            "total_current_assets": f"={M('cash', y)}+{M('receivables', y)}+{M('inventory', y)}+{M('other_current_assets', y)}",
            "ppe_net": f"={M('ppe_net', p)}+{M('capex', y)}-{M('d_and_a', y)}",
            "goodwill": f"={M('goodwill', p)}",
            "intangibles": f"={M('intangibles', p)}",
            "other_noncurrent_assets": f"={M('other_noncurrent_assets', p)}",
            "total_assets": f"={M('total_current_assets', y)}+{M('ppe_net', y)}+{M('goodwill', y)}+{M('intangibles', y)}+{M('other_noncurrent_assets', y)}",
            "payables": f"={M('cogs', y)}*{I('dpo', y)}/365",
            "other_current_liabilities": f"={M('other_current_liabilities', p)}",
            "short_term_debt": f"={M('short_term_debt', p)}",
            "total_current_liabilities": f"={M('payables', y)}+{M('other_current_liabilities', y)}+{M('short_term_debt', y)}",
            "long_term_debt": f"={M('long_term_debt', p)}+{M('net_debt_issuance', y)}",
            "other_noncurrent_liabilities": f"={M('other_noncurrent_liabilities', p)}",
            "total_liabilities": f"={M('total_current_liabilities', y)}+{M('long_term_debt', y)}+{M('other_noncurrent_liabilities', y)}",
            "total_equity": f"={M('total_equity', p)}+{M('net_income', y)}+{M('sbc', y)}-{M('dividends', y)}-{M('buybacks', y)}",
            "total_liabilities_and_equity": f"={M('total_liabilities', y)}+{M('total_equity', y)}",
            "balance_check": f"={M('total_assets', y)}-{M('total_liabilities_and_equity', y)}",
            "cf_net_income": f"={M('net_income', y)}",
            "cf_da": f"={M('d_and_a', y)}",
            "cf_sbc": f"={M('sbc', y)}",
            "d_nwc": f"=-(({M('receivables', y)}-{M('receivables', p)})+({M('inventory', y)}-{M('inventory', p)})-({M('payables', y)}-{M('payables', p)}))",
            "cfo": f"={M('cf_net_income', y)}+{M('cf_da', y)}+{M('cf_sbc', y)}+{M('d_nwc', y)}",
            "capex": f"={M('revenue', y)}*{I('capex_pct', y)}",
            "cfi": f"=-{M('capex', y)}+{M('other_investing', y)}",
            "dividends": f"={I('dividends', y)}",
            "buybacks": f"={I('buybacks', y)}",
            "net_debt_issuance": f"={I('net_debt_issuance', y)}",
            "cff": f"=-{M('dividends', y)}-{M('buybacks', y)}+{M('net_debt_issuance', y)}",
            "net_change_cash": f"={M('cfo', y)}+{M('cfi', y)}+{M('cff', y)}",
            "cash_roll_check": f"={M('cash', p)}+{M('net_change_cash', y)}-{M('cash', y)}",
        }
        for key, _, kind in MODEL_ROWS:
            if kind == "section":
                continue
            cell = ws[M(key, y)]
            cell.number_format = _fmt(kind)
            if key == "other_investing":
                cell.value, cell.font = 0.0, BLUE  # input: no other investing in forecast by default
            else:
                cell.value, cell.font = f[key], BLACK
    ws.column_dimensions["A"].width = 44
    ws.freeze_panes = "C4"

    # Inputs: historical ratios (green links) and forecast inputs (blue)
    def Mx(key: str, y: int) -> str:
        return f"Model!{col[y]}{MR[key]}"

    for key, label, unit in PER_YEAR_DRIVERS:
        rr = IR[key]
        for y in hist_years:
            p = prev.get(y)
            cell = ws_in[f"{col[y]}{rr}"]
            cell.font = GREEN
            cell.number_format = {"pct": FMT_PCT, "days": FMT_DAYS, "amount": FMT_AMT}[unit]
            rev = Mx("revenue", y)
            cogs = Mx("cogs", y)
            cell.value = {
                "revenue_growth": (f"=IF({Mx('revenue', p)}=0,0,{rev}/{Mx('revenue', p)}-1)" if p else None),
                "gross_margin": f"=IF({rev}=0,0,({rev}-{cogs})/{rev})",
                "sga_pct": f"=IF({rev}=0,0,{Mx('sga', y)}/{rev})",
                "rnd_pct": f"=IF({rev}=0,0,{Mx('rnd', y)}/{rev})",
                "other_opex_pct": f"=IF({rev}=0,0,{Mx('other_opex', y)}/{rev})",
                "da_pct": f"=IF({rev}=0,0,{Mx('d_and_a', y)}/{rev})",
                "sbc_pct": f"=IF({rev}=0,0,{Mx('sbc', y)}/{rev})",
                "capex_pct": f"=IF({rev}=0,0,{Mx('capex', y)}/{rev})",
                "tax_rate": f"=IF({Mx('pretax_income', y)}=0,0,{Mx('income_tax', y)}/{Mx('pretax_income', y)})",
                "dso": f"=IF({rev}=0,0,{Mx('receivables', y)}/{rev}*365)",
                "dio": f"=IF({cogs}=0,0,{Mx('inventory', y)}/{cogs}*365)",
                "dpo": f"=IF({cogs}=0,0,{Mx('payables', y)}/{cogs}*365)",
                "other_income": f"={Mx('other_income', y)}",
                "dividends": f"={Mx('dividends', y)}",
                "buybacks": f"={Mx('buybacks', y)}",
                "net_debt_issuance": f"={Mx('net_debt_issuance', y)}",
            }[key]
        vals = getattr(drv, key)
        for i, y in enumerate(fc_years):
            cell = ws_in[f"{col[y]}{rr}"]
            if scenarios and key in ("revenue_growth", "gross_margin", "capex_pct"):
                names = [n for n in scenarios if not n.startswith("_")]
                refs = ",".join(f"Scenarios!{col[y]}{SCN_ROW[(n, key)]}" for n in names)
                cell.value, cell.font = f"=CHOOSE(Inputs!$B$1,{refs})", BLUE
            elif template and key in DRV_ROW and key in ("revenue_growth", "gross_margin", "capex_pct", "da_pct"):
                cell.value, cell.font = f"=Drivers!{col[y]}{DRV_ROW[key]}", GREEN
            else:
                cell.value, cell.font = float(vals[i]), BLUE
            cell.number_format = {"pct": FMT_PCT, "days": FMT_DAYS, "amount": FMT_AMT}[unit]
            if key in ("revenue_growth", "gross_margin", "tax_rate"):
                cell.fill = YELLOW
    ws_in.column_dimensions["A"].width = 34
    ws_in.freeze_panes = "C4"

    # ------------------------------------------------------------------ DCF
    wd = wb.create_sheet("DCF")
    wd["A1"], wd["A1"].font = f"{company}: DCF (unlevered, end-of-year discounting unless mid-year flag = 1)", TITLE
    inputs = [
        ("Risk-free rate", wacc_in.risk_free, FMT_PCT), ("Beta", wacc_in.beta, "0.00"), ("Equity risk premium", wacc_in.erp, FMT_PCT),
        ("Size premium", wacc_in.size_premium, FMT_PCT), ("Pre-tax cost of debt", wacc_in.pretax_cost_of_debt, FMT_PCT),
        ("Tax rate (for WACC)", wacc_in.tax_rate, FMT_PCT), ("Debt weight D/(D+E)", wacc_in.debt_weight, FMT_PCT),
    ]
    for i, (lab, val, fmt) in enumerate(inputs, start=3):
        wd[f"A{i}"], wd[f"B{i}"] = lab, float(val)
        wd[f"B{i}"].font, wd[f"B{i}"].number_format = BLUE, fmt
    wd["A10"], wd["B10"] = "Cost of equity", "=B3+B4*B5+B6"
    wd["A11"], wd["B11"] = "WACC", "=(1-B9)*B10+B9*B7*(1-B8)"
    wd["A12"], wd["B12"] = "Terminal growth", float(terminal_growth)
    wd["A13"], wd["B13"] = "Mid-year convention (1 = yes)", 1.0 if mid_year else 0.0
    for a in ("B10", "B11", "B12"):
        wd[a].number_format = FMT_PCT
    wd["B12"].font, wd["B13"].font = BLUE, BLUE
    wd["B12"].fill, wd["B11"].fill = YELLOW, YELLOW
    wd["A15"], wd["A15"].font = "Free cash flow build", BOLD
    for y in fc_years:
        c = wd[f"{col[y]}15"]
        c.value, c.font, c.alignment = f"{y}E", BOLD, Alignment(horizontal="right")
    DR = {"ebit": 16, "tax": 17, "nopat": 18, "da": 19, "capex": 20, "dnwc": 21, "ufcf": 22, "t": 23, "df": 24, "pv": 25}
    labels = {"ebit": "EBIT", "tax": "Less: taxes on EBIT", "nopat": "NOPAT", "da": "Plus: D&A", "capex": "Less: capex",
              "dnwc": "Less: increase in NWC", "ufcf": "Unlevered FCF", "t": "Period (years)", "df": "Discount factor", "pv": "PV of FCF"}
    for k, rr in DR.items():
        wd[f"A{rr}"] = labels[k]
    for i, y in enumerate(fc_years, start=1):
        c = col[y]
        wd[f"{c}{DR['ebit']}"] = f"=Model!{c}{MR['operating_income']}"
        wd[f"{c}{DR['tax']}"] = f"={c}{DR['ebit']}*Inputs!{c}{IR['tax_rate']}"
        wd[f"{c}{DR['nopat']}"] = f"={c}{DR['ebit']}-{c}{DR['tax']}"
        wd[f"{c}{DR['da']}"] = f"=Model!{c}{MR['d_and_a']}"
        wd[f"{c}{DR['capex']}"] = f"=Model!{c}{MR['capex']}"
        wd[f"{c}{DR['dnwc']}"] = f"=-Model!{c}{MR['d_nwc']}"
        wd[f"{c}{DR['ufcf']}"] = f"={c}{DR['nopat']}+{c}{DR['da']}-{c}{DR['capex']}-{c}{DR['dnwc']}"
        wd[f"{c}{DR['t']}"] = f"={i}-0.5*$B$13"
        wd[f"{c}{DR['df']}"] = f"=1/(1+$B$11)^{c}{DR['t']}"
        wd[f"{c}{DR['pv']}"] = f"={c}{DR['ufcf']}*{c}{DR['df']}"
        for k in ("ebit", "tax", "nopat", "da", "capex", "dnwc", "ufcf", "pv"):
            wd[f"{c}{DR[k]}"].number_format = FMT_AMT
            wd[f"{c}{DR[k]}"].font = GREEN if k in ("ebit", "da", "capex", "dnwc") else BLACK
        wd[f"{c}{DR['df']}"].number_format = "0.000"
    fc0, fcN = col[fc_years[0]], col[fc_years[-1]]
    ly = hist_years[-1]
    wd["A27"], wd["B27"] = "Sum of PV of FCF", f"=SUM({fc0}{DR['pv']}:{fcN}{DR['pv']})"
    wd["A28"], wd["B28"] = "Terminal value (Gordon)", f"={fcN}{DR['ufcf']}*(1+B12)/(B11-B12)"
    wd["A29"], wd["B29"] = "PV of terminal value", f"=B28*{fcN}{DR['df']}"
    wd["A30"], wd["B30"] = "Enterprise value", "=B27+B29"
    wd["A31"], wd["B31"] = f"Net debt (FY{ly}A)", f"=Model!{col[ly]}{MR['short_term_debt']}+Model!{col[ly]}{MR['long_term_debt']}-Model!{col[ly]}{MR['cash']}"
    wd["A36"], wd["B36"] = "Non-operating assets (added to equity; a bitcoin treasury at the price path)", (f"=Drivers!{col[fc_years[-1]]}{DRV_ROW['treasury_value']}" if template else float(getattr(drv, "non_operating_assets", 0.0) or 0.0))
    wd["B36"].number_format, wd["B36"].font = FMT_AMT, (GREEN if template else BLUE)
    wd["A32"], wd["B32"] = "Equity value", "=B30-B31+B36"
    if drv.shares_override:
        wd["A33"], wd["B33"] = "Diluted shares (treasury stock method at the as-of price; see Inputs note)", float(drv.shares_override)
    else:
        wd["A33"], wd["B33"] = f"Diluted shares (FY{ly}A)", f"=Model!{col[ly]}{MR['shares_diluted']}"
    wd["A34"], wd["B34"] = "Implied value per share", "=IF(B33=0,0,B32/B33)"
    wd["A35"], wd["B35"] = "Terminal value share of EV", "=IF(B30=0,0,B29/B30)"
    for a, fmt in (("B27", FMT_AMT), ("B28", FMT_AMT), ("B29", FMT_AMT), ("B30", FMT_AMT), ("B31", FMT_AMT), ("B32", FMT_AMT),
                   ("B33", FMT_AMT), ("B34", FMT_PS), ("B35", FMT_PCT)):
        wd[a].number_format = fmt
    wd["B31"].font, wd["B33"].font = GREEN, (BLUE if drv.shares_override else GREEN)
    wd["B34"].font = Font(name="Arial", bold=True)

    # Sensitivity: value per share across WACC (rows) x terminal growth (cols), all formulas
    wd["A37"], wd["A37"].font = "Sensitivity: value per share (rows = WACC, columns = terminal growth)", BOLD
    base_w, base_g = wacc_in.wacc, terminal_growth
    waccs = [base_w + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    growths = [base_g + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    hdr_row, first_row = 38, 39
    for j, g in enumerate(growths):
        c = get_column_letter(3 + j)
        wd[f"{c}{hdr_row}"] = float(g)
        wd[f"{c}{hdr_row}"].number_format, wd[f"{c}{hdr_row}"].font = FMT_PCT, BLUE
    ufcf_rng = f"${fc0}${DR['ufcf']}:${fcN}${DR['ufcf']}"
    t_rng = f"${fc0}${DR['t']}:${fcN}${DR['t']}"
    for i, w in enumerate(waccs):
        rr = first_row + i
        wd[f"B{rr}"] = float(w)
        wd[f"B{rr}"].number_format, wd[f"B{rr}"].font = FMT_PCT, BLUE
        for j in range(len(growths)):
            c = get_column_letter(3 + j)
            pv = f"SUMPRODUCT({ufcf_rng},1/(1+$B{rr})^{t_rng})"
            tv = f"${fcN}${DR['ufcf']}*(1+{c}${hdr_row})/($B{rr}-{c}${hdr_row})/(1+$B{rr})^${fcN}${DR['t']}"
            wd[f"{c}{rr}"] = f'=IF($B{rr}<={c}${hdr_row},"n/a",IF($B$33=0,0,({pv}+{tv}-$B$31)/$B$33))'
            wd[f"{c}{rr}"].number_format = FMT_PS
    wd.column_dimensions["A"].width = 40

    # ------------------------------------------------------------------ Drivers (industry template build, formulas)
    if template:
        wdr = wb.create_sheet("Drivers", 1)
        tin, tb = template["inputs"], template["build"]
        wdr["A1"], wdr["A1"].font = f"{company}: {template.get('label', template['key'])} driver build (blue = driver input, black = formula)", TITLE
        wdr["A2"] = "Basis: " + template.get("basis", "")
        wdr["A3"] = ("btc_mined = hashrate / network hashrate x issuance x (1 + fee share); revenue = btc_mined x price / (1 - other revenue share); "
                     "energy MWh = EH/s x J/TH x 8760; capex = (new EH/s + retiring fleet) x $m per EH/s; D&A straight-line over the fleet life")
        wdr["A4"], wdr["A4"].font = "driver", BOLD
        wdr["B4"], wdr["B4"].font = "unit", BOLD
        for y in all_years:
            c = wdr[f"{col[y]}4"]
            c.value, c.font, c.alignment = (f"{y}A" if y in hist_years else f"{y}E"), BOLD, Alignment(horizontal="right")
        lyc = col[hist_years[-1]]
        labels = {"hashrate_eh": ("average energized hashrate", "EH/s"), "network_hashrate_eh": ("effective network hashrate", "EH/s"), "btc_price": ("realized bitcoin price", "USD"),
                  "fee_share": ("fees as share of subsidy", "pct"), "efficiency_j_th": ("fleet efficiency", "J/TH"), "power_price_mwh": ("blended power + hosting cost", "USD/MWh"),
                  "other_cash_cogs_pct": ("other cash cost of revenue", "% revenue"), "other_revenue_pct": ("non-mining revenue share", "% revenue"),
                  "capex_per_eh": ("capex per EH/s added", "USD m"), "btc_sold_pct": ("share of production sold", "pct"), "issuance_btc": ("network issuance (protocol)", "BTC"),
                  "network_share": ("network share", "pct"), "btc_mined": ("bitcoin mined", "BTC"), "mining_revenue": ("mining revenue", units), "revenue": ("total revenue", units),
                  "energy_mwh": ("energy consumed", "MWh"), "energy_cost": ("energy + hosting cost", units), "other_cash_cogs": ("other cash cost of revenue", units),
                  "capex": ("capex", units), "d_and_a": ("depreciation", units), "cogs": ("cost of revenue incl. D&A", units), "gross_margin": ("gross margin", "pct"),
                  "capex_pct": ("capex % revenue", "pct"), "da_pct": ("D&A % revenue", "pct"), "revenue_growth": ("revenue growth", "pct"),
                  "btc_held_end": ("bitcoin held, end", "BTC"), "treasury_value": ("treasury value", units)}
        for k, (lab, unit) in labels.items():
            wdr[f"A{DRV_ROW[k]}"], wdr[f"B{DRV_ROW[k]}"] = lab, unit
        R = DRV_ROW
        wdr[f"A{R['fleet_life_years']}"], wdr[f"B{R['fleet_life_years']}"] = "fleet life", "years"
        wdr[f"C{R['fleet_life_years']}"], wdr[f"C{R['fleet_life_years']}"].font = float(tin.get("fleet_life_years", 3.0)), BLUE
        wdr[f"A{R['btc_held']}"], wdr[f"B{R['btc_held']}"] = "bitcoin held at the last balance date", "BTC"
        wdr[f"C{R['btc_held']}"], wdr[f"C{R['btc_held']}"].font = float(tin.get("btc_held", 0.0)), BLUE
        wdr[f"A{R['ppe0']}"], wdr[f"B{R['ppe0']}"] = "PP&E at the last actual (legacy fleet, depreciated over the fleet life)", units
        wdr[f"C{R['ppe0']}"] = f"=Model!{lyc}{MR['ppe_net']}"
        life_n = max(1, int(round(float(tin.get("fleet_life_years", 3.0)))))
        L = f"$C${R['fleet_life_years']}"
        # last-actual reference values in the last historical column
        last_hr = float(template.get("last_hashrate", tin["hashrate_eh"][0] / (1 + template.get("hashrate_g0", 0.0)))) if template.get("last_hashrate") else None
        if last_hr:
            wdr[f"{lyc}{R['hashrate_eh']}"], wdr[f"{lyc}{R['hashrate_eh']}"].font = last_hr, BLUE
        wdr[f"{lyc}{R['revenue']}"] = f"=Model!{lyc}{MR['revenue']}"
        for i, y in enumerate(fc_years):
            c = col[y]
            pc = col[fc_years[i - 1]] if i else lyc
            for k in ("hashrate_eh", "network_hashrate_eh", "btc_price", "fee_share", "efficiency_j_th", "power_price_mwh", "other_cash_cogs_pct", "other_revenue_pct", "capex_per_eh", "btc_sold_pct"):
                cell = wdr[f"{c}{R[k]}"]
                cell.value, cell.font = float(tin[k][i]), BLUE
                cell.number_format = FMT_PCT if k in ("fee_share", "other_cash_cogs_pct", "other_revenue_pct", "btc_sold_pct") else "#,##0.0"
            wdr[f"{c}{R['issuance_btc']}"] = float(tb.at[y, "issuance_btc"])
            wdr[f"{c}{R['network_share']}"] = f"=IF({c}{R['network_hashrate_eh']}=0,0,{c}{R['hashrate_eh']}/{c}{R['network_hashrate_eh']})"
            wdr[f"{c}{R['btc_mined']}"] = f"={c}{R['network_share']}*{c}{R['issuance_btc']}*(1+{c}{R['fee_share']})"
            wdr[f"{c}{R['mining_revenue']}"] = f"={c}{R['btc_mined']}*{c}{R['btc_price']}/1000000"
            wdr[f"{c}{R['revenue']}"] = f"=IF({c}{R['other_revenue_pct']}>=1,{c}{R['mining_revenue']},{c}{R['mining_revenue']}/(1-{c}{R['other_revenue_pct']}))"
            wdr[f"{c}{R['energy_mwh']}"] = f"={c}{R['hashrate_eh']}*{c}{R['efficiency_j_th']}*8760"
            wdr[f"{c}{R['energy_cost']}"] = f"={c}{R['energy_mwh']}*{c}{R['power_price_mwh']}/1000000"
            wdr[f"{c}{R['other_cash_cogs']}"] = f"={c}{R['revenue']}*{c}{R['other_cash_cogs_pct']}"
            prev_hr = f"{pc}{R['hashrate_eh']}" if (i or last_hr) else f"{c}{R['hashrate_eh']}"
            wdr[f"{c}{R['capex']}"] = f"=(MAX({c}{R['hashrate_eh']}-{prev_hr},0)+{prev_hr}/{L})*{c}{R['capex_per_eh']}"
            window = [col[fc_years[j]] for j in range(max(0, i - life_n + 1), i + 1)]
            legacy = f"IF({i + 1}<={L},$C${R['ppe0']}/{L},0)"
            wdr[f"{c}{R['d_and_a']}"] = f"={legacy}+SUM({window[0]}{R['capex']}:{window[-1]}{R['capex']})/{L}"
            wdr[f"{c}{R['cogs']}"] = f"={c}{R['energy_cost']}+{c}{R['other_cash_cogs']}+{c}{R['d_and_a']}"
            wdr[f"{c}{R['gross_margin']}"] = f"=IF({c}{R['revenue']}=0,0,({c}{R['revenue']}-{c}{R['cogs']})/{c}{R['revenue']})"
            wdr[f"{c}{R['capex_pct']}"] = f"=IF({c}{R['revenue']}=0,0,{c}{R['capex']}/{c}{R['revenue']})"
            wdr[f"{c}{R['da_pct']}"] = f"=IF({c}{R['revenue']}=0,0,{c}{R['d_and_a']}/{c}{R['revenue']})"
            wdr[f"{c}{R['revenue_growth']}"] = f"=IF({pc}{R['revenue']}=0,0,{c}{R['revenue']}/{pc}{R['revenue']}-1)"
            held_prev = f"{pc}{R['btc_held_end']}" if i else f"$C${R['btc_held']}"
            wdr[f"{c}{R['btc_held_end']}"] = f"={held_prev}+{c}{R['btc_mined']}*(1-{c}{R['btc_sold_pct']})"
            wdr[f"{c}{R['treasury_value']}"] = f"={c}{R['btc_held_end']}*{c}{R['btc_price']}/1000000"
            for k in ("network_share", "gross_margin", "capex_pct", "da_pct", "revenue_growth"):
                wdr[f"{c}{R[k]}"].number_format = FMT_PCT
            for k in ("btc_mined", "issuance_btc", "energy_mwh", "btc_held_end"):
                wdr[f"{c}{R[k]}"].number_format = "#,##0"
            for k in ("mining_revenue", "revenue", "energy_cost", "other_cash_cogs", "capex", "d_and_a", "cogs", "treasury_value"):
                wdr[f"{c}{R[k]}"].number_format = FMT_AMT
        wdr[f"A{R['ppe0'] + 2}"] = (f"D&A window is the fleet life at build time ({life_n} years of capex); change the life cell and rebuild for a different window. "
                                     "Energy uses the blended $/MWh: owned-site power plus third-party hosting per MWh consumed.")
        wdr.column_dimensions["A"].width = 46
        wdr.freeze_panes = "C5"

    # ------------------------------------------------------------------ Scenarios / Elasticity / Audit
    if scenarios:
        wsc = wb.create_sheet("Scenarios")
        names = [n for n in scenarios if not n.startswith("_")]
        wsc["A1"], wsc["A1"].font = f"{company}: scenario driver sets (Inputs!B1 selects which set the Model and DCF use; per-share values below are engine-computed for reference)", TITLE
        wsc["A3"], wsc["A3"].font = "scenario / driver", BOLD
        for y in fc_years:
            c = wsc[f"{col[y]}3"]
            c.value, c.font, c.alignment = f"{y}E", BOLD, Alignment(horizontal="right")
        for n in names:
            sc = scenarios[n]
            for key, label in (("revenue_growth", "revenue growth"), ("gross_margin", "gross margin"), ("capex_pct", "capex % revenue")):
                rr = SCN_ROW[(n, key)]
                wsc[f"A{rr}"] = f"{n}: {label}"
                series = sc["drivers"]["growth" if key == "revenue_growth" else ("gross_margin" if key == "gross_margin" else "capex_pct")]
                delta_key = {"revenue_growth": "growth_delta", "gross_margin": "margin_delta", "capex_pct": "capex_delta"}[key]
                for i, y in enumerate(fc_years):
                    cell = wsc[f"{col[y]}{rr}"]
                    if template and key in DRV_ROW:
                        shift = float(sc.get(delta_key, 0.0)) * (float(sc.get("taper", 1.0)) ** i)
                        cell.value, cell.font = (f"=Drivers!{col[y]}{DRV_ROW[key]}" + (f"+({shift:.6f})" if abs(shift) > 1e-12 else "")), GREEN
                    else:
                        cell.value, cell.font = float(series[i]), BLUE
                    cell.number_format = FMT_PCT
            wsc[f"A{SCN_ROW[(n, 'capex_pct')] + 1}"] = (f"{n}: engine per-share {sc['per_share']:.2f} | prob {sc.get('prob', 0):.0%} | "
                                                      f"shifts: growth {sc.get('growth_delta', 0):+.1%}, margin {sc.get('margin_delta', 0):+.1%}, WACC {sc.get('wacc_delta', 0):+.1%}")
        pw = scenarios.get("_probability_weighted")
        if pw:
            rr = max(SCN_ROW.values()) + 3
            wsc[f"A{rr}"], wsc[f"A{rr}"].font = f"probability-weighted per-share (engine): {pw['per_share']:.2f}", BOLD
            wsc[f"A{rr + 1}"] = "live selected-scenario per-share (formula):"
            wsc[f"B{rr + 1}"], wsc[f"B{rr + 1}"].number_format = "=DCF!B34", FMT_PS
        wsc.column_dimensions["A"].width = 60
    if elasticity_rows:
        wel = wb.create_sheet("Elasticity")
        wel["A1"], wel["A1"].font = f"{company}: per-driver sensitivity of per-share value (engine-computed, each driver shocked alone)", TITLE
        hdr = ["driver", "shock", "unit", "per-share after shock", "delta vs base", "% change in value"]
        for j, h in enumerate(hdr, start=1):
            c = wel.cell(row=3, column=j, value=h)
            c.font = BOLD
        wel["A2"] = f"base per-share {elasticity_rows[0]['base_per_share']:.2f}"
        for i, r in enumerate(elasticity_rows, start=4):
            wel.cell(row=i, column=1, value=r["driver"]); wel.cell(row=i, column=2, value=r["shock"]); wel.cell(row=i, column=3, value=r["unit"])
            wel.cell(row=i, column=4, value=r["per_share"]); wel.cell(row=i, column=5, value=r["delta"])
            c = wel.cell(row=i, column=6, value=r["pct_change_in_value"]); c.number_format = FMT_PCT
        wel.column_dimensions["A"].width = 20
    if audit:
        wa = wb.create_sheet("Audit")
        wa["A1"], wa["A1"].font = "Linkage checks: every check row and the formulas behind it (sheet, cell, formula)", TITLE
        wa.append(["sheet", "row label", "cell", "formula"])
        for c in wa[2]:
            c.font = BOLD
        for sheet in ("Model", "DCF"):
            wsx = wb[sheet]
            for row in wsx.iter_rows(min_row=1, max_row=wsx.max_row):
                label = str(row[0].value or "")
                if any(k in label.lower() for k in ("check", "roll", "sum of", "total assets", "total liabilities", "ending cash", "net change", "enterprise value", "per share")):
                    for cell in row[1:]:
                        if isinstance(cell.value, str) and cell.value.startswith("="):
                            wa.append([sheet, label, cell.coordinate, cell.value])
        wa.column_dimensions["B"].width = 36
        wa.column_dimensions["D"].width = 90

    # ------------------------------------------------------------------ Sources / README
    wsrc = wb.create_sheet("Sources")
    wsrc["A1"], wsrc["A1"].font = "Data provenance", TITLE
    wsrc["A2"] = f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by crucible. Units: {units}."
    wsrc["A3"], wsrc["B3"] = "Fiscal year", "Source (form, accession, filing date)"
    wsrc["A3"].font = wsrc["B3"].font = BOLD
    r = 4
    for y in hist_years:
        wsrc[f"A{r}"], wsrc[f"B{r}"] = str(y), (sources or {}).get(str(y), "see data/<ticker>/meta.json")
        r += 1
    r += 1
    wsrc[f"A{r}"], wsrc[f"A{r}"].font = "Mapping notes (residual lines absorb what the standard schema does not map)", BOLD
    for note in (mapping_notes or []):
        r += 1
        wsrc[f"A{r}"] = note
    wsrc.column_dimensions["A"].width = 16
    wsrc.column_dimensions["B"].width = 90
    wr = wb.create_sheet("README", 0)
    wr["A1"], wr["A1"].font = "How to use this workbook", TITLE
    lines = [
        "Inputs: change blue cells in the forecast columns. Yellow cells are the assumptions the debate engine stress-tests.",
        "Model: historical actuals are blue (hardcoded from filings, see Sources); forecast columns are formulas only.",
        "DCF: WACC inputs in B3:B9, terminal growth in B12, mid-year flag in B13. Sensitivity grid at the bottom is fully formula-driven.",
        "Conventions: interest is charged on beginning-of-year balances (no circularity). Residual lines make history tie to reported totals.",
        "Sign conventions: capex, dividends and buybacks are shown as positive outflows; 'Change in NWC & other' is a cash effect.",
    ]
    for i, line in enumerate(lines, start=3):
        wr[f"A{i}"] = line
    wr.column_dimensions["A"].width = 120
    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.font is None or cell.font.name != "Arial":
                    cell.font = Font(name="Arial", bold=cell.font.bold if cell.font else False,
                                     color=cell.font.color if cell.font else None, size=cell.font.size if cell.font else 11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path

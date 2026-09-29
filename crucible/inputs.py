"""Analyst inputs: the analyst's entry point, filled from data instead of typed in.

data/<TICKER>/analyst_inputs.json holds, per field, a value, its citation and the date it was derived. Fields:
  risk_free      FRED DGS10 (10-year Treasury), latest observation on or before the as-of date
  beta           OLS of 2-year weekly returns vs the S&P 500 (Yahoo Finance), plus the Blume-adjusted beta and r2
  peer_betas     same regression for each peer
  erp            market-wide; not in any filing. Damodaran's implied ERP when reachable, else the analyst sets it
  cost_of_debt   interest expense / average debt from the 10-K facts (accession cited)
  debt_weight    book debt / (book debt + market cap); market cap = cover-page shares (dei facts) x Yahoo price
  tax_rate       three-year effective rate from the 10-K facts, statutory rate noted
  overrides      analyst-set values that win over the derived ones, each with a reason
  notes          free text the analyst wants the debate to see (becomes evidence)

The file is reused when its as_of date is within ``refresh_days`` of the requested date; --refresh forces a rebuild.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import marketdata as md
from .facts import load_facts

STATUTORY_US = 0.21


def _shares_outstanding(facts: pd.DataFrame | None, as_of: date) -> tuple[float, str] | None:
    """Cover-page shares outstanding (dei:EntityCommonStockSharesOutstanding) from the latest filing on or before as_of."""
    if facts is None:
        return None
    f = facts[(facts.concept == "dei:EntityCommonStockSharesOutstanding") & (facts.filing_date <= pd.Timestamp(as_of))]
    if not len(f):
        return None
    last = f.sort_values("filing_date").iloc[-1]
    return float(last.numeric_value), f"{last.form_type} cover page as of {str(last.period_end)[:10]} (accession {last.accession}, filed {str(last.filing_date)[:10]})"


def _tax_rate(hist: pd.DataFrame) -> dict:
    pretax, tax = hist.loc["pretax_income"].iloc[-3:], hist.loc["income_tax"].iloc[-3:]
    years = list(hist.columns[-3:])
    if pretax.sum() > 0:
        eff = float(np.clip(tax.sum() / pretax.sum(), 0.0, 0.40))
        return {"value": round(eff, 4), "source": f"effective tax rate = income tax / pretax income summed over FY{years[0]}-FY{years[-1]} (10-K facts: IncomeTaxExpenseBenefit, IncomeLossFromContinuingOperationsBeforeIncomeTaxes); US statutory {STATUTORY_US:.0%}",
                "statutory": STATUTORY_US, "effective_by_year": {int(y): (round(float(t / p), 4) if p else None) for y, p, t in zip(years, pretax, tax)}}
    return {"value": STATUTORY_US, "source": f"US statutory rate (cumulative pretax income over FY{years[0]}-FY{years[-1]} not positive, effective rate not meaningful)", "statutory": STATUTORY_US}


def _cost_of_debt(hist: pd.DataFrame) -> dict | None:
    debt = hist.loc["short_term_debt"] + hist.loc["long_term_debt"]
    if len(debt) < 2 or debt.iloc[-2:].mean() <= 0:
        return None
    last = int(hist.columns[-1])
    kd = float(hist.loc["interest_expense", last] / debt.iloc[-2:].mean())
    return {"value": round(kd, 4), "source": f"FY{last} interest expense {hist.loc['interest_expense', last]:,.0f} / average debt {debt.iloc[-2:].mean():,.0f} (USD m, 10-K facts: InterestExpense, DebtCurrent + LongTermDebtNoncurrent)",
            "book_debt": round(float(debt.iloc[-1]), 1)}


def _dilution(facts: pd.DataFrame | None, as_of: date, price: float | None, basic_shares: float | None) -> dict | None:
    """Treasury-stock-method dilution from the equity-award facts on or before as_of: options (count, weighted exercise
    price) and unvested RSUs. Convertible notes are flagged, not converted (conversion prices are not in the facts)."""
    if facts is None or not price or not basic_shares:
        return None
    floor = pd.Timestamp(as_of) - pd.Timedelta(days=456)  # award facts older than ~15 months describe a different company

    def latest(concept):
        f = facts[(facts.concept.str.lower() == f"us-gaap:{concept}".lower()) & (facts.filing_date <= pd.Timestamp(as_of)) & (facts.period_end >= floor)]
        if not len(f):
            return None
        r = f.sort_values(["period_end", "filing_date"]).iloc[-1]
        return float(r.numeric_value), f"{r.form_type} {str(r.period_end)[:10]}"
    opts = latest("ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber")
    strike = latest("ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice")
    rsus = latest("ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber")
    conv = latest("ConvertibleNotesPayable") or latest("ConvertibleDebtNoncurrent") or latest("ConvertibleNotesPayableNoncurrent") or latest("ConvertibleDebt")
    if not (opts or rsus):
        return {"value": basic_shares, "basic": basic_shares, "from_options": 0.0, "from_rsus": 0.0, "price": price,
                "source": f"cover-page count; no option or RSU facts within 15 months of {as_of} so no treasury-stock dilution is modeled"
                          + (f"; convertible notes {conv[0] / 1e6:,.0f}m carrying value ({conv[1]}): conversion not modeled" if conv else "")}
    add_opt = 0.0
    if opts and strike and price > strike[0]:
        add_opt = opts[0] * (1 - strike[0] / price)
    add_rsu = rsus[0] if rsus else 0.0
    diluted = basic_shares + add_opt + add_rsu
    parts = []
    if opts:
        parts.append(f"options {opts[0] / 1e6:,.1f}m at weighted strike {strike[0]:.2f} ({opts[1]})" if strike else f"options {opts[0] / 1e6:,.1f}m (no strike fact)")
    if rsus:
        parts.append(f"unvested RSUs {rsus[0] / 1e6:,.1f}m ({rsus[1]})")
    if conv:
        parts.append(f"convertible notes {conv[0] / 1e6:,.0f}m carrying value ({conv[1]}); conversion not modeled: price not in facts")
    return {"value": round(diluted, 0), "basic": basic_shares, "from_options": round(add_opt, 0), "from_rsus": round(add_rsu, 0), "price": price,
            "source": f"treasury stock method at {price:.2f}: " + ("; ".join(parts) if parts else "no award facts found") + f"; basic {basic_shares / 1e6:,.1f}m"}


def derive_inputs(ticker: str, data_dir: Path, hist: pd.DataFrame, peers: list[str] | None = None, as_of: str | None = None,
                  benchmark: str = "^GSPC", fetch_fred=md.fetch_fred, fetch_prices=md.fetch_yahoo_weekly, fetch_erp=md.fetch_damodaran_histimpl,
                  existing: dict | None = None, erp_method: str = "hist30y") -> dict:
    """Build the inputs dict. Network fetchers are injectable; failures leave a field marked 'unavailable' rather than crashing."""
    as_of_d = date.fromisoformat(as_of) if as_of else date.today()
    out: dict = {"ticker": ticker.upper(), "as_of": as_of_d.isoformat(), "benchmark": benchmark, "fields": {}, "overrides": (existing or {}).get("overrides", {}),
                 "notes": (existing or {}).get("notes", [])}
    F = out["fields"]
    warnings: list[str] = []
    # risk-free
    try:
        F["risk_free"] = md.risk_free_rate(as_of_d, data_dir=data_dir, fetch=fetch_fred)
    except Exception as e:
        warnings.append(f"risk_free: {e}")
    # beta (target and peers)
    try:
        F["beta"] = md.weekly_beta(ticker, benchmark, as_of_d, data_dir=data_dir, fetch=fetch_prices)
    except Exception as e:
        warnings.append(f"beta: {e}")
    F["peer_betas"] = {}
    for p in peers or []:
        try:
            F["peer_betas"][p.upper()] = md.weekly_beta(p, benchmark, as_of_d, data_dir=data_dir, fetch=fetch_prices)
        except Exception as e:
            warnings.append(f"peer beta {p}: {e}")
    # ERP: expected market return by method minus the risk-free rate on the same as-of date; the monthly series is kept
    # in data/market/erp_series.csv for backtests and for the record; analyst-set values still win
    prev_erp = ((existing or {}).get("fields") or {}).get("erp")
    if prev_erp and prev_erp.get("analyst_set"):
        F["erp"] = prev_erp
    else:
        rf_val = (F.get("risk_free") or {}).get("value")
        try:
            if rf_val is None:
                raise RuntimeError("no risk-free rate to subtract")
            alts = {}
            for m in md.ERP_METHODS:
                try:
                    rm_m = md.expected_market_return(as_of_d, m, data_dir, fetch_prices)
                    alts[m] = round(rm_m["value"] - rf_val, 5)
                except Exception:
                    pass
            rm = md.expected_market_return(as_of_d, erp_method, data_dir, fetch_prices)
            F["erp"] = {"value": round(rm["value"] - rf_val, 5), "method": erp_method, "rm": rm["value"], "rm_date": rm["date"], "rf": rf_val,
                        "rf_date": F["risk_free"].get("date"), "alternatives": alts,
                        "source": f"{rm['source']} = {rm['value']:.2%}, minus DGS10 {rf_val:.2%} on {F['risk_free'].get('date', as_of_d)}; "
                                  f"other methods: " + ", ".join(f"{k} {v:.1%}" for k, v in alts.items() if k != erp_method) + "; series in data/market/erp_series.csv"}
            try:
                md.erp_series(as_of_d, data_dir=data_dir, fetch_rf=fetch_fred, fetch_px=fetch_prices)
            except Exception:
                pass
        except Exception as e:
            warnings.append(f"erp: {e}; falling back to Damodaran")
            try:
                F["erp"] = md.damodaran_implied_erp(fetch=fetch_erp)
            except Exception as e2:
                warnings.append(f"erp: {e2}; set it with `crucible inputs {ticker} --erp <decimal> --erp-source '<source, date>'`")
                if prev_erp:
                    F["erp"] = prev_erp
        try:
            F["erp_damodaran_crosscheck"] = md.damodaran_implied_erp(fetch=fetch_erp)
        except Exception:
            pass
        erp_v = (F.get("erp") or {}).get("value")
        if erp_v is not None and not (0.02 <= erp_v <= 0.08):
            xc = (F.get("erp_damodaran_crosscheck") or {}).get("value")
            warnings.append(f"erp {erp_v:.1%} is outside the 2-8% range practitioners use" + (f" (Damodaran implied {xc:.1%})" if xc else "") +
                            "; pick another --erp-method or set --erp with a source")
    # filings-derived
    F["tax_rate"] = _tax_rate(hist)
    kd = _cost_of_debt(hist)
    rf_v = (F.get("risk_free") or {}).get("value")
    if kd and (kd.get("value") in (None, 0.0)) and rf_v is not None:
        # no interest expense fact on the face of the statements (Apple stopped presenting it): proxy at the risk-free rate plus 100bp
        kd = {"value": round(rf_v + 0.01, 4), "source": f"proxy: no interest expense fact in the 10-K facts; DGS10 {rf_v:.2%} + 100bp; override with the disclosed weighted-average rate from the debt note"}
    if kd:
        F["cost_of_debt"] = kd
        if rf_v is not None and kd.get("value") is not None and kd["value"] < rf_v:
            warnings.append(f"cost_of_debt {kd['value']:.1%} is below the risk-free rate {rf_v:.1%}: low-coupon convertible or zero-coupon notes; "
                            f"a market yield on straight debt is the economic cost (override with --cost-of-debt if the flag exists, or an analyst override in analyst_inputs.json)")
    # debt weight: market value of equity when a price and share count exist, else book
    facts = load_facts(data_dir, ticker)
    book_debt = float((hist.loc["short_term_debt"] + hist.loc["long_term_debt"]).iloc[-1])
    book_equity = float(hist.loc["total_equity"].iloc[-1])
    dw: dict = {"book_weight": round(book_debt / (book_debt + book_equity), 4) if (book_debt + book_equity) > 0 else None,
                "book_debt_usd_m": round(book_debt, 1), "book_equity_usd_m": round(book_equity, 1)}
    try:
        sh = _shares_outstanding(facts, as_of_d)
        px = md.price_on_or_before(ticker, as_of_d, data_dir=data_dir, fetch=fetch_prices)
        if sh:
            mcap = sh[0] * px["value"] / 1e6
            dw.update({"value": round(book_debt / (book_debt + mcap), 4), "market_cap_usd_m": round(mcap, 0), "shares": sh[0], "price": px["value"],
                       "source": f"book debt {book_debt:,.0f} (10-K FY{int(hist.columns[-1])} facts) / (book debt + market cap {mcap:,.0f} = {sh[0] / 1e6:,.1f}m shares [{sh[1]}] x {px['source']} {px['value']:.2f})"})
            dil = _dilution(facts, as_of_d, px["value"], sh[0])
            if dil:
                F["diluted_shares"] = dil
        else:
            raise RuntimeError("no cover-page share count in facts")
    except Exception as e:
        warnings.append(f"market debt weight: {e}; using book weight")
        dw.update({"value": dw["book_weight"], "source": f"book debt / (book debt + book equity) from the 10-K FY{int(hist.columns[-1])} balance sheet (market cap unavailable)"})
    F["debt_weight"] = dw
    out["warnings"] = warnings
    return out


def flatten(inputs: dict) -> dict:
    """The flat dict the evidence builders and the model consume; overrides win and carry their reason into the source."""
    F = inputs.get("fields", {})
    ov = inputs.get("overrides", {})
    flat: dict = {"notes": list(inputs.get("notes", [])), "as_of": inputs.get("as_of")}
    for key in ("risk_free", "erp", "beta", "cost_of_debt", "debt_weight", "tax_rate"):
        if key in ov and isinstance(ov[key], dict) and "value" in ov[key]:
            flat[key] = float(ov[key]["value"])
            flat[f"{key}_source"] = f"analyst override: {ov[key].get('reason', 'no reason given')}"
        elif key in F and F[key].get("value") is not None:
            flat[key] = float(F[key]["value"])
            flat[f"{key}_source"] = F[key].get("source", key)
    if "beta" in F:
        flat["beta_adjusted"] = F["beta"].get("adjusted")
        flat["beta_r2"] = F["beta"].get("r2")
    flat["peer_betas"] = {p: v["value"] for p, v in F.get("peer_betas", {}).items() if v.get("value") is not None}
    flat["peer_beta_sources"] = {p: v.get("source", "") for p, v in F.get("peer_betas", {}).items()}
    if "debt_weight" in F:
        flat["debt_weight_book"] = F["debt_weight"].get("book_weight")
    if "tax_rate" in F:
        flat["tax_rate_statutory"] = F["tax_rate"].get("statutory", STATUTORY_US)
    xc = F.get("erp_damodaran_crosscheck")
    if xc and xc.get("value") is not None:
        flat["erp_damodaran_crosscheck"] = float(xc["value"])
        flat["erp_damodaran_crosscheck_source"] = xc.get("source", "Damodaran implied ERP")
    if "erp" in F:
        flat["erp_method"] = F["erp"].get("method")
    ds = F.get("diluted_shares")
    if ds and ds.get("value"):
        flat["diluted_shares"] = float(ds["value"])
        flat["diluted_shares_source"] = ds.get("source", "")
    return flat


def inputs_path(data_dir: Path, ticker: str) -> Path:
    return Path(data_dir) / ticker.upper() / "analyst_inputs.json"


def load_inputs(data_dir: Path, ticker: str) -> dict | None:
    p = inputs_path(data_dir, ticker)
    return json.loads(p.read_text()) if p.exists() else None


def is_fresh(inputs: dict | None, as_of: str | None, refresh_days: int = 7) -> bool:
    if not inputs or not inputs.get("as_of"):
        return False
    want = date.fromisoformat(as_of) if as_of else date.today()
    have = date.fromisoformat(inputs["as_of"])
    return abs((want - have).days) <= refresh_days


def save_inputs(data_dir: Path, ticker: str, inputs: dict) -> Path:
    p = inputs_path(data_dir, ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(inputs, indent=2, default=str))
    return p


def summary(inputs: dict) -> str:
    F, ov = inputs.get("fields", {}), inputs.get("overrides", {})
    lines = [f"analyst inputs for {inputs.get('ticker')} as of {inputs.get('as_of')}"]
    for key in ("risk_free", "erp", "beta", "cost_of_debt", "debt_weight", "tax_rate"):
        f = F.get(key)
        if key in ov:
            lines.append(f"  {key:13s} {float(ov[key]['value']):.4f}  OVERRIDE: {ov[key].get('reason', '')}")
        elif f and f.get("value") is not None:
            lines.append(f"  {key:13s} {float(f['value']):.4f}  <- {f.get('source', '')}")
        else:
            lines.append(f"  {key:13s} unavailable")
    for p, v in F.get("peer_betas", {}).items():
        lines.append(f"  beta {p:8s} {v['value']:.3f} (adj {v['adjusted']:.2f}, r2 {v['r2']:.2f})")
    for w in inputs.get("warnings", []):
        lines.append(f"  warning: {w}")
    if inputs.get("notes"):
        lines.append(f"  notes: {len(inputs['notes'])}")
    return "\n".join(lines)

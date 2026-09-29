"""Map filing data onto the standard schema.

Sources, in priority order:
  1. concept-level facts (``facts.py``): exact us-gaap concepts, restated or as-of a date;
  2. edgartools' standardized statement frames: exact labels, then regex, main section
     rows before "Additional"/"Calculated" rows; several candidate rows for one key are
     coalesced (the row with the most values leads, the others fill gaps).

Residual lines are then computed so every historical total ties to what the
company reported; the size of each residual is reported as a mapping-quality metric.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CONFIDENCE, DERIVED_ITEMS, ITEM_BY_KEY, MappingReport, items_for, match_candidates, match_row, norm_label

_PERIOD_RE = re.compile(r"(?:^|\D)((?:19|20)\d{2})(?:\D|$)")
_META_COLS = {"label", "concept", "level", "abstract", "dimension", "units", "unit", "section", "statement", "depth", "is_abstract", "is_total", "confidence"}


def _period_year(col: str) -> int | None:
    m = _PERIOD_RE.search(str(col))
    return int(m.group(1)) if m else None


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    if s in ("", "-", "—", "nan", "None"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def period_columns(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in df.columns:
        if str(c).lower() in _META_COLS:
            continue
        y = _period_year(c)
        if y is not None:
            out[str(c)] = y
    return out


def _section_rank(row) -> int:
    sec = str(row.get("section", "")).lower()
    if sec in ("", "nan", "none"):
        return 0
    if "calculated" in sec:
        return 1
    return 2  # Additional


def map_statement(df: pd.DataFrame, statement: str, overrides: dict[str, str] | None = None, ledger=None, ticker: str | None = None,
                  ) -> tuple[pd.DataFrame, MappingReport]:
    """Map one standardized statement frame to standard keys (label-based).

    Precedence: analyst-accepted ledger rows > mapping_overrides.json > rules. Rejected ledger rows are never mapped.
    Every rule match is recorded in the ledger (auto-accepted) with its confidence and the alternatives that also fired."""
    overrides = {norm_label(k): v for k, v in (overrides or {}).items()}
    rejected: set[str] = set()
    if ledger is not None and ticker:
        for src, tgt in ledger.decided_links(ticker, statement, "label").items():
            overrides[norm_label(src)] = tgt
        rejected = {norm_label(x) for x in ledger.decided_rejects(ticker, statement, "label")}
    report = MappingReport(statement=statement)
    cols = period_columns(df)
    if not cols:
        raise ValueError(f"No period columns found in {statement} dataframe: {list(df.columns)[:10]}")
    label_col = next((c for c in df.columns if str(c).lower() == "label"), None)
    concept_col = next((c for c in df.columns if str(c).lower() == "concept"), None)
    years = sorted(set(cols.values()))
    keys = [it.key for it in items_for(statement)]
    out = pd.DataFrame(np.nan, index=keys, columns=years, dtype=float)
    candidates: dict[str, list[tuple[int, str, dict[int, float]]]] = {k: [] for k in keys}
    for _, row in df.iterrows():
        if "is_abstract" in df.columns and bool(row["is_abstract"]):
            continue
        label = str(row[label_col]) if label_col is not None else ""
        concept = str(row[concept_col]) if concept_col is not None and pd.notna(row[concept_col]) else ""
        vals = {}
        for c, y in cols.items():
            x = _to_float(row[c])
            if x is not None:
                vals[y] = x
        if not vals:
            continue  # abstract/header rows carry no numbers
        cands = match_candidates(label, concept, statement)
        ov = overrides.get(norm_label(label))
        if norm_label(label) in rejected or ov in ("residual", "ignore"):
            key = None
        elif ov in ITEM_BY_KEY and ITEM_BY_KEY[ov].statement == statement:
            key = ov
        else:
            key = cands[0][0] if cands else None
        if ledger is not None and ticker and (cands or ov):
            latest_col = max(cols, key=cols.get)
            ledger.upsert(ticker, statement, "label", label or concept, key or ov or "", (CONFIDENCE["ledger"] if ov else cands[0][1]),
                          [{"target": k, "confidence": c} for k, c, _ in cands[1:]], "accepted" if (key or ov) else "rejected",
                          ("analyst" if ov else cands[0][2]) if (cands or ov) else "rule:label",
                          evidence=f"{latest_col}: {vals.get(cols[latest_col], '')}")
        if key is None:
            if label not in report.unmapped and norm_label(label) not in rejected and ov not in ("residual", "ignore"):
                report.unmapped.append(label or concept)
            continue
        item = ITEM_BY_KEY[key]
        vals = {y: (abs(x) if item.sign == "abs" else x) for y, x in vals.items()}
        # analyst-linked labels are components of the key: they never displace a rule match (see facts.resolve_links)
        candidates[key].append((1000 + _section_rank(row) if ov and not any(k == key for k, _, _ in cands[:1]) else _section_rank(row), label or concept, vals))
    for key, cands in candidates.items():
        if not cands:
            continue
        cands.sort(key=lambda c: (c[0], -len(c[2])))  # main section first, then most complete; linked components last
        rule_cands = [c for c in cands if c[0] < 1000]
        link_cands = [c for c in cands if c[0] >= 1000]
        if rule_cands:
            lead = rule_cands[0]
            for y, x in lead[2].items():
                out.at[key, y] = x
            report.mapped[key] = f"statement: {lead[1]}"
            for rank, label, vals in rule_cands[1:]:
                filled = False
                for y, x in vals.items():
                    if np.isnan(out.at[key, y]):
                        out.at[key, y] = x
                        filled = True
                if not filled and any(abs(vals.get(y, np.nan) - out.at[key, y]) > 1e-6 for y in vals if not np.isnan(out.at[key, y])):
                    report.duplicates.append((key, label))
        if link_cands:
            by_year: dict[int, float] = {}
            seen_vals: set[tuple] = set()
            for _, label, vals in link_cands:
                sig = tuple(sorted((y, round(x, 2)) for y, x in vals.items()))
                if sig in seen_vals:
                    continue  # a label twin of a component already counted
                seen_vals.add(sig)
                for y, x in vals.items():
                    if x:
                        by_year[y] = by_year.get(y, 0.0) + x
            names = "+".join(l for _, l, _ in link_cands)
            diffs = []
            for y, x in sorted(by_year.items()):
                cur = out.at[key, y]
                if np.isnan(cur):
                    out.at[key, y] = abs(x) if ITEM_BY_KEY[key].sign == "abs" else x
                elif cur and abs(x - cur) / abs(cur) > 0.03 and y >= max(years) - 5:
                    diffs.append(f"FY{y} {x / 1e6:,.1f}m vs {cur / 1e6:,.1f}m")
            if diffs:
                report.notes.append(f"{key}: linked labels ({names}) differ from the mapped {report.mapped.get(key, 'total')}; total kept: " + "; ".join(diffs))
            if not rule_cands:
                report.mapped[key] = f"statement (linked components): {names}"
    return out, report


def _has_core(is_df: pd.DataFrame, bs_df: pd.DataFrame, y) -> bool:
    try:
        rev, ta = is_df.at["revenue", y], bs_df.at["total_assets", y]
    except KeyError:
        return False
    return bool(pd.notna(rev) and rev > 0 and pd.notna(ta) and ta > 0)


def _g(df: pd.DataFrame, key: str) -> pd.Series:
    if key in df.index:
        return df.loc[key].astype(float).fillna(0.0)
    return pd.Series(0.0, index=df.columns)


def _has(df: pd.DataFrame, key: str) -> bool:
    return key in df.index and df.loc[key].notna().any() and df.loc[key].abs().sum() > 0


def add_residuals(is_df: pd.DataFrame, bs_df: pd.DataFrame, cf_df: pd.DataFrame, reports: dict[str, MappingReport],
                  keep_years: int | None = None) -> pd.DataFrame:
    """Combine statements, apply fallbacks for missing lines, compute residuals so totals tie."""
    years = sorted(set(is_df.columns) & set(bs_df.columns) & set(cf_df.columns))
    if not years:
        raise ValueError("Statements share no fiscal years")
    # a fiscal year without revenue or total assets is a shell (pre-merger SPAC years, deprecated concepts); drop it
    years = [y for y in years if _has_core(is_df, bs_df, y)] or years
    # keep the longest run of consecutive fiscal years ending at the latest one (gaps break growth and roll-forward logic)
    run = [years[-1]]
    for y in reversed(years[:-1]):
        if y == run[0] - 1:
            run.insert(0, y)
        else:
            break
    years = run
    if keep_years:
        years = years[-keep_years:]
    is_df, bs_df, cf_df = (d.reindex(columns=years) for d in (is_df, bs_df, cf_df))
    raw = pd.concat([is_df, bs_df, cf_df])
    for st, d in (("IS", is_df), ("BS", bs_df), ("CF", cf_df)):
        reports[st].missing = [k for k in d.index if ITEM_BY_KEY[k].in_model and not d.loc[k].notna().any()]
    hist = raw.fillna(0.0)

    # ---- Income statement fallbacks (per cell: a line can exist in some years and not others)
    def _fill(key: str, values: pd.Series, note: str) -> None:
        mask = raw.loc[key].isna() if key in raw.index else pd.Series(True, index=years)
        if mask.any():
            hist.loc[key, mask[mask].index] = values[mask[mask].index]
            if key in ("operating_income", "pretax_income"):
                reports["IS"].notes.append(f"{note} for FY{', FY'.join(str(y) for y in mask[mask].index)}")

    _fill("gross_profit", _g(hist, "revenue") - _g(hist, "cogs"), "gross profit = revenue - cost of revenue")
    if not _has(raw, "cogs") and _has(raw, "gross_profit"):
        _fill("cogs", _g(hist, "revenue") - _g(hist, "gross_profit"), "cogs = revenue - gross profit")
    # a reported gross profit must equal revenue - cogs; where both are on the statement and disagree, the identity wins
    # (edgartools' "Gross Profit (Calculated)" can subtract total costs) and the disagreement is noted
    if "cogs" in raw.index and "gross_profit" in raw.index:
        both = raw.loc["cogs"].notna() & raw.loc["gross_profit"].notna() & raw.loc["revenue"].notna()
        for y in [y for y in years if both.get(y, False)]:
            ident = float(hist.at["revenue", y]) - float(hist.at["cogs", y])
            if abs(ident - float(hist.at["gross_profit", y])) > 0.005 * max(abs(float(hist.at["revenue", y])), 1.0):
                reports["IS"].notes.append(f"FY{y}: reported gross profit {float(hist.at['gross_profit', y]):,.1f} differs from revenue - cogs {ident:,.1f}; identity kept")
                hist.at["gross_profit", y] = ident
    _fill("pretax_income", _g(hist, "net_income") + _g(hist, "income_tax"), "pretax reconstructed as net income + tax")
    if "nonoperating_total" in raw.index and raw.loc["nonoperating_total"].notna().any():
        ebit_fallback = _g(hist, "pretax_income") - _g(hist, "nonoperating_total")
        _fill("operating_income", ebit_fallback, "operating income reconstructed as pretax - total non-operating (no OperatingIncomeLoss)")
    _fill("operating_income", _g(hist, "pretax_income") + _g(hist, "interest_expense") - _g(hist, "other_nonoperating"),
          "operating income reconstructed as pretax + interest - other non-operating (no OperatingIncomeLoss)")
    hist.loc["other_opex"] = _g(hist, "gross_profit") - _g(hist, "sga") - _g(hist, "rnd") - _g(hist, "operating_income")
    # R&D disclosed in the notes but embedded in COGS/SG&A on the face of the statement shows up as a
    # negative residual. Reconciliation, not the label, decides: treat R&D as a memo in that case.
    rev = _g(hist, "revenue").replace(0, np.nan)
    embedded = ((hist.loc["other_opex"] < -0.01 * rev) & (_g(hist, "rnd") > 0))
    if embedded.sum() >= max(1, int(0.5 * (_g(hist, "rnd") > 0).sum())):
        reports["IS"].notes.append("R&D is disclosed in the notes but embedded in COGS/SG&A on the face of the statement; treated as memo (0) so EBIT reconciles")
        hist.loc["rnd"] = 0.0
        hist.loc["other_opex"] = _g(hist, "gross_profit") - _g(hist, "sga") - _g(hist, "operating_income")
    hist.loc["other_income"] = _g(hist, "pretax_income") - _g(hist, "operating_income") + _g(hist, "interest_expense")
    hist.loc["ebitda"] = _g(hist, "operating_income") + _g(hist, "d_and_a")
    hist.loc["nci_and_other"] = _g(hist, "pretax_income") - _g(hist, "income_tax") - _g(hist, "net_income")

    # ---- Balance sheet residuals (equity absorbs NCI/mezzanine so the sheet balances)
    hist.loc["other_current_assets"] = _g(hist, "total_current_assets") - _g(hist, "cash") - _g(hist, "receivables") - _g(hist, "inventory")
    hist.loc["other_noncurrent_assets"] = (_g(hist, "total_assets") - _g(hist, "total_current_assets") - _g(hist, "ppe_net")
                                           - _g(hist, "goodwill") - _g(hist, "intangibles"))
    if not _has(raw, "total_liabilities_and_equity"):
        hist.loc["total_liabilities_and_equity"] = _g(hist, "total_assets")
    if not _has(raw, "total_liabilities"):
        hist.loc["total_liabilities"] = _g(hist, "total_liabilities_and_equity") - _g(hist, "total_equity")
        reports["BS"].notes.append("total liabilities reconstructed as L&E - equity")
    hist.loc["total_equity"] = _g(hist, "total_liabilities_and_equity") - _g(hist, "total_liabilities")
    hist.loc["other_current_liabilities"] = _g(hist, "total_current_liabilities") - _g(hist, "payables") - _g(hist, "short_term_debt")
    hist.loc["other_noncurrent_liabilities"] = _g(hist, "total_liabilities") - _g(hist, "total_current_liabilities") - _g(hist, "long_term_debt")

    # ---- Cash flow residuals (cash-effect sign convention)
    if not _has(raw, "cf_net_income"):
        hist.loc["cf_net_income"] = _g(hist, "net_income")
    hist.loc["d_nwc"] = _g(hist, "cfo") - _g(hist, "cf_net_income") - _g(hist, "d_and_a") - _g(hist, "sbc")
    hist.loc["other_investing"] = _g(hist, "cfi") + _g(hist, "capex")
    hist.loc["net_debt_issuance"] = _g(hist, "cff") + _g(hist, "dividends") + _g(hist, "buybacks")
    if not _has(raw, "net_change_cash"):
        hist.loc["net_change_cash"] = _g(hist, "cfo") + _g(hist, "cfi") + _g(hist, "cff") + _g(hist, "fx_effect")

    def _pct(res: str, base: str, rep: str) -> None:
        b = _g(hist, base).abs().replace(0, np.nan)
        val = (_g(hist, res).abs() / b).max()
        if not np.isnan(val):
            reports[rep].residuals[res] = float(val)

    _pct("other_opex", "revenue", "IS")
    _pct("other_income", "revenue", "IS")
    _pct("nci_and_other", "revenue", "IS")
    _pct("other_current_assets", "total_assets", "BS")
    _pct("other_noncurrent_assets", "total_assets", "BS")
    _pct("other_current_liabilities", "total_assets", "BS")
    _pct("other_noncurrent_liabilities", "total_assets", "BS")
    _pct("d_nwc", "revenue", "CF")
    _pct("other_investing", "revenue", "CF")
    _pct("net_debt_issuance", "revenue", "CF")

    order = [it.key for st in ("IS", "BS", "CF") for it in items_for(st)]
    derived = [k for k, _, _ in DERIVED_ITEMS]
    hist = hist.reindex(order + derived).fillna(0.0)
    hist.columns = [int(c) for c in hist.columns]
    return hist


def build_history(is_df: pd.DataFrame, bs_df: pd.DataFrame, cf_df: pd.DataFrame, overrides: dict | None = None,
                  facts: pd.DataFrame | None = None, as_of: str | None = None, keep_years: int | None = None, ledger=None,
                  ticker: str | None = None) -> tuple[pd.DataFrame, dict[str, MappingReport]]:
    """Facts (if given) are the primary source; statement frames fill whatever facts lack. The ledger supplies analyst
    decisions (accepted concept and label links, rejections) and receives every rule match."""
    overrides = overrides or {}
    is_m, r_is = map_statement(is_df, "IS", overrides.get("IS"), ledger, ticker)
    bs_m, r_bs = map_statement(bs_df, "BS", overrides.get("BS"), ledger, ticker)
    cf_m, r_cf = map_statement(cf_df, "CF", overrides.get("CF"), ledger, ticker)
    reports = {"IS": r_is, "BS": r_bs, "CF": r_cf}
    if facts is not None:
        from .facts import history_from_facts

        concept_links, concept_rejects = {}, set()
        if ledger is not None and ticker:
            for st in ("IS", "BS", "CF"):
                for src, tgt in ledger.decided_links(ticker, st, "concept").items():
                    if tgt in ITEM_BY_KEY and ITEM_BY_KEY[tgt].statement == st:
                        concept_links[src.lower()] = tgt
                concept_rejects |= {x.lower() for x in ledger.decided_rejects(ticker, st, "concept")}
        fh, prov = history_from_facts(facts, as_of=as_of, concept_links=concept_links, concept_rejects=concept_rejects)
        if ledger is not None and ticker:
            for key, pv in prov.items():
                it = ITEM_BY_KEY[key]
                for concept in pv["concepts"]:
                    src = concept if ":" in concept else f"us-gaap:{concept}"
                    how = "rule:sum" if "+" in concept else "rule:concept"
                    ledger.upsert(ticker, it.statement, "concept", src, key, CONFIDENCE["sum"] if how == "rule:sum" else CONFIDENCE["concept"], [],
                                  "accepted", how, evidence=f"FY{pv['years'][-1]} accession {pv['accessions'].get(pv['years'][-1], '')}")
        merged = {}
        for st, m in (("IS", is_m), ("BS", bs_m), ("CF", cf_m)):
            years = sorted(set(fh.columns) | set(m.columns)) if as_of is None else sorted(fh.columns)
            f_st = fh.reindex(index=m.index, columns=years)
            s_st = m.reindex(columns=years)
            # facts lead, statements fill; in as-of mode the statement frames are latest-known, so no fill (no leakage)
            out = f_st.where(f_st.notna(), s_st) if as_of is None else f_st
            merged[st] = out
            for k in m.index:
                if k in prov:
                    reports[st].mapped[k] = "facts: " + "/".join(prov[k]["concepts"])
                    for n in prov[k].get("notes", []):
                        reports[st].notes.append(n)
                    if s_st.loc[k].notna().any() and f_st.loc[k].isna().any() and out.loc[k].notna().any():
                        reports[st].mapped[k] += " (+statement fill)"
        is_m, bs_m, cf_m = merged["IS"], merged["BS"], merged["CF"]
    return add_residuals(is_m, bs_m, cf_m, reports, keep_years=keep_years), reports


def scale_history(hist: pd.DataFrame, divisor: float = 1e6) -> pd.DataFrame:
    """Express amounts and share counts in millions; per-share values untouched."""
    out = hist.copy()
    keep = [k for k in out.index if k.startswith("eps")]
    rows = [k for k in out.index if k not in keep]
    out.loc[rows] = out.loc[rows] / divisor
    return out


def load_history(data_dir: Path, ticker: str, as_of: str | None = None, scale: float | None = 1e6,
                 use_facts: bool = True, hist_years: int = 5) -> tuple[pd.DataFrame, dict[str, MappingReport]]:
    """Load data/<TICKER>/ (written by ``ingest``) and map it. Amounts in USD millions; last ``hist_years`` fiscal years."""
    d = Path(data_dir) / ticker.upper()
    frames = {}
    for st, name in (("IS", "annual_is.csv"), ("BS", "annual_bs.csv"), ("CF", "annual_cf.csv")):
        p = d / name
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run: crucible ingest {ticker}")
        frames[st] = pd.read_csv(p)
    ov_path = d / "mapping_overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    facts = None
    if use_facts:
        from .facts import load_facts

        facts = load_facts(data_dir, ticker)
    from .ledger import MappingLedger

    ledger = MappingLedger(Path(data_dir))
    if facts is not None:
        from .facts import load_xbrl_facts

        extra = load_xbrl_facts(data_dir, ticker)  # full-filing facts (custom concepts) when the ingest saved them
        if extra is not None and len(extra):
            facts = pd.concat([facts, extra[~extra.concept.isin(set(facts.concept))]], ignore_index=True)
    hist, reports = build_history(frames["IS"], frames["BS"], frames["CF"], overrides, facts=facts, as_of=as_of, keep_years=hist_years,
                                  ledger=ledger, ticker=ticker)
    if ledger._dirty:
        ledger.save()
    if facts is not None:
        from .facts import unmapped_facts

        try:
            linked = {src.lower() for st in ("IS", "BS", "CF") for src, tgt in ledger.decided_links(ticker, st, "concept").items()}
            linked |= {src.lower() for st in ("IS", "BS", "CF") for src in ledger.decided_rejects(ticker, st, "concept")}
            uf = unmapped_facts(facts, list(hist.columns), as_of=as_of, linked=linked)
            for st_name, rep in reports.items():
                sub = uf[uf.statement_type.str.contains({"IS": "Income", "BS": "Balance", "CF": "CashFlow"}[st_name], na=False)].head(8)
                if len(sub):
                    rep.notes.append("largest unmapped facts (latest FY, absorbed by residuals): " + "; ".join(
                        f"{c.split(':')[-1]} {v / 1e6:,.0f}m" for c, v in zip(sub.concept, sub.numeric_value)))
        except Exception as e:  # diagnostics only
            reports["IS"].notes.append(f"unmapped-facts scan failed: {e}")
    if scale:
        hist = scale_history(hist, scale)
    return hist, reports


def load_fixture(path: Path) -> pd.DataFrame:
    """Load a synthetic fixture (already in standard keys) for offline demos and tests."""
    raw = json.loads(Path(path).read_text())
    df = pd.DataFrame(raw["items"]).T
    df.columns = [int(c) for c in df.columns]
    reports = {"IS": MappingReport("IS"), "BS": MappingReport("BS"), "CF": MappingReport("CF")}
    parts = {st: df.reindex([it.key for it in items_for(st)]).astype(float) for st in ("IS", "BS", "CF")}
    return add_residuals(parts["IS"], parts["BS"], parts["CF"], reports)

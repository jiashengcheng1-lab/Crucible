"""Build the standardized annual history from concept-level XBRL facts.

Why facts and not only the standardized statements: edgartools' statement
standardization occasionally drops a line (Eaton's operating income, CleanSpark's
FY2025 capex) or mixes in "Additional" items. The companyfacts data has every
non-dimensional fact with its us-gaap concept, period and filing date, so exact
concept matching is more reliable, and the filing date lets us rebuild the
history as it was known on any date (``as_of``) for backtests.

Rules:
  * annual duration facts: 340..380 day periods; balance-sheet facts: instants on a fiscal year end
  * fiscal year label comes from the filing whose own FY period ends on that date
  * for each (concept, period end) take the value from the latest filing on or before ``as_of``
  * first concept in the schema list that has a value for a year wins; other concepts fill gaps
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import STANDARD_ITEMS, ITEM_BY_KEY, items_for


def load_facts(data_dir: Path, ticker: str) -> pd.DataFrame | None:
    d = Path(data_dir) / ticker.upper()
    for name in ("facts_pit.parquet", "facts_pit.csv"):
        p = d / name
        if p.exists():
            f = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            for c in ("period_start", "period_end", "filing_date"):
                f[c] = pd.to_datetime(f[c], errors="coerce")
            return f
    return None


def fiscal_year_map(facts: pd.DataFrame) -> dict[pd.Timestamp, int]:
    """period_end -> fiscal year label, learned from each 10-K's own current period."""
    fy = facts[(facts.form_type.isin(["10-K", "10-K/A", "20-F", "40-F"])) & (facts.fiscal_period == "FY") & (facts.period_type == "duration")]
    out: dict[pd.Timestamp, int] = {}
    for acc, g in fy.groupby("accession"):
        end = g.period_end.max()
        yr = int(g.loc[g.period_end == end, "fiscal_year"].mode().iloc[0])
        out[end] = yr
    return out


def _annual(facts: pd.DataFrame, as_of: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    f = facts.copy()
    if as_of is not None:
        f = f[f.filing_date <= pd.Timestamp(as_of)]
    fymap = fiscal_year_map(f)
    if not fymap:
        return f.iloc[0:0], f.iloc[0:0], {}
    dur = f[f.period_type == "duration"].copy()
    dur["days"] = (dur.period_end - dur.period_start).dt.days
    dur = dur[(dur.days >= 340) & (dur.days <= 380) & dur.period_end.isin(fymap)]
    inst = f[(f.period_type == "instant") & f.period_end.isin(fymap)].copy()
    for df in (dur, inst):
        df["fy"] = df.period_end.map(fymap)
    return dur, inst, fymap


def _pick_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (concept, fy): the most recently filed."""
    df = df.dropna(subset=["numeric_value"]).sort_values(["concept", "fy", "filing_date"])
    return df.groupby(["concept", "fy"], as_index=False).last()[["concept", "fy", "numeric_value", "accession", "filing_date", "form_type"]]


_OUTFLOW = ("paymentsto", "paymentsfor", "repayments", "purchaseof", "purchasesof", "acquisition", "payments")


def _component_sign(concept: str, key: str) -> float:
    """Analyst-linked cash-flow components carry no sign convention of their own: payment-style concepts are outflows
    when summed into a cash-flow total; everything else is taken as reported."""
    name = concept.split(":")[-1].lower()
    if key in ("cfo", "cfi", "cff", "net_debt_issuance") and name.startswith(_OUTFLOW):
        return -1.0
    return 1.0


def resolve_links(rule_by_year: dict[int, tuple[float, str, str]], link_rows: pd.DataFrame, key: str, sign: str, tol: float = 0.03,
                  ) -> tuple[dict[int, tuple[float, str, str]], list[tuple[int, str]]]:
    """Merge analyst-linked concepts into a key's yearly values.

    Rule: the schema total wins wherever it has a value. Linked concepts are components: where the total is missing for a
    year, one linked concept fills it and several are summed (payment-style cash-flow concepts negated). A linked value of
    zero never counts as a fill. Where both exist, the components are checked against the total and a note is returned
    when they differ by more than ``tol`` (a restatement, a component outside the total, or a wrong link)."""
    out = dict(rule_by_year)
    notes: list[tuple[int, str]] = []
    if link_rows is None or not len(link_rows):
        return out, notes
    for y, g in link_rows.groupby("fy"):
        y = int(y)
        g = g[g.numeric_value.notna() & (g.numeric_value != 0)]
        if not len(g):
            continue
        g = g.drop_duplicates("numeric_value")  # twin concepts carrying the same fact (D&A tagged twice) count once
        parts = [(str(r.concept), float(r.numeric_value) * _component_sign(str(r.concept), key), str(r.accession)) for _, r in g.iterrows()]
        v = sum(x for _, x, _ in parts)
        label = "+".join(sorted(c.split(":")[-1] for c, _, _ in parts))
        if y in out:
            total = out[y][0]
            comp = abs(v) if sign == "abs" else v
            if total and abs(comp - total) / abs(total) > tol:
                notes.append((y, f"FY{y} {comp / 1e6:,.1f}m vs total {out[y][1]} {total / 1e6:,.1f}m ({(comp / total - 1):+.1%})"))
            continue
        out[y] = (abs(v) if sign == "abs" else v, label if len(parts) > 1 else parts[0][0].split(":")[-1], parts[-1][2])
    return out, notes


def history_from_facts(facts: pd.DataFrame, as_of: str | None = None, min_years: int = 2, concept_links: dict[str, str] | None = None,
                       concept_rejects: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Return (hist DataFrame keyed by schema keys, provenance dict key -> {concepts, years, accessions, notes}).
    ``concept_links`` (lowercased concept -> key) come from the ledger and are treated as components of the key (see
    ``resolve_links``); ``concept_rejects`` are never used, which is how an analyst overrides a wrong schema total."""
    concept_links = concept_links or {}
    concept_rejects = concept_rejects or set()
    dur, inst, fymap = _annual(facts, as_of)
    if not fymap:
        raise ValueError("no annual 10-K periods in facts")
    dur = _pick_latest(dur)
    inst = _pick_latest(inst)
    years = sorted(set(dur.fy) | set(inst.fy))
    hist = pd.DataFrame(np.nan, index=[it.key for it in STANDARD_ITEMS], columns=years, dtype=float)
    prov: dict[str, dict] = {}
    for it in STANDARD_ITEMS:
        src = inst if it.statement == "BS" else dur
        order: dict[str, int] = {}
        for c in it.concepts + it.neg_concepts:
            order.setdefault(f"us-gaap:{c}".lower(), len(order))
        for c in concept_rejects:
            order.pop(c, None)
        neg = {f"us-gaap:{c}".lower() for c in it.neg_concepts}
        rows = src[src.concept.str.lower().isin(order)].copy()
        filled: dict[int, tuple[float, str, str]] = {}
        if len(rows):
            rows["clower"] = rows.concept.str.lower()
            rows["prio"] = rows.clower.map(order)
            # schema concept order wins; a zero from an older filing yields to a non-zero later one
            # (e.g. a SPAC year restated once the operating company reported)
            rows = rows.sort_values(["fy", "prio", "filing_date"], ascending=[True, True, False])
            for y, g in rows.groupby("fy", sort=True):
                g = g.drop_duplicates("prio", keep="first")
                r = g.iloc[0]
                if r.numeric_value == 0 and len(g) > 1:
                    later = g[(g.numeric_value != 0) & (g.filing_date > r.filing_date)]
                    if len(later):
                        r = later.iloc[0]
                v = float(r.numeric_value)
                if r.clower in neg:
                    v = -v
                filled[int(y)] = (abs(v) if it.sign == "abs" else v, r.concept.split(":")[-1], str(r.accession))
        if it.sum_concepts:
            parts = src[src.concept.str.lower().isin({f"us-gaap:{c}".lower() for c in it.sum_concepts} - concept_rejects)]
            for y, g in parts.groupby("fy"):
                y = int(y)
                if y in filled or g.numeric_value.isna().all():
                    continue
                v = float(g.numeric_value.sum())
                filled[y] = (abs(v) if it.sign == "abs" else v, "+".join(sorted(c.split(":")[-1] for c in g.concept)), str(g.accession.iloc[-1]))
        linked = [c for c, k in concept_links.items() if k == it.key and c not in concept_rejects]
        notes: list[str] = []
        if linked:
            link_rows = src[src.concept.str.lower().isin(set(linked))]
            filled, raw_notes = resolve_links(filled, link_rows, it.key, it.sign)
            recent = [n for y, n in raw_notes if y >= max(years) - 5]
            if recent:
                names = sorted({str(c).split(":")[-1] for c in link_rows.concept.unique()}) or sorted(c.split(":")[-1] for c in linked)
                notes = [f"{it.key}: linked components ({', '.join(names)}) differ from the mapped total; total kept: " + "; ".join(recent)]
        for y, (v, _, _) in filled.items():
            hist.at[it.key, y] = v
        if filled:
            prov[it.key] = {"concepts": sorted({c for _, c, _ in filled.values()}), "years": sorted(filled),
                            "accessions": {y: a for y, (_, _, a) in filled.items()}, "notes": notes,
                            "linked": sorted(c.split(":")[-1] for c in linked)}
    # keep only years with a reasonably complete core
    core = ["revenue", "total_assets", "cfo"]
    good = [y for y in years if hist.loc[core, y].notna().all()]
    hist = hist[good]
    if len(good) < min_years:
        raise ValueError(f"only {len(good)} complete fiscal years in facts")
    return hist, prov


def unmapped_facts(facts: pd.DataFrame, hist_years: list[int], as_of: str | None = None, top: int = 25, linked: set[str] | None = None) -> pd.DataFrame:
    """Largest statement facts (latest year) whose concept is not in the schema or linked by the analyst: what the residual lines absorb."""
    dur, inst, fymap = _annual(facts, as_of)
    both = pd.concat([_pick_latest(dur), _pick_latest(inst)])
    known = {f"us-gaap:{c}".lower() for it in STANDARD_ITEMS for c in it.concepts} | {c.lower() for c in (linked or set())}
    latest = max(hist_years)
    st_types = facts.drop_duplicates("concept").set_index("concept")["statement_type"].to_dict()
    b = both[(both.fy == latest) & (~both.concept.str.lower().isin(known))].copy()
    b["statement_type"] = b.concept.map(st_types)
    b = b[b.statement_type.notna()]
    b["abs_value"] = b.numeric_value.abs()
    return b.sort_values("abs_value", ascending=False).head(top)[["concept", "statement_type", "numeric_value"]]


def load_xbrl_facts(data_dir: Path, ticker: str) -> pd.DataFrame | None:
    """Full-filing XBRL facts saved by ``ingest`` (xbrl_facts_<FY>.parquet), normalized to the companyfacts columns so
    custom extension concepts (e.g. mara:CostOfRevenueEnergy) can be linked through the ledger like any other."""
    d = Path(data_dir) / ticker.upper()
    frames = []
    for p in sorted(d.glob("xbrl_facts_*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return None
    x = pd.concat(frames, ignore_index=True)
    need = {"concept", "numeric_value", "period_type", "period_start", "period_end", "filing_date", "accession", "form_type", "fiscal_period", "fiscal_year"}
    if not need <= set(x.columns):
        return None
    for c in ("period_start", "period_end", "filing_date"):
        x[c] = pd.to_datetime(x[c], errors="coerce")
    return x.dropna(subset=["numeric_value"])


QUARTER_KEYS = ("revenue", "cogs", "gross_profit", "sga", "operating_income", "net_income")


def _period_values(rows: pd.DataFrame, order: dict[str, int], neg: set[str], linked: list[str], key: str, sign: str,
                   require: set[str] | None = None) -> dict[pd.Timestamp, float]:
    """One value per period end from rule concepts (priority order, latest filing), else the sum of linked components
    (see resolve_links); a linked zero never counts. With ``require``, a linked sum is only accepted when every one of
    those concepts is present for the period, so a partial component set (a 10-Q that lacks the custom lines) never
    masquerades as the total."""
    out: dict[pd.Timestamp, float] = {}
    r = rows[rows.concept.str.lower().isin(order)].copy()
    if len(r):
        r["prio"] = r.concept.str.lower().map(order)
        r = r.sort_values(["period_end", "prio", "filing_date"], ascending=[True, True, False]).drop_duplicates("period_end")
        for _, x in r.iterrows():
            v = float(x.numeric_value)
            if x.concept.lower() in neg:
                v = -v
            out[pd.Timestamp(x.period_end)] = abs(v) if sign == "abs" else v
    if linked:
        l = rows[rows.concept.str.lower().isin(set(linked))].copy()
        l = l[l.numeric_value.notna() & (l.numeric_value != 0)]
        l = l.sort_values(["period_end", "concept", "filing_date"], ascending=[True, True, False]).drop_duplicates(["period_end", "concept"])
        for pe, g in l.groupby("period_end"):
            pe = pd.Timestamp(pe)
            if pe in out:
                continue
            if require and not require.issubset(set(g.concept.str.lower())):
                continue
            g = g.drop_duplicates("numeric_value")
            v = sum(float(x.numeric_value) * _component_sign(str(x.concept), key) for _, x in g.iterrows())
            out[pe] = abs(v) if sign == "abs" else v
    return out


def quarterly_from_facts(facts: pd.DataFrame, as_of: str | None = None, concept_links: dict[str, str] | None = None,
                         concept_rejects: set[str] | None = None) -> pd.DataFrame:
    """Quarterly income-statement history keyed by schema key: Q1-Q3 from 10-Q duration facts (about 90 days), Q4 = fiscal
    year minus the three quarters. Analyst concept links act as components exactly as in the annual history. Columns are
    'FY2025Q1' style labels in time order; values in USD millions."""
    concept_links = concept_links or {}
    concept_rejects = concept_rejects or set()
    f = facts.copy()
    if as_of:
        f = f[f.filing_date <= pd.Timestamp(as_of)]
    f = f[f.period_type == "duration"].copy()
    f["days"] = (f.period_end - f.period_start).dt.days
    q = f[(f.days >= 80) & (f.days <= 100)]
    fy = f[(f.days >= 340) & (f.days <= 380) & (f.form_type == "10-K")]
    fymap = fiscal_year_map(facts)
    out: dict[str, dict[str, float]] = {}
    for it in STANDARD_ITEMS:
        if it.key not in QUARTER_KEYS:
            continue
        order: dict[str, int] = {}
        for c in it.concepts + it.neg_concepts:
            order.setdefault(f"us-gaap:{c}".lower(), len(order))
        for c in concept_rejects:
            order.pop(c, None)
        neg = {f"us-gaap:{c}".lower() for c in it.neg_concepts}
        linked = [c for c, k in concept_links.items() if k == it.key and c not in concept_rejects]
        yv = _period_values(fy, order, neg, linked, it.key, it.sign)
        require: set[str] = set()
        if linked and len(yv):
            last_end = max(yv)
            ly = fy[(fy.period_end == last_end) & fy.concept.str.lower().isin(set(linked)) & fy.numeric_value.notna() & (fy.numeric_value != 0)]
            require = set(ly.concept.str.lower())
        qv = _period_values(q, order, neg, linked, it.key, it.sign, require=require or None)
        series: dict[str, float] = {}
        for end, total in sorted(yv.items()):
            year = fymap.get(pd.Timestamp(end).normalize(), pd.Timestamp(end).year)
            qs = sorted((pe, v) for pe, v in qv.items() if end - pd.Timedelta(days=370) < pe < end)
            vals = [v / 1e6 for _, v in qs[-3:]]
            if len(vals) == 3:
                for i, v in enumerate(vals, start=1):
                    series[f"FY{year}Q{i}"] = round(v, 2)
                series[f"FY{year}Q4"] = round(total / 1e6 - sum(vals), 2)
        out[it.key] = series
    df = pd.DataFrame(out).T
    cols = sorted(df.columns, key=lambda c: (int(c[2:6]), int(c[-1])))
    df = df[cols] if len(cols) else df
    if "gross_profit" in df.index and "revenue" in df.index and "cogs" in df.index:
        gp = df.loc["gross_profit"]
        df.loc["gross_profit"] = gp.where(gp.notna(), df.loc["revenue"] - df.loc["cogs"])
    return df

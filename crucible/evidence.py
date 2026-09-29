"""Evidence packets: the frozen, numbered set of facts a debate is allowed to use.

Every item has an id, a kind, and a source. Advocates may only cite ids that
exist in the packet, so provenance is preserved end to end and the ablation
harness can remove items one at a time.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Kind = Literal["historical_metric", "peer_metric", "filing_text", "analyst_input", "derived_metric", "macro"]


class EvidenceItem(BaseModel):
    id: str
    kind: Kind
    source: str
    content: str
    value: float | None = None
    unit: str | None = None
    period: str | None = None
    direct: bool = False  # directly comparable to the assumption's unit (e.g. a growth rate for a growth assumption)

    def render(self) -> str:
        tag = f"({self.kind}, direct)" if self.direct else f"({self.kind})"
        per = f" [{self.period}]" if self.period else ""
        return f"[{self.id}] {tag} {self.content}{per}  <source: {self.source}>"


class AssumptionSpec(BaseModel):
    key: str
    description: str
    unit: str = "pct"
    bull_direction: Literal["up", "down"] = "up"  # which direction is value-supportive
    lower_bound: float = -0.5
    upper_bound: float = 1.0
    horizon: str = "next 3 fiscal years"


ASSUMPTIONS: dict[str, AssumptionSpec] = {
    "revenue_growth_3y": AssumptionSpec(
        key="revenue_growth_3y",
        description="Average annual revenue growth rate over the next three fiscal years (decimal, e.g. 0.12 = 12%)",
        unit="pct", bull_direction="up", lower_bound=-0.5, upper_bound=1.0),
    "wacc": AssumptionSpec(
        key="wacc",
        description="Weighted average cost of capital used to discount unlevered free cash flows (decimal, e.g. 0.09 = 9%)",
        unit="pct", bull_direction="down", lower_bound=0.04, upper_bound=0.20, horizon="valuation date"),
}


class EvidencePacket(BaseModel):
    company: str
    assumption: AssumptionSpec
    items: list[EvidenceItem]
    as_of: str | None = None
    notes: list[str] = Field(default_factory=list)

    def ids(self) -> list[str]:
        return [i.id for i in self.items]

    def get(self, eid: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.id == eid), None)

    def render(self, order: list[str] | None = None) -> str:
        items = self.items if order is None else [self.get(i) for i in order if self.get(i)]
        return "\n".join(i.render() for i in items)

    def without(self, eid: str) -> "EvidencePacket":
        return self.model_copy(update={"items": [i for i in self.items if i.id != eid]})

    def hash(self) -> str:
        payload = json.dumps({"assumption": self.assumption.key, "items": sorted((i.id, i.content) for i in self.items)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "EvidencePacket":
        return cls.model_validate_json(Path(path).read_text())


# ----------------------------------------------------------------------------- builders

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _cagr(series: pd.Series, n: int) -> float | None:
    if len(series) <= n or series.iloc[-1 - n] <= 0 or series.iloc[-1] <= 0:
        return None
    return float((series.iloc[-1] / series.iloc[-1 - n]) ** (1 / n) - 1)


def growth_evidence(hist: pd.DataFrame, company: str, source_prefix: str, start_id: int = 1, peer: bool = False) -> list[EvidenceItem]:
    """Revenue growth facts from a mapped history. ``peer=True`` tags them as peer metrics."""
    out: list[EvidenceItem] = []
    kind: Kind = "peer_metric" if peer else "historical_metric"
    rev = hist.loc["revenue"]
    years = list(hist.columns)
    k = start_id
    for y0, y in zip(years[:-1], years[1:]):
        if y != y0 + 1 or rev[y0] <= 0:
            continue
        g = rev[y] / rev[y0] - 1
        out.append(EvidenceItem(id=f"E{k}", kind=kind, source=f"{source_prefix} FY{y} vs FY{y0}",
                                content=f"{company} revenue growth FY{y}: {_pct(g)} (revenue {rev[y]:,.0f} vs {rev[y0]:,.0f})",
                                value=float(g), unit="pct", period=f"FY{y}", direct=True))
        k += 1
    c3 = _cagr(rev, 3)
    if c3 is not None:
        out.append(EvidenceItem(id=f"E{k}", kind=kind, source=f"{source_prefix} FY{years[-4]}-FY{years[-1]}",
                                content=f"{company} 3-year revenue CAGR FY{years[-4]}-FY{years[-1]}: {_pct(c3)}",
                                value=c3, unit="pct", period=f"FY{years[-4]}-FY{years[-1]}", direct=True))
        k += 1
    if not peer:
        gm = (hist.loc["gross_profit"] / rev.replace(0, np.nan))
        om = (hist.loc["operating_income"] / rev.replace(0, np.nan))
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} income statements",
                                content=f"{company} gross margin trend: " + ", ".join(f"FY{y} {_pct(v)}" for y, v in gm.items()),
                                unit="pct", period=f"FY{years[0]}-FY{years[-1]}"))
        k += 1
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} income statements",
                                content=f"{company} operating margin trend: " + ", ".join(f"FY{y} {_pct(v)}" for y, v in om.items()),
                                unit="pct", period=f"FY{years[0]}-FY{years[-1]}"))
        k += 1
        capex_int = (hist.loc["capex"] / rev.replace(0, np.nan))
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} cash flow statements",
                                content=f"{company} capex intensity (capex/revenue): " + ", ".join(f"FY{y} {_pct(v)}" for y, v in capex_int.items()),
                                unit="pct", period=f"FY{years[0]}-FY{years[-1]}"))
        k += 1
    return out


def long_history_evidence(full_hist: pd.DataFrame, company: str, source_prefix: str, start_id: int = 1, max_years: int = 12) -> list[EvidenceItem]:
    """One compact item covering every fiscal year on disk (up to ``max_years``): growth by year, long-run CAGR,
    volatility, down years. The per-year items cover the modeling window; this gives the debate the cycle context."""
    rev = full_hist.loc["revenue"]
    years = [y for y in full_hist.columns][-max_years - 1:]
    rows = []
    for y0, y in zip(years[:-1], years[1:]):
        if y == y0 + 1 and rev[y0] > 0:
            rows.append((y, rev[y] / rev[y0] - 1))
    if len(rows) < 3:
        return []
    g = np.array([r[1] for r in rows])
    n = len(rows)
    y_first, y_last = rows[0][0] - 1, rows[-1][0]
    cagr = (rev[y_last] / rev[y_first]) ** (1 / (y_last - y_first)) - 1
    content = (f"{company} long-run revenue growth FY{rows[0][0]}-FY{y_last} ({n} years): " + ", ".join(f"FY{y} {_pct(v)}" for y, v in rows)
               + f"; CAGR FY{y_first}-FY{y_last} {_pct(cagr)}; std of annual growth {_pct(float(g.std()))}; "
               f"{int((g < 0).sum())} negative year(s); worst {_pct(float(g.min()))}, best {_pct(float(g.max()))}")
    return [EvidenceItem(id=f"E{start_id}", kind="historical_metric", source=f"{source_prefix} FY{y_first}-FY{y_last} (all fiscal years available on the as-of date)",
                         content=content, unit="pct", period=f"FY{y_first}-FY{y_last}")]


def cost_evidence(hist: pd.DataFrame, company: str, source_prefix: str, start_id: int = 1) -> list[EvidenceItem]:
    """Cost lines the bear should see: cost of revenue growth vs revenue growth, SG&A share, working-capital drag."""
    out: list[EvidenceItem] = []
    k = start_id
    years = list(hist.columns)
    rev = hist.loc["revenue"]
    cogs = hist.loc["cogs"]
    if len(years) >= 2 and rev.iloc[-2] > 0 and cogs.iloc[-2] > 0:
        rg, cg = rev.iloc[-1] / rev.iloc[-2] - 1, cogs.iloc[-1] / cogs.iloc[-2] - 1
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} FY{years[-1]} income statement",
                                content=f"{company} cost of revenue grew {_pct(cg)} in FY{years[-1]} vs revenue growth {_pct(rg)}",
                                unit="pct", period=f"FY{years[-1]}"))
        k += 1
    sga = (hist.loc["sga"] / rev.replace(0, np.nan))
    out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} income statements",
                            content=f"{company} SG&A as % of revenue (cost line): " + ", ".join(f"FY{y} {_pct(v)}" for y, v in sga.items()),
                            unit="pct", period=f"FY{years[0]}-FY{years[-1]}"))
    k += 1
    dnwc = hist.loc["d_nwc"]
    cfo = hist.loc["cfo"]
    out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"{source_prefix} cash flow statements",
                            content=f"{company} change in working capital & other (cash effect, USD millions): " + ", ".join(f"FY{y} {v:,.0f}" for y, v in dnwc.items())
                                    + "; cash from operations: " + ", ".join(f"FY{y} {v:,.0f}" for y, v in cfo.items()),
                            unit="amount", period=f"FY{years[0]}-FY{years[-1]}"))
    k += 1
    return out


def wacc_evidence(hist: pd.DataFrame, company: str, source_prefix: str, analyst: dict, start_id: int = 1) -> list[EvidenceItem]:
    """Cost-of-capital facts. Filing-derived items cite the 10-K; market items cite the data source and date recorded in
    analyst_inputs.json (FRED, Yahoo regression, Damodaran); analyst overrides say so. The mechanical WACC at the end is
    the arithmetic the debate is meant to challenge (beta choice, ERP, target vs current leverage)."""
    out: list[EvidenceItem] = []
    k = start_id
    years = list(hist.columns)
    last = years[-1]
    debt = hist.loc["short_term_debt"] + hist.loc["long_term_debt"]
    avg_debt = float(debt.iloc[-2:].mean()) if len(years) >= 2 else float(debt.iloc[-1])
    kd = None
    if avg_debt > 0:
        kd = float(hist.loc["interest_expense", last] / avg_debt)
        out.append(EvidenceItem(id=f"E{k}", kind="historical_metric", source=f"{source_prefix} FY{last} interest expense / average debt",
                                content=f"{company} implied pre-tax cost of debt FY{last}: {_pct(kd)} (interest {hist.loc['interest_expense', last]:,.0f} on average debt {avg_debt:,.0f}, USD m)",
                                value=kd, unit="pct", period=f"FY{last}"))
        k += 1
    equity_book = float(hist.loc["total_equity", last])
    dw_book = None
    if equity_book > 0 and debt.iloc[-1] > 0:
        dw_book = float(debt.iloc[-1] / (debt.iloc[-1] + equity_book))
        out.append(EvidenceItem(id=f"E{k}", kind="historical_metric", source=f"{source_prefix} FY{last} balance sheet",
                                content=f"{company} book debt weight D/(D+E) FY{last}: {_pct(dw_book)} (debt {debt.iloc[-1]:,.0f}, book equity {equity_book:,.0f}, USD m)",
                                value=dw_book, unit="pct", period=f"FY{last}"))
        k += 1
    dw_mkt = analyst.get("debt_weight")
    if dw_mkt is not None and analyst.get("debt_weight_source"):
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=analyst["debt_weight_source"],
                                content=f"{company} market-value debt weight D/(D+E): {_pct(float(dw_mkt))}", value=float(dw_mkt), unit="pct"))
        k += 1
    pretax = hist.loc["pretax_income"].iloc[-3:].sum()
    tr = None
    if pretax > 0:
        tr = float(hist.loc["income_tax"].iloc[-3:].sum() / pretax)
        out.append(EvidenceItem(id=f"E{k}", kind="historical_metric", source=f"{source_prefix} FY{years[-3] if len(years) >= 3 else years[0]}-FY{last} income statements",
                                content=f"{company} 3-year effective tax rate: {_pct(tr)}" + (f" (US statutory {analyst['tax_rate_statutory']:.0%})" if analyst.get("tax_rate_statutory") else ""),
                                value=tr, unit="pct"))
        k += 1
    rf, beta, erp = analyst.get("risk_free"), analyst.get("beta"), analyst.get("erp")
    if rf is not None:
        out.append(EvidenceItem(id=f"E{k}", kind="analyst_input", source=analyst.get("risk_free_source", "analyst input"),
                                content=f"Risk-free rate: {_pct(float(rf))}", value=float(rf), unit="pct"))
        k += 1
    if erp is not None:
        out.append(EvidenceItem(id=f"E{k}", kind="analyst_input", source=analyst.get("erp_source", "analyst input"),
                                content=f"Equity risk premium (market-wide expected return over the risk-free rate): {_pct(float(erp))}", value=float(erp), unit="pct"))
        k += 1
        xc = analyst.get("erp_damodaran_crosscheck")
        if xc is not None:
            out.append(EvidenceItem(id=f"E{k}", kind="analyst_input", source=analyst.get("erp_damodaran_crosscheck_source", "Damodaran implied ERP"),
                                    content=f"Cross-check: Damodaran implied ERP {_pct(float(xc))} versus the series ERP {_pct(float(erp))}", value=float(xc), unit="pct"))
        k += 1
    if beta is not None:
        extra = ""
        if analyst.get("beta_adjusted") is not None:
            extra = f"; Blume-adjusted beta {float(analyst['beta_adjusted']):.2f}"
        if analyst.get("beta_r2") is not None:
            extra += f"; regression r2 {float(analyst['beta_r2']):.2f}"
        out.append(EvidenceItem(id=f"E{k}", kind="analyst_input", source=analyst.get("beta_source", "analyst input"),
                                content=f"{company} equity beta: {float(beta):.2f}{extra}", value=float(beta), unit="x"))
        k += 1
    for name, pb in (analyst.get("peer_betas") or {}).items():
        out.append(EvidenceItem(id=f"E{k}", kind="peer_metric", source=(analyst.get("peer_beta_sources") or {}).get(name) or analyst.get("beta_source", "analyst input"),
                                content=f"Peer {name} equity beta: {float(pb):.2f}", value=float(pb), unit="x"))
        k += 1
    if None not in (rf, beta, erp):
        ke = float(rf) + float(beta) * float(erp)
        out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source="CAPM from the risk-free, beta and ERP items above",
                                content=f"CAPM cost of equity = rf + beta x ERP = {_pct(ke)}", value=ke, unit="pct", direct=True))
        k += 1
        w = analyst.get("debt_weight", dw_book)
        t = analyst.get("tax_rate", tr if tr is not None else 0.21)
        kd_used = analyst.get("cost_of_debt", kd)
        if w is not None and kd_used is not None:
            wacc = (1 - float(w)) * ke + float(w) * float(kd_used) * (1 - float(t))
            out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source="mechanical WACC from the items above",
                                    content=f"Mechanical WACC: {_pct(wacc)} (at debt weight {_pct(float(w))}, pre-tax cost of debt {_pct(float(kd_used))}, tax rate {_pct(float(t))}, cost of equity {_pct(ke)})",
                                    value=wacc, unit="pct", direct=True))
            k += 1
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
KEYWORDS = ("expect", "guidance", "outlook", "backlog", "orders", "demand", "growth", "pricing", "capacity", "pipeline", "headwind", "tailwind")


_TEXT_BOILERPLATE = ("forward-looking statement", "annual report on form 10-k", "private securities litigation", "safe harbor",
                     "words such as", "undertakes no obligation", "such forward-looking")


def _dup_key(sent: str) -> str:
    """Near-duplicate key: the first eight normalized words (bullet and body often repeat the same sentence)."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", sent.lower()).split()[:8])


def text_evidence(sections: dict[str, str], company: str, source_prefix: str, start_id: int = 1, max_items: int = 8,
                  keywords: tuple[str, ...] = KEYWORDS, seen: set[str] | None = None) -> list[EvidenceItem]:
    """Pull forward-looking sentences out of MD&A / business sections deterministically.

    This is intentionally a keyword filter, not an LLM summary: the packet must contain the filing's own words so the
    validator can check numbers against them. Safe-harbor boilerplate is excluded; near-duplicates across sections are
    kept once; the competition excerpt is skipped when Item 1 itself is present (it is a substring of it).
    """
    out: list[EvidenceItem] = []
    k = start_id
    seen = set() if seen is None else seen
    sections = {sec: txt for sec, txt in sections.items() if txt and not (sec.lower().startswith("competition") and sections.get("Item 1 Business"))}
    for section, text in sections.items():
        sents = _SENT.split(re.sub(r"\s+", " ", text))
        scored = []
        for s in sents:
            s = s.strip()
            if not (40 <= len(s) <= 400):
                continue
            low = s.lower()
            if any(b in low for b in _TEXT_BOILERPLATE):
                continue
            hits = sum(1 for w in keywords if w in low)
            has_num = bool(re.search(r"\d", s))
            if hits:
                scored.append((hits + (1 if has_num else 0), s))
        scored.sort(key=lambda x: -x[0])
        taken = 0
        for _, s in scored:
            key = _dup_key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(EvidenceItem(id=f"E{k}", kind="filing_text", source=f"{source_prefix} {section}", content=f'"{s}"'))
            k += 1
            taken += 1
            if taken >= max(1, max_items // max(1, len(sections))):
                break
    return out


GUIDANCE_KEYWORDS = ("guidance", "expect", "outlook", "orders", "backlog", "book-to-bill", "organic", "full year", "full-year", "pipeline")
_BOILERPLATE = ("forward-looking statement", "private securities litigation", "non-gaap", "reconciliation", "conference call", "webcast",
                "safe harbor", "undertakes no obligation")
_GUIDE_SALES = re.compile(r"net sales of \$\s?([\d,]+(?:\.\d+)?)\s*(?:million|billion)?\s*(?:to|-|–)\s*\$?\s?([\d,]+(?:\.\d+)?)\s*(million|billion)", re.I)
_GUIDE_POINT = re.compile(r"net sales of \$\s?([\d,]+(?:\.\d+)?)\s*(million|billion)", re.I)


_GROWTH_PCT = re.compile(r"(?:net sales|revenue|sales) growth(?:\s*\(\d\))? of (?:approximately |about |~)?(\d+(?:\.\d+)?)%(?:\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)%)?", re.I)
_TABLE_HEAD = re.compile(r"full[- ]year (20\d\d) guidance", re.I)
_PAIR_SALES = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(M|B|million|billion)?\s*(?:to|-|–)\s*\$?\s?([\d,]+(?:\.\d+)?)\s*(M|B|million|billion)", re.I)
_PAIR_PCT = re.compile(r"\(?(-?\d+(?:\.\d+)?)%\)?\s*(?:to|-|–)\s*\(?(-?\d+(?:\.\d+)?)%\)?")


def parse_guidance_table(flat: str) -> tuple[int, tuple[float, float] | None, tuple[float, float] | None] | None:
    """(fiscal year, (sales lo, hi) in USD millions, (organic growth lo, hi)) from a guidance table. Releases print the
    quarter and the full year side by side ("Net sales $1,100M - $1,150M $5,500M - $5,800M"), so the full-year
    column is the LAST pair after each label."""
    mh = _TABLE_HEAD.search(flat)
    if not mh:
        return None
    fy = int(mh.group(1))
    window = flat[mh.end():mh.end() + 400]
    sales = organic = None
    labels = r"organic|adjusted|operating|diluted|free cash|eps|net income|margin"
    ms = re.search(r"net sales\s*(?:\(\d\))?\s*", window, re.I)
    if ms:
        seg = re.split(labels, window[ms.end():], maxsplit=1, flags=re.I)[0]  # this row only
        pairs = _PAIR_SALES.findall(seg)
        if pairs:
            lo, ulo, hi, uhi = pairs[-1]  # last pair = full-year column
            sales = (float(lo.replace(",", "")) * _mult(ulo or uhi), float(hi.replace(",", "")) * _mult(uhi))
    mo = re.search(r"organic net sales growth\s*(?:\(\d\))?\s*", window, re.I)
    if mo:
        seg = re.split(r"adjusted|operating|diluted|free cash|eps|net income|margin|net sales", window[mo.end():], maxsplit=1, flags=re.I)[0]
        pairs = _PAIR_PCT.findall(seg)
        if pairs:
            lo, hi = pairs[-1]
            organic = (float(lo) / 100, float(hi) / 100)
    if sales is None and organic is None:
        return None
    return fy, sales, organic


def _segments(text: str) -> list[str]:
    """Paragraphs, bullets and table rows first, then sentences inside them, so a bullet that runs into the dateline
    does not swallow the guidance sentence."""
    out = []
    for para in re.split(r"\n+|\s*[•\u2022]\s*", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        out.extend(x.strip() for x in _SENT.split(para) if x.strip())
    return out


def _mult(unit: str | None) -> float:
    return 1000.0 if (unit or "").lower() in ("b", "billion") else 1.0


def guidance_evidence(releases: list[dict], company: str, last_fy_revenue: float | None, start_id: int = 1,
                      max_items: int = 8, unit_divisor: float = 1.0) -> list[EvidenceItem]:
    """Forward-looking sentences from earnings releases (guidance, orders, backlog, book-to-bill), newest first,
    plus derived items: implied growth from a full-year net sales guide (range, point or table) and any explicit
    guided growth percentage. Deterministic: the release's own words, so the validator can check numbers.
    ``last_fy_revenue`` is in packet units (USD millions by default)."""
    out: list[EvidenceItem] = []
    k = start_id
    seen: set[str] = set()
    derived_done = False
    for rel in releases:
        text = rel.get("text", "")
        src = f"8-K EX-99.1 filed {rel.get('filing_date', '?')}"
        segs = _segments(text)
        scored = []
        for sent in segs:
            if not (40 <= len(sent) <= 500) or not re.search(r"\d", sent):
                continue
            low = sent.lower()
            if any(b in low for b in _BOILERPLATE):
                continue
            hits = sum(1 for w in GUIDANCE_KEYWORDS if w in low)
            if not hits or _dup_key(sent) in seen:
                continue
            score = hits
            has_pct = "%" in sent
            if ("guidance" in low or "expect" in low or "anticipate" in low or "outlook" in low) and has_pct:
                score += 4          # a guide with a number is the most valuable sentence in a release
            elif has_pct and ("growth" in low or "organic" in low or "orders" in low or "backlog" in low):
                score += 3
            elif "guidance" in low or "expect" in low:
                score += 1
            if "backlog" in low or "book-to-bill" in low or "orders" in low:
                score += 1
            scored.append((score, sent))
        scored.sort(key=lambda x: -x[0])
        per_release = max(2, max_items // max(1, len(releases)))
        taken = 0
        for _, sent in scored:
            key = _dup_key(sent)
            if key in seen:
                continue
            seen.add(key)
            out.append(EvidenceItem(id=f"E{k}", kind="filing_text", source=src, content=f'"{sent}"', period=rel.get("date_of_report") or None))
            k += 1
            taken += 1
            if taken >= per_release:
                break
        if derived_done or not last_fy_revenue:
            continue
        flat = re.sub(r"\s+", " ", text)
        lo = hi = None
        # 1) guidance table: "Full Year 2023 Guidance ... Net sales $6,450M - $6,600M ... Organic net sales growth 14% - 17%"
        tbl = parse_guidance_table(flat)
        if tbl:
            fy_t, sales_t, org_t = tbl
            if sales_t:
                lo, hi = sales_t
            if org_t:
                g_lo, g_hi = org_t
                out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"derived: guidance table in {src}",
                                        content=f"{company} guided organic net sales growth for full year {fy_t}: {_pct(g_lo)} to {_pct(g_hi)} (midpoint {_pct((g_lo + g_hi) / 2)})",
                                        value=(g_lo + g_hi) / 2, unit="pct", direct=True))
                k += 1
        # 2) full-year sentence guides: ranges or a point
        if lo is None:
            for sent in segs:
                low = sent.lower()
                if not (("full year" in low or "full-year" in low or "fiscal 20" in low or re.search(r"\b20\d\d\b", low)) and ("expect" in low or "guidance" in low or "outlook" in low or "anticipate" in low)):
                    continue
                if "quarter" in low.split("net sales")[0][-60:]:
                    continue
                m = _GUIDE_SALES.search(sent)
                if m:
                    lo, hi = float(m.group(1).replace(",", "")) * _mult(m.group(3)), float(m.group(2).replace(",", "")) * _mult(m.group(3))
                    break
                m2 = _GUIDE_POINT.search(sent)
                if m2:
                    lo = hi = float(m2.group(1).replace(",", "")) * _mult(m2.group(2))
                    break
        if lo is not None:
            lo, hi = lo / unit_divisor, hi / unit_divisor
            mid = (lo + hi) / 2
            g = mid / last_fy_revenue - 1
            if -0.5 < g < 1.5:
                rng = f"{lo:,.0f} to {hi:,.0f}" if lo != hi else f"{mid:,.0f}"
                out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"derived: guidance midpoint in {src} vs last reported FY revenue",
                                        content=f"{company} full-year net sales guidance {rng} (USD millions) implies {_pct(g)} growth at the midpoint vs last fiscal year revenue {last_fy_revenue:,.0f}",
                                        value=g, unit="pct", direct=True))
                k += 1
                derived_done = True
        # 3) an explicit guided growth percentage in prose: "expect 2023 net sales growth of 15%"
        for sent in segs:
            low = sent.lower()
            if not ("expect" in low or "anticipate" in low or "guidance" in low or "outlook" in low) or "quarter" in low[:40]:
                continue
            mg = _GROWTH_PCT.search(sent)
            if mg:
                g_lo = float(mg.group(1)) / 100
                g_hi = float(mg.group(2)) / 100 if mg.group(2) else g_lo
                out.append(EvidenceItem(id=f"E{k}", kind="derived_metric", source=f"derived: guidance stated in {src}",
                                        content=f"{company} management guided net sales growth of {_pct(g_lo)}" + (f" to {_pct(g_hi)}" if g_hi != g_lo else "") + f" (as stated in the release: '{sent[:140]}')",
                                        value=(g_lo + g_hi) / 2, unit="pct", direct=True))
                k += 1
                derived_done = True
                break
    return out


_GUIDE_YEAR = re.compile(r"(?:full[- ]year|fiscal(?: year)?|fy)\s?(20\d\d)", re.I)
_GROWTH_YEAR = re.compile(r"(20\d\d)\s+(?:organic )?net sales growth of (?:approximately |about |~)?(\d+(?:\.\d+)?)%", re.I)


def parse_initial_guide(text: str, prior_revenue: float | None) -> tuple[int, float, str] | None:
    """(fiscal year guided, growth guided, how) from one release: stated growth, table organic growth midpoint, or
    sales midpoint vs prior-year revenue. None when the release carries no full-year guide."""
    flat = re.sub(r"\s+", " ", text)
    m = _GROWTH_YEAR.search(flat)
    if m:
        return int(m.group(1)), float(m.group(2)) / 100, "stated"
    for sent in _segments(text):  # "Full year 2022 guidance at the midpoint: net sales of $5.7 billion (net sales growth of 13%)"
        low = sent.lower()
        if ("guidance" in low or "expect" in low or "anticipate" in low) and _GUIDE_YEAR.search(sent) and "quarter" not in low[:60]:
            mg = _GROWTH_PCT.search(sent)
            if mg:
                g_lo = float(mg.group(1)) / 100
                g_hi = float(mg.group(2)) / 100 if mg.group(2) else g_lo
                return int(_GUIDE_YEAR.search(sent).group(1)), (g_lo + g_hi) / 2, "stated"
    tbl = parse_guidance_table(flat)
    if tbl:
        fy, sales_t, org_t = tbl
        if org_t:
            return fy, (org_t[0] + org_t[1]) / 2, "table organic"
        if sales_t and prior_revenue:
            return fy, (sales_t[0] + sales_t[1]) / 2 / prior_revenue - 1, "table sales midpoint"
    for sent in _segments(text):
        low = sent.lower()
        if not (("expect" in low or "anticipate" in low or "guidance" in low) and _GUIDE_YEAR.search(sent)):
            continue
        if "quarter" in low.split("net sales")[0][-60:]:
            continue
        fy = int(_GUIDE_YEAR.search(sent).group(1))
        mg = _GROWTH_PCT.search(sent)
        if mg:
            g_lo = float(mg.group(1)) / 100
            g_hi = float(mg.group(2)) / 100 if mg.group(2) else g_lo
            return fy, (g_lo + g_hi) / 2, "stated"
        ms = _GUIDE_SALES.search(sent) or _GUIDE_POINT.search(sent)
        if ms and prior_revenue:
            if ms.re is _GUIDE_SALES:
                lo, hi = float(ms.group(1).replace(",", "")) * _mult(ms.group(3)), float(ms.group(2).replace(",", "")) * _mult(ms.group(3))
            else:
                lo = hi = float(ms.group(1).replace(",", "")) * _mult(ms.group(2))
            return fy, (lo + hi) / 2 / prior_revenue - 1, "sales midpoint"
    return None


def guidance_track_record(all_releases: list[dict], hist: pd.DataFrame, company: str, start_id: int = 1, max_years: int = 4) -> list[EvidenceItem]:
    """Initial full-year guide vs what the company then reported, for every fully realized year in the history.
    Management's guidance style (sandbagging or over-promising) is itself evidence for the next guide."""
    rev = hist.loc["revenue"]
    years = list(hist.columns)
    guides: dict[int, tuple[float, str, str]] = {}
    for rel in sorted(all_releases, key=lambda r: r.get("filing_date", "")):  # earliest release per guided year = initial guide
        fy_hint = int(rel.get("filing_date", "0000")[:4] or 0)
        prior = float(rev[fy_hint - 1]) if (fy_hint - 1) in years else None
        parsed = parse_initial_guide(rel.get("text", ""), prior)
        if not parsed:
            continue
        fy, g, how = parsed
        fd = rel.get("filing_date", "?")
        initial = fd[:4] == str(fy) and int(fd[5:7] or 12) <= 4  # guide issued in the first four months of the year it covers
        if fy in years and abs(fy - fy_hint) <= 1 and -0.5 < g < 1.5 and (fy not in guides or (initial and not guides[fy][3])):
            guides[fy] = (g, how, fd, initial)
    rows = []
    for fy in sorted(guides)[-max_years:]:
        if (fy - 1) not in years or rev[fy - 1] <= 0:
            continue
        actual = float(rev[fy] / rev[fy - 1] - 1)
        g, how, fd, initial = guides[fy]
        rows.append((fy, g, actual, fd, initial))
    if not rows:
        return []
    beats = sum(1 for _, g, a, _, _ in rows if a > g)
    content = (f"{company} full-year guidance vs reported revenue growth: " +
               "; ".join(f"FY{fy} {'initial guide' if ini else 'guide as of ' + fd} {_pct(g)} vs actual {_pct(a)} ({(a - g) * 100:+.1f}pp)" for fy, g, a, fd, ini in rows) +
               f". Actual beat the guide in {beats} of {len(rows)} years; mean gap {sum(a - g for _, g, a, _, _ in rows) / len(rows) * 100:+.1f}pp")
    return [EvidenceItem(id=f"E{start_id}", kind="derived_metric", source="derived: guidance track record (8-K initial guides vs 10-K actuals)",
                         content=content, unit="pct", period=f"FY{rows[0][0]}-FY{rows[-1][0]}")]


def analyst_evidence(notes: list[str], start_id: int = 1) -> list[EvidenceItem]:
    return [EvidenceItem(id=f"E{start_id + i}", kind="analyst_input", source="analyst note", content=n) for i, n in enumerate(notes)]


def build_packet(assumption_key: str, company: str, hist: pd.DataFrame, peers: dict[str, pd.DataFrame] | None = None,
                 sections: dict[str, str] | None = None, analyst: dict | None = None, source_prefix: str = "10-K",
                 as_of: str | None = None, releases: list[dict] | None = None, all_releases: list[dict] | None = None,
                 long_hist: pd.DataFrame | None = None, kpis: pd.DataFrame | None = None) -> EvidencePacket:
    spec = ASSUMPTIONS[assumption_key]
    analyst = analyst or {}
    items: list[EvidenceItem] = []
    if assumption_key in ("hashrate_growth_3y", "network_hashrate_growth_3y", "btc_price_change_3y"):
        from .templates import driver_evidence

        items += driver_evidence(assumption_key, kpis if kpis is not None else pd.DataFrame(), company, start_id=1)
        cagr = growth_evidence(hist, company, source_prefix, start_id=len(items) + 1)[-1:]
        items += [c.model_copy(update={"id": f"E{len(items) + 1}"}) for c in cagr]
        items += guidance_evidence(releases or [], company, float(hist.loc["revenue"].iloc[-1]), start_id=len(items) + 1)
        kw = {"hashrate_growth_3y": ("hashrate", "exahash", "miners", "megawatt", "capacity", "energized", "fleet", "expansion"),
              "network_hashrate_growth_3y": ("network hashrate", "difficulty", "global hashrate", "halving", "competition", "network"),
              "btc_price_change_3y": ("price of bitcoin", "bitcoin price", "spot price", "volatil", "halving", "demand for bitcoin")}[assumption_key]
        items += text_evidence(sections or {}, company, source_prefix, start_id=len(items) + 1, max_items=8, keywords=kw)
    elif assumption_key == "revenue_growth_3y":
        items += growth_evidence(hist, company, source_prefix, start_id=1)
        if long_hist is not None and len(long_hist.columns) > len(hist.columns):
            items += long_history_evidence(long_hist, company, source_prefix, start_id=len(items) + 1)
        items += cost_evidence(hist, company, source_prefix, start_id=len(items) + 1)
        items += guidance_evidence(releases or [], company, float(hist.loc["revenue"].iloc[-1]), start_id=len(items) + 1)
        items += guidance_track_record(all_releases or releases or [], hist, company, start_id=len(items) + 1)
        for name, ph in (peers or {}).items():
            items += growth_evidence(ph, name, f"{name} 10-K", start_id=len(items) + 1, peer=True)
        items += text_evidence(sections or {}, company, source_prefix, start_id=len(items) + 1)
    elif assumption_key == "wacc":
        items += wacc_evidence(hist, company, source_prefix, analyst, start_id=1)
        items += text_evidence({k: v for k, v in (sections or {}).items() if "risk" in k.lower() or "liquidity" in k.lower()},
                               company, source_prefix, start_id=len(items) + 1, max_items=4,
                               keywords=("interest rate", "credit", "leverage", "covenant", "refinanc", "volatil", "debt"))
    items += analyst_evidence(analyst.get("notes", []), start_id=len(items) + 1)
    return EvidencePacket(company=company, assumption=spec, items=items, as_of=as_of)

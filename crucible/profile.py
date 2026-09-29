"""The company profile: what industry this is and what it runs on, taken from the filings rather than assumed.

Two sources. The filing header gives the SIC code and EDGAR's industry description (data/<T>/meta.json, `sic`,
`industry`), which places the company in a family with a generic driver template. The model then reads Item 1 and the
dossier's KPI and segment items and returns the business model, the revenue and cost drivers, the KPIs the company
actually discloses, unit-economics and KPI-scheme candidates, company-specific driver templates and the competitors it
names. Every element carries a verbatim quote that is checked against the filing text; an element whose quote is not
found is dropped. The decision specs (segment, unit economics, KPI scheme, peer set, industry template, revenue method)
take their options from this profile, so a miner gets hashrate options and a phone maker does not.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .debate import _JSON_RULES, _complete
from .evidence import EvidencePacket
from .llm import LLM

# SIC families: (low, high, family key, label, generic template key, generic template formula)
SIC_FAMILIES = [
    (100, 999, "agriculture", "agriculture", "volume_price", "acreage or head count x yield x realized price; input costs per unit"),
    (1000, 1499, "mining", "mining and energy extraction", "volume_price", "production volume x realized commodity price; cash cost per unit; reserve replacement capex"),
    (1500, 1799, "construction", "construction and engineering", "backlog", "orders to backlog to revenue conversion; margin on backlog; working capital by project"),
    (2000, 2099, "consumer_staples", "food and beverage", "volume_price", "volume x price/mix; input cost inflation; marketing as % of sales"),
    (2800, 2899, "chemicals_pharma", "chemicals and pharmaceuticals", "pipeline", "product revenue by franchise; R&D pipeline; patent expiry; gross-to-net"),
    (3300, 3499, "materials", "metals and materials", "volume_price", "shipments x realized price less input costs; capacity utilization"),
    (3500, 3569, "industrial", "industrial machinery", "backlog", "orders, backlog and book-to-bill; capacity; incremental margins"),
    (3570, 3579, "hardware", "computer hardware", "segment", "units x ASP by product line where disclosed, else segment growth x margin; services attach"),
    (3580, 3599, "industrial", "industrial equipment (thermal, power, engines)", "backlog", "orders to backlog to revenue; capacity additions; price realization"),
    (3600, 3699, "electrical", "electrical and electronic equipment", "backlog", "orders and backlog; content per unit; capacity; component pricing"),
    (3700, 3799, "transportation_equipment", "transportation equipment", "volume_price", "units x ASP; production schedule; supplier pass-through"),
    (3800, 3899, "instruments", "instruments and medical devices", "segment", "procedure or installed-base volume x price; consumables attach"),
    (4000, 4799, "transport", "transportation and logistics", "volume_price", "volume x yield (revenue per unit shipped); fuel and labor per unit; fleet capex"),
    (4800, 4899, "telecom", "telecommunications and media", "subscribers", "subscribers x ARPU; churn; network capex"),
    (4900, 4999, "utilities", "utilities", "rate_base", "rate base x allowed return; volumes and rates; capex plan and regulatory lag"),
    (5000, 5199, "distribution", "wholesale distribution", "volume_price", "volume x price/mix; gross margin per unit; inventory turns"),
    (5200, 5999, "retail", "retail and restaurants", "unit_economics", "stores or units x sales per unit (comps); gross margin; new unit capex"),
    (6000, 6099, "banks", "banks", "spread", "earning assets x net interest margin; credit costs; capital ratios"),
    (6100, 6199, "finance", "non-bank finance and crypto/digital assets", "spread", "assets or production volume x realized yield or price; cost per unit; funding cost"),
    (6200, 6299, "brokers", "brokers and exchanges", "volume_price", "transaction volume x take rate; net interest on balances"),
    (6300, 6499, "insurance", "insurance", "premium", "premiums x combined ratio; investment yield on float"),
    (6500, 6799, "real_estate", "real estate and REITs", "rate_base", "occupancy x rent per unit; NOI; development pipeline"),
    (7000, 7299, "services", "personal and business services", "unit_economics", "customers or units x revenue per unit; labor cost per unit"),
    (7370, 7379, "software", "software and IT services", "subscribers", "customers x ARPU or ARR; net retention; gross margin; R&D and S&M as % of revenue"),
    (7300, 7369, "services", "business services", "unit_economics", "billable units x rate; utilization; headcount"),
    (7380, 7999, "services", "services and entertainment", "unit_economics", "customers x spend per customer; capacity utilization"),
    (8000, 8099, "healthcare_services", "healthcare services", "unit_economics", "patients or visits x revenue per unit; payer mix; labor per unit"),
    (8100, 8999, "services", "professional services", "unit_economics", "billable headcount x utilization x rate"),
]

# the library of driver templates the model can run (crucible.templates), keyed by family hint words in the industry text
LIBRARY_HINTS = {"bitcoin_miner": ("bitcoin", "crypto", "mining rig", "hashrate", "digital asset")}


def family_from_sic(sic: str | int | None, industry_text: str = "") -> dict:
    """The SIC family for a code, with the generic template that family usually runs on."""
    try:
        code = int(str(sic)[:4])
    except (TypeError, ValueError):
        code = None
    if code is not None:
        for lo, hi, key, label, tmpl, formula in SIC_FAMILIES:
            if lo <= code <= hi:
                return {"sic": code, "family": key, "label": label, "template": tmpl, "formula": formula}
    return {"sic": code, "family": "generic", "label": industry_text or "unclassified", "template": "segment", "formula": "segment growth x margin x capex intensity"}


def library_templates_for(industry_text: str, item1: str) -> list[str]:
    """Library templates whose hint words appear in the industry description or the business section."""
    low = (industry_text + " " + item1[:20000]).lower()
    return [k for k, hints in LIBRARY_HINTS.items() if any(h in low for h in hints)]


_WS = re.compile(r"\s+")


def _norm_text(t: str) -> str:
    return _WS.sub(" ", str(t or "")).lower()


def verify_quote(quote: str, sections: dict[str, str], packet: EvidencePacket | None = None) -> str | None:
    """The section (or evidence id) whose text contains the quote verbatim, else None. Quotes must be 4 to 40 words."""
    q = _norm_text(quote).strip(' "\u201c\u201d')
    if not (4 <= len(q.split()) <= 40):
        return None
    for name, text in sections.items():
        if text and q in _norm_text(text):
            return name
    if packet is not None:
        for it in packet.items:
            if q in _norm_text(it.content):
                return it.id
    return None


_SYSTEM = """ROLE: profiler
You profile a company from its own filings so that a financial model can be built on the drivers the company actually discloses.
Inputs: the SIC code and industry description from the filing header, an excerpt of Item 1 (Business), and numbered evidence items
(KPI sentences from earnings releases, segment facts, growth and margin history).
Return JSON with:
- "industry": the industry in at most 8 words, consistent with the SIC description unless the business section clearly says otherwise;
- "business_model": at most 60 words on what is sold, to whom, how it is priced;
- "revenue_drivers": up to 4 [{"name", "unit", "disclosed": true|false, "frequency": "quarterly"|"annual"|"none", "quote"}] the operating quantities revenue follows;
- "cost_drivers": up to 3 [{"name", "unit", "quote"}];
- "kpis": up to 6 [{"name", "unit", "frequency", "quote"}] operating metrics the company reports with a number;
- "kpi_scheme_candidates": 2 to 3 [{"label", "value", "description", "quote"}] ways to group those KPIs for a model (label under 6 words, value a slug);
- "unit_economics_candidates": 2 to 3 [{"label", "value", "description", "quote"}] unit definitions the disclosures support (a store, a subscriber, a unit shipped, a coin, a MWh);
- "template_candidates": 1 to 3 [{"label", "value", "formula", "drivers": [..], "quote"}] company-specific driver templates: revenue as a formula of disclosed drivers;
- "competitors_named": companies the filing itself names as competitors (empty if none).
Every "quote" is a verbatim span of 4 to 40 words copied exactly from the Item 1 excerpt or from an evidence item; an element whose quote cannot be found is discarded, so copy exactly.
Do not invent metrics the company does not disclose; a hardware company that stopped reporting units gets "disclosed": false for units.
""" + _JSON_RULES + """
Output JSON: {"industry": "...", "business_model": "...", "revenue_drivers": [...], "cost_drivers": [...], "kpis": [...], "kpi_scheme_candidates": [...],
"unit_economics_candidates": [...], "template_candidates": [...], "competitors_named": ["..."]}"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40] or "option"


def build_profile(packet: EvidencePacket, sections: dict[str, str], meta: dict, llm: LLM, seed: int = 1) -> dict:
    """Profile from the filing header, Item 1 and the dossier. Quotes are verified; unverifiable elements are dropped and counted."""
    sic = meta.get("sic")
    industry_text = str(meta.get("industry") or "")
    fam = family_from_sic(sic, industry_text)
    item1 = next((v for k, v in sections.items() if k.startswith("Item 1 ")), "") or ""
    item1_excerpt = _WS.sub(" ", item1)[:9000]
    kinds = ("filing_text", "derived_metric", "historical_metric")
    ev = [it for it in packet.items if it.kind in kinds and ("[kpi" in it.source or "segment" in it.source.lower() or it.kind != "filing_text")][:40]
    rendered = "\n".join(it.render()[:300] for it in ev)
    user = (f"FILING HEADER: SIC {sic} ({industry_text or 'no description'}); family guess from SIC: {fam['label']}.\n\n"
            f"ITEM 1 (BUSINESS) EXCERPT:\n{item1_excerpt}\n\nEVIDENCE ITEMS:\n{rendered}\n\nProfile the company.")
    stats: list[dict] = []
    raw = _complete(llm, _SYSTEM, user, seed, 0.2, stats, "profiler")
    out = {"ticker": meta.get("ticker"), "company": packet.company, "sic": sic, "industry_filing": industry_text, "family": fam,
           "industry": str(raw.get("industry", industry_text))[:80], "business_model": str(raw.get("business_model", ""))[:600],
           "dropped": 0, "model": getattr(llm, "name", "?"), "call_stats": stats}
    secs = dict(sections)
    secs["Item 1 excerpt"] = item1_excerpt

    def keep(items, fields):
        kept = []
        for x in items or []:
            if not isinstance(x, dict):
                continue
            where = verify_quote(str(x.get("quote", "")), secs, packet)
            if not where:
                out["dropped"] += 1
                continue
            y = {f: (str(x.get(f, ""))[:200] if f not in ("disclosed", "drivers") else x.get(f)) for f in fields}
            y["quote"], y["cited"] = str(x.get("quote", ""))[:300], where
            if "value" in y:
                y["value"] = _slug(y.get("value") or y.get("label"))
            kept.append(y)
        return kept

    out["revenue_drivers"] = keep(raw.get("revenue_drivers"), ("name", "unit", "disclosed", "frequency"))[:4]
    out["cost_drivers"] = keep(raw.get("cost_drivers"), ("name", "unit"))[:3]
    out["kpis"] = keep(raw.get("kpis"), ("name", "unit", "frequency"))[:6]
    out["kpi_scheme_candidates"] = keep(raw.get("kpi_scheme_candidates"), ("label", "value", "description"))[:3]
    out["unit_economics_candidates"] = keep(raw.get("unit_economics_candidates"), ("label", "value", "description"))[:3]
    out["template_candidates"] = keep(raw.get("template_candidates"), ("label", "value", "formula", "drivers"))[:3]
    out["competitors_named"] = [str(c)[:60] for c in (raw.get("competitors_named") or []) if isinstance(c, str) and c.strip()][:10]
    out["library_templates"] = library_templates_for(industry_text, item1)
    return out


def load_profile(data_dir: Path, ticker: str) -> dict | None:
    p = Path(data_dir) / ticker.upper() / "profile.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_profile(data_dir: Path, ticker: str, profile: dict) -> Path:
    p = Path(data_dir) / ticker.upper() / "profile.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2))
    return p


def profile_markdown(pr: dict) -> str:
    fam = pr.get("family") or {}
    md = [f"# {pr.get('company')} profile", "",
          f"Industry (filing header): {pr.get('industry_filing') or 'n/a'} (SIC {pr.get('sic')}); family: {fam.get('label')}; model reads: {pr.get('industry')}", "",
          f"Business model: {pr.get('business_model')}", ""]
    for key, title in (("revenue_drivers", "Revenue drivers"), ("cost_drivers", "Cost drivers"), ("kpis", "KPIs disclosed"),
                       ("kpi_scheme_candidates", "KPI scheme candidates"), ("unit_economics_candidates", "Unit economics candidates"), ("template_candidates", "Template candidates")):
        rows = pr.get(key) or []
        if rows:
            md.append(f"## {title}")
            for r in rows:
                head = r.get("name") or r.get("label")
                extra = " | ".join(str(r[k]) for k in ("unit", "frequency", "description", "formula") if r.get(k))
                md.append(f"- {head}" + (f" ({extra})" if extra else "") + f' | quote: "{r.get("quote", "")}" [{r.get("cited")}]')
            md.append("")
    if pr.get("competitors_named"):
        md.append("Competitors named: " + ", ".join(pr["competitors_named"]))
    md.append(f"\nlibrary templates matched: {', '.join(pr.get('library_templates') or []) or 'none'}; elements dropped for unverifiable quotes: {pr.get('dropped', 0)}")
    return "\n".join(md)

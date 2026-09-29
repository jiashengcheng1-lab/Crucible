"""The company dossier: one evidence packet per company that every decision and the brief draw from.

Deterministic builders only. Text items are verbatim filing sentences chosen by keyword family, so anything the judge
cites is filing text by construction. Macro features come from FRED (when a key is set), the capital-allocation record
from the facts, incentive metrics from the proxy statement when it has been ingested.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .decisions import DecisionSpec, Option
from .evidence import (AssumptionSpec, EvidenceItem, EvidencePacket, _pct, cost_evidence, growth_evidence, guidance_evidence,
                       guidance_track_record, long_history_evidence)

DOSSIER_SPEC = AssumptionSpec(key="company_dossier", description="shared evidence for modeling decisions", unit="text", lower_bound=0, upper_bound=1)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# keyword families: which filing sentences matter for which decisions
FAMILIES = {
    "segments": ("segment", "reportable", "operating segments", "geographic", "product line", "business unit", "disaggregat"),
    "unit_economics": ("per unit", "per bitcoin", "cost to mine", "cost per", "average selling price", "average revenue per", "arpu", "unit cost",
                       "cost of power", "per megawatt", "per mwh", "per coin", "hosting fee", "price realization", "contribution margin"),
    "kpi": ("hashrate", "hash rate", "exahash", "eh/s", "bitcoin mined", "btc mined", "backlog", "orders", "book-to-bill", "megawatt", " mw ",
            "capacity", "utilization", "subscribers", "active users", "arr", "units shipped", "fleet", "efficiency", "j/th"),
    "competition": ("compet", "rival", "market share", "peers", "industry participants", "fragmented", "consolidat", "substitute"),
    "pricing": ("pricing", "price increase", "pass through", "surcharge", "price realization", "commodit", "switching cost", "spot price",
                "bitcoin price", "hashprice", "contract", "long-term agreement", "pricing power", "inflation"),
    "cycle": ("cyclical", "cycle", "demand environment", "macroeconomic", "interest rate", "recession", "slowdown", "supply chain",
              "halving", "difficulty", "capacity expansion", "overcapacity"),
    "incentives": ("annual incentive", "long-term incentive", "performance-based", "performance share", "return on invested capital", "roic",
                   "free cash flow", "adjusted ebitda", "total shareholder return", "tsr", "compensation committee", "peer group", "metric"),
    "allocation": ("capital allocation", "share repurchase", "buyback", "dividend", "acquisition", "merger", "reinvest", "leverage target",
                   "net leverage", "debt reduction", "capital expenditure"),
    "normalization": ("non-gaap", "adjusted ebitda", "adjusted", "stock-based compensation", "share-based compensation", "restructuring",
                      "impairment", "one-time", "non-recurring", "operating lease", "finance lease", "lease liabilit", "right-of-use"),
    "forecast": ("backlog", "orders", "capacity", "utilization", "average selling price", "volume", "pricing", "mix", "fixed cost",
                 "variable cost", "operating leverage", "hashrate", "bitcoin price", "energy cost", "hosting", "megawatt"),
    "capex": ("capital expenditure", "maintenance capital", "growth capital", "capacity", "expansion", "new facility", "miners purchased",
              "fleet upgrade", "committed", "purchase commitments"),
    "valuation": ("segment", "backlog", "recurring", "contract", "volatility", "bitcoin", "cyclical", "cash flow", "leverage", "dilution"),
}

KPI_PATTERNS = [
    ("hashrate", re.compile(r"(\d+(?:\.\d+)?)\s*(?:EH/s|exahash(?:es)? per second|EH)", re.I)),
    ("bitcoin_mined", re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:bitcoin|BTC)\s*(?:mined|produced)", re.I)),
    ("cost_per_bitcoin", re.compile(r"cost (?:to mine|per)\s*(?:a |one |each )?(?:bitcoin|BTC)[^.]{0,60}?\$\s?([\d,]+)", re.I)),
    ("backlog", re.compile(r"backlog[^.]{0,80}?\$\s?([\d,.]+)\s*(million|billion)", re.I)),
    ("orders", re.compile(r"orders[^.]{0,60}?(?:increased|decreased|up|down|grew|declined)[^.]{0,20}?(\d+(?:\.\d+)?)%", re.I)),
    ("book_to_bill", re.compile(r"book-to-bill[^.]{0,40}?(\d+(?:\.\d+)?)", re.I)),
    ("capacity_mw", re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:MW|megawatts?)\b", re.I)),
    ("fleet_efficiency", re.compile(r"(\d+(?:\.\d+)?)\s*J/TH", re.I)),
]

INDUSTRY_TEMPLATES = {
    "bitcoin_miner": ("hashrate x hashprice: revenue = EH/s x network share x BTC price; cost = power (MW x price) + hosting; capex = miners", ("hashrate", "bitcoin", "hosting", "megawatt")),
    "data_center_supplier": ("orders to backlog to revenue: orders, backlog conversion, capacity; margins from mix and price realization", ("backlog", "orders", "data center", "thermal", "power")),
    "industrial_generic": ("segment growth x margin: segment revenue growth, incremental margins, capex intensity", ("segment", "industrial")),
    "subscription_software": ("ARR build: customers x ARPU, net retention, gross margin", ("subscription", "arr", "recurring")),
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(re.sub(r"\s+", " ", text or "")) if 40 <= len(s.strip()) <= 420]


def family_items(sections: dict[str, str], family: str, source_prefix: str, start_id: int, per_family: int = 6, seen: set | None = None) -> list[EvidenceItem]:
    kws = FAMILIES[family]
    seen = set() if seen is None else seen
    scored = []
    for section, text in sections.items():
        if not text or section.lower().startswith("competition"):
            continue
        for s in _sentences(text):
            low = s.lower()
            hits = sum(1 for k in kws if k in low)
            if hits and not any(b in low for b in ("forward-looking", "safe harbor", "undertakes no obligation")):
                scored.append((hits + (1 if re.search(r"\d", s) else 0) + (1 if section.startswith("Item 8") else 0), section, s))
    scored.sort(key=lambda x: -x[0])
    out, k = [], start_id
    for _, section, s in scored:
        key = s.lower()[:70]
        if key in seen:
            continue
        seen.add(key)
        out.append(EvidenceItem(id=f"E{k}", kind="filing_text", source=f"{source_prefix} {section} [{family}]", content=f'"{s}"'))
        k += 1
        if len(out) >= per_family:
            break
    return out


def kpi_items(releases: list[dict], company: str, start_id: int, max_items: int = 8) -> list[EvidenceItem]:
    """KPIs quoted from the releases as verbatim sentences, newest first, one per pattern per release."""
    out, k, seen = [], start_id, set()
    for rel in releases[:6]:
        text = re.sub(r"\s+", " ", rel.get("text", ""))
        for name, pat in KPI_PATTERNS:
            m = pat.search(text)
            if not m or (name, rel.get("filing_date")) in seen:
                continue
            seen.add((name, rel.get("filing_date")))
            s = max(0, text.rfind(".", 0, m.start()) + 1)
            e = text.find(".", m.end())
            sent = text[s:e + 1 if e > 0 else m.end() + 80].strip()
            if 20 <= len(sent) <= 400:
                out.append(EvidenceItem(id=f"E{k}", kind="filing_text", source=f"8-K EX-99.1 filed {rel.get('filing_date', '?')} [kpi:{name}]", content=f'"{sent}"',
                                        period=rel.get("date_of_report") or None))
                k += 1
            if len(out) >= max_items:
                return out
    return out


def segment_items(data_dir: Path, ticker: str, company: str, start_id: int) -> list[EvidenceItem]:
    """Segment revenue by member from dimensional XBRL facts (xbrl_segments_<FY>.parquet), latest fiscal year."""
    d = Path(data_dir) / ticker.upper()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("xbrl_segments_*.parquet"))]
    if not frames:
        return []
    x = pd.concat(frames, ignore_index=True)
    x = x[x.concept.str.lower().str.contains("revenue")]
    if not len(x):
        return []
    fy = x.fiscal_year.max()
    x = x[x.fiscal_year == fy]
    rows = x.groupby(["axis", "member"]).numeric_value.max().sort_values(ascending=False)
    if not len(rows):
        return []
    parts = "; ".join(f"{str(m).split(':')[-1].replace('Member', '')} {v / 1e6:,.0f}" for (a, m), v in rows.head(10).items())
    axes = sorted({str(a).split(":")[-1] for a, _ in rows.index})
    return [EvidenceItem(id=f"E{start_id}", kind="derived_metric", source=f"10-K FY{fy} dimensional XBRL (segment facts)",
                         content=f"{company} FY{fy} revenue by disclosed segment member (USD m): {parts}. Axes: {', '.join(axes)}", period=f"FY{fy}")]


def allocation_items(hist: pd.DataFrame, company: str, start_id: int) -> list[EvidenceItem]:
    """Five-year capital allocation record and a ROIC proxy from the mapped history."""
    yrs = list(hist.columns)
    n = len(yrs)
    def tot(k):
        return float(hist.loc[k].sum()) if k in hist.index else 0.0
    capex, div, bb, sbc = tot("capex"), tot("dividends"), tot("buybacks"), tot("sbc")
    cfo = tot("cfo")
    nd0 = float((hist.loc["short_term_debt"] + hist.loc["long_term_debt"] - hist.loc["cash"]).iloc[0])
    nd1 = float((hist.loc["short_term_debt"] + hist.loc["long_term_debt"] - hist.loc["cash"]).iloc[-1])
    sh0, sh1 = float(hist.loc["shares_diluted"].iloc[0]), float(hist.loc["shares_diluted"].iloc[-1])
    inv = float((hist.loc["short_term_debt"] + hist.loc["long_term_debt"] + hist.loc["total_equity"]).iloc[-1])
    nopat = float(hist.loc["operating_income"].iloc[-1]) * 0.79
    roic = nopat / inv if inv > 0 else float("nan")
    items = [EvidenceItem(id=f"E{start_id}", kind="derived_metric", source=f"10-K FY{yrs[0]}-FY{yrs[-1]} cash flow statements",
                          content=(f"{company} capital allocation FY{yrs[0]}-FY{yrs[-1]} (USD m): cash from operations {cfo:,.0f}; capex {capex:,.0f}; "
                                   f"dividends {div:,.0f}; buybacks {bb:,.0f}; stock comp {sbc:,.0f}; net debt {nd0:,.0f} -> {nd1:,.0f}; "
                                   f"diluted shares {sh0:,.0f}m -> {sh1:,.0f}m ({(sh1 / sh0 - 1) if sh0 else 0:+.0%})"), period=f"FY{yrs[0]}-FY{yrs[-1]}"),
             EvidenceItem(id=f"E{start_id + 1}", kind="derived_metric", source=f"10-K FY{yrs[-1]} (NOPAT at 21% tax / debt plus equity)",
                          content=f"{company} ROIC proxy FY{yrs[-1]}: {_pct(roic)} (operating income {float(hist.loc['operating_income'].iloc[-1]):,.0f}, invested capital {inv:,.0f})",
                          value=roic if roic == roic else None, unit="pct", period=f"FY{yrs[-1]}")]
    return items


def proxy_items(data_dir: Path, ticker: str, company: str, start_id: int) -> list[EvidenceItem]:
    """Incentive metrics from the proxy statement (DEF 14A) when ingested: metric mentions in the CD&A plus sentences."""
    p = Path(data_dir) / ticker.upper() / "proxy_text.json"
    if not p.exists():
        return []
    try:
        px = json.loads(p.read_text())
    except Exception:
        return []
    text = px.get("text", "")
    if not text:
        return []
    low = text.lower()
    metrics = {"ROIC/ROCE": len(re.findall(r"\broic\b|return on invested capital|\broce\b|return on capital employed", low)),
               "free cash flow": low.count("free cash flow"), "adjusted EBITDA": low.count("adjusted ebitda"), "revenue": low.count("revenue"),
               "EPS": len(re.findall(r"\beps\b|earnings per share", low)), "TSR": len(re.findall(r"\btsr\b|total shareholder return", low)),
               "hashrate/operational": len(re.findall(r"hashrate|exahash|operational", low))}
    items = [EvidenceItem(id=f"E{start_id}", kind="derived_metric", source=f"DEF 14A filed {px.get('filing_date', '?')} (metric mention counts)",
                          content=f"{company} proxy statement metric mentions: " + ", ".join(f"{k} {v}" for k, v in metrics.items()))]
    fam = family_items({"DEF 14A CD&A": text}, "incentives", f"DEF 14A filed {px.get('filing_date', '?')}", start_id + 1, per_family=6)
    return items + fam


def macro_items(as_of: str | None, data_dir: Path, start_id: int, fetch=None) -> list[EvidenceItem]:
    """Cycle features from FRED: industrial production growth, unemployment change, yield curve, credit spread, 10-year."""
    from . import marketdata as md

    fetch = fetch or md.fetch_fred
    out, k = [], start_id
    specs = [("INDPRO", "industrial production, y/y", "yoy"), ("UNRATE", "unemployment rate, 12-month change (pp)", "d12"),
             ("T10Y2Y", "10y-2y Treasury spread (pp)", "level"), ("BAA10Y", "Baa corporate spread over 10y (pp)", "level"), ("DGS10", "10-year Treasury (%)", "level")]
    for sid, label, mode in specs:
        try:
            s = md.fred_series(sid, as_of, years=3, data_dir=data_dir, fetch=fetch)
        except Exception:
            continue
        if not len(s):
            continue
        s = s.set_index("date")["value"].astype(float)
        last = float(s.iloc[-1])
        if mode == "yoy":
            prior = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
            val = (last / float(prior.iloc[-1]) - 1) * 100 if len(prior) and float(prior.iloc[-1]) else float("nan")
            content = f"{label}: {val:+.1f}% as of {s.index[-1].date()} (FRED {sid})"
        elif mode == "d12":
            prior = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
            val = last - float(prior.iloc[-1]) if len(prior) else float("nan")
            content = f"{label}: {val:+.2f} (level {last:.1f}%) as of {s.index[-1].date()} (FRED {sid})"
        else:
            val = last
            content = f"{label}: {last:.2f} as of {s.index[-1].date()} (FRED {sid}); 12-month range {float(s.tail(252).min()):.2f} to {float(s.tail(252).max()):.2f}"
        out.append(EvidenceItem(id=f"E{k}", kind="macro", source=f"FRED {sid}", content=content, value=val if val == val else None, unit="pct"))
        k += 1
    return out


def build_dossier(data_dir: Path, ticker: str, hist: pd.DataFrame, long_hist: pd.DataFrame | None, sections: dict[str, str], releases: list[dict],
                  all_releases: list[dict], peers: dict[str, pd.DataFrame] | None = None, peer_sections: dict[str, dict] | None = None,
                  company: str | None = None, as_of: str | None = None, with_macro: bool = True) -> EvidencePacket:
    company = company or ticker.upper()
    items: list[EvidenceItem] = []
    items += growth_evidence(hist, company, "10-K", start_id=1)
    if long_hist is not None and len(long_hist.columns) > len(hist.columns):
        items += long_history_evidence(long_hist, company, "10-K", start_id=len(items) + 1)
    items += cost_evidence(hist, company, "10-K", start_id=len(items) + 1)
    items += guidance_evidence(releases, company, float(hist.loc["revenue"].iloc[-1]), start_id=len(items) + 1)
    items += guidance_track_record(all_releases, hist, company, start_id=len(items) + 1)
    items += kpi_items(all_releases, company, start_id=len(items) + 1)
    items += segment_items(data_dir, ticker, company, start_id=len(items) + 1)
    items += allocation_items(hist, company, start_id=len(items) + 1)
    items += proxy_items(data_dir, ticker, company, start_id=len(items) + 1)
    seen: set = set()
    for fam in ("segments", "unit_economics", "kpi", "competition", "pricing", "cycle", "allocation", "normalization", "forecast", "capex", "valuation"):
        items += family_items(sections, fam, "10-K", len(items) + 1, per_family=5, seen=seen)
    comp = sections.get("Competition excerpt") or ""
    if comp:
        for s in _sentences(comp)[:6]:
            key = s.lower()[:70]
            if key not in seen:
                seen.add(key)
                items.append(EvidenceItem(id=f"E{len(items) + 1}", kind="filing_text", source="10-K Item 1 Competition excerpt", content=f'"{s}"'))
    for name, ph in (peers or {}).items():
        cagr = growth_evidence(ph, name, f"{name} 10-K", start_id=len(items) + 1, peer=True)[-1:]  # the peer CAGR line only
        items += [c.model_copy(update={"id": f"E{len(items) + 1}"}) for c in cagr]
    if with_macro:
        try:
            items += macro_items(as_of, data_dir, len(items) + 1)
        except Exception:
            pass
    return EvidencePacket(company=company, assumption=DOSSIER_SPEC, items=items, as_of=as_of)


# ----------------------------------------------------------------------------- decision specs

_GEO = {"Americas", "Asia", "Asia Pacific", "Europe", "Middle East", "Africa", "North America", "Latin America", "China", "India", "Japan", "Germany",
        "United States", "EMEA", "APAC", "Canada", "Mexico", "Brazil", "Korea", "Taiwan", "Ireland", "Italy", "France", "United Kingdom"}
_GENERIC = {"The", "Our", "We", "Company", "Item", "Business", "Competition", "Data", "United", "States", "Annual", "Report", "Form", "Bitcoin", "AI",
            "These", "Examples", "Some", "Many", "Other", "Certain", "Several", "In", "As", "For", "Additionally", "However", "Because", "If", "This",
            "Competitors", "Competitor", "Customers", "Customer", "Products", "Product", "Services", "Service", "Board", "Directors", "Securities", "Exchange",
            "Commission", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
            "Chief", "Executive", "Officer", "President", "Nasdaq", "NYSE", "GAAP", "SEC", "Bitcoin Mining", "Hosting", "HPC", "GPU", "ASIC", "ATM",
            "ESG", "EBITDA", "FERC", "ERCOT", "PJM", "MW", "EH", "BTC", "ETH", "USD", "IT", "OEM", "IoT", "LLC", "IPO"}


def competitor_names(sections: dict[str, str], company: str) -> list[str]:
    """Company names the filing puts next to 'compete': the part of a competition sentence after 'include', 'such as',
    'including', 'against' or 'with', split on commas and 'and', kept when it looks like a proper name."""
    text = " ".join(sections.get(k, "") for k in sections if k.startswith("Item 1") or k.startswith("Competition"))
    own = {w for w in re.findall(r"[A-Z][A-Za-z]+", company)}
    freq: dict[str, int] = {}
    for sent in _sentences(text):
        if "compet" not in sent.lower():
            continue
        m = re.search(r"(?:includ(?:e|es|ing)|such as|against|with|from|by)\s+(.+)$", sent)
        tail = m.group(1) if m else sent
        for chunk in re.split(r",|;| and | or |\band\b", tail):
            chunk = chunk.strip(" .")
            mm = re.match(r"^((?:[A-Z][A-Za-z&\.'-]*\s?){1,4})", chunk)
            if not mm:
                continue
            name = mm.group(1).strip()
            words = name.split()
            if len(name) < 3 or name in _GEO or words[0] in _GENERIC or words[0] in own or words[-1] in ("Our", "In", "The", "Act"):
                continue
            if name.lower() in ("inc", "llc", "corp", "corporation", "ltd", "s.e", "plc"):
                continue
            has_suffix = bool(re.search(r"\b(Inc|Corp|Corporation|Plc|PLC|SA|S\.A|GmbH|AG|Ltd|Holdings?|Electric|Industries|Controls|Technologies|Group|Co|Company|International|Systems|Energy|Digital|Mining|Platforms)\b\.?$", name))
            acronym = len(words) == 1 and name.isupper() and 2 <= len(name) <= 6
            if not (has_suffix or acronym or len(words) >= 2) or (len(words) == 1 and (name.endswith("ing") or name.endswith("ly"))):
                continue
            if len(words) >= 2 and not has_suffix and any(w.endswith("ing") or w.endswith("ly") for w in words):
                continue
            freq[name] = freq.get(name, 0) + 1
    return [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]


def _slug_local(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40] or "option"


def peer_options(sections: dict[str, str], current_peers: list[str], company: str = "", named_extra: list[str] | None = None) -> list[Option]:
    """Candidate peer schemes: what the filing names (regex plus the profile's model-read names), the current set, and a
    hybrid; tickers are for the analyst to supply."""
    named = list(dict.fromkeys(competitor_names(sections, company) + [n for n in (named_extra or []) if n]))[:10]
    return [Option("filing-named competitors", "named:" + ",".join(named) if named else "named:none", f"Companies the 10-K names as competitors: {', '.join(named) or 'none extracted'}; tickers to be supplied"),
            Option("current peer set", "current:" + ",".join(current_peers), f"The peers already ingested: {', '.join(current_peers) or 'none'}"),
            Option("size-and-model matched", "matched:tbd", "Peers matched on business model, end market and scale rather than on filing mentions; analyst to nominate")]


def _profile_options(profile: dict | None, key: str, fallback: list[Option], extra_last: Option | None = None) -> list[Option]:
    """Options from the company profile (quote-verified), else the fallback list; the generic last option is always kept."""
    opts: list[Option] = []
    for c in (profile or {}).get(key) or []:
        desc = str(c.get("description") or c.get("formula") or "")
        cite = f' (filing: "{c.get("quote", "")[:120]}")' if c.get("quote") else ""
        opts.append(Option(str(c.get("label", ""))[:60], str(c.get("value") or c.get("label")), desc[:220] + cite))
    if not opts:
        return fallback
    if extra_last and extra_last.label not in {o.label for o in opts}:
        opts.append(extra_last)
    return opts[:4]


def decision_specs(sections: dict[str, str], current_peers: list[str], company: str, profile: dict | None = None) -> list[DecisionSpec]:
    fam = (profile or {}).get("family") or {}
    industry_line = f"{(profile or {}).get('industry_filing') or ''} (SIC {(profile or {}).get('sic') or '?'})" if profile else "unknown"
    # industry templates: the company's own candidates first, then the SIC family's generic template, then library templates the filing text matches
    tmpl: list[Option] = []
    for c in (profile or {}).get("template_candidates") or []:
        tmpl.append(Option(f"{c.get('label', 'company template')} (from the filing)", str(c.get("value") or _slug_local(c.get("label", ""))),
                           f"{c.get('formula', '')}; drivers: {', '.join(c.get('drivers') or [])}; filing: \"{c.get('quote', '')[:100]}\""))
    if fam:
        tmpl.append(Option(f"{fam.get('label', 'family')} template (SIC family)", str(fam.get("template", "segment")), f"generic for SIC {fam.get('sic')}: {fam.get('formula', '')}"))
    for k in (profile or {}).get("library_templates") or []:
        if k in INDUSTRY_TEMPLATES:
            tmpl.append(Option(k, k, INDUSTRY_TEMPLATES[k][0] + " (library template the model can run)"))
    if not tmpl:
        tmpl = [Option(k, k, v[0]) for k, v in INDUSTRY_TEMPLATES.items()]
    tmpl = tmpl[:4] if len(tmpl) > 1 else tmpl + [Option("segment growth x margin (generic)", "segment", "segment revenue growth, incremental margins, capex intensity")]
    kpi_fallback = [Option("operating KPIs disclosed", "operating", "The operating metrics the company reports each quarter (volumes, capacity, pricing) as disclosed"),
                    Option("demand KPIs", "demand", "Orders, backlog, book-to-bill, pipeline: what customers have committed"),
                    Option("financial KPIs only", "financial", "Growth, margins, cash conversion; no operating KPI is reliably disclosed")]
    ue_fallback = [Option("volume x price", "volume_price", "Units times a realized price or rate, as the filings disclose them"),
                   Option("cost per unit", "cost_per_unit", "Direct cost per unit produced versus realized price"),
                   Option("customer economics", "customer", "Customers or contracts times revenue per customer, retention and acquisition cost"),
                   Option("none disclosed", "none", "The filings do not disclose a unit; model on aggregate growth and margins")]
    specific_desc = "A driver the filings disclose that determines revenue"
    if (profile or {}).get("template_candidates"):
        c = profile["template_candidates"][0]
        specific_desc = f"{c.get('formula', specific_desc)} (drivers: {', '.join(c.get('drivers') or [])})"
    peer_opts = peer_options(sections, current_peers, company, named_extra=(profile or {}).get("competitors_named") or [])
    return [
        DecisionSpec("segment_scheme", f"Which segment scheme should the {company} model use?",
                     [Option("reported segments", "reported", "The segments the company reports under ASC 280 (segment note), as disclosed"),
                      Option("product or service lines", "product", "Revenue by product or service line as disaggregated in the revenue note, even if not a reportable segment"),
                      Option("geography", "geography", "Revenue by geographic region"),
                      Option("single segment", "single", "One consolidated line; the company does not disclose usable segments")],
                     evidence_keywords=FAMILIES["segments"] + ("segment",), considerations=("what the filing discloses", "stability of the scheme over time", "link to drivers and KPIs"),
                     settles="the segment note and the revenue disaggregation table"),
        DecisionSpec("unit_economics", f"Which unit-economics scheme fits {company} ({industry_line})?",
                     _profile_options(profile, "unit_economics_candidates", ue_fallback, Option("none disclosed", "none", "The filings do not disclose a unit; model on aggregate growth and margins")),
                     evidence_keywords=FAMILIES["unit_economics"] + FAMILIES["kpi"], considerations=("whether the unit is disclosed each quarter", "whether price and cost per unit are both observable"),
                     settles="MD&A and release disclosures of units, realized prices and unit costs"),
        DecisionSpec("kpi_scheme", f"Which KPI set should drive the {company} model ({industry_line})?",
                     _profile_options(profile, "kpi_scheme_candidates", kpi_fallback, Option("financial KPIs only", "financial", "Growth, margins, cash conversion; no operating KPI is reliably disclosed")),
                     evidence_keywords=FAMILIES["kpi"], considerations=("disclosure frequency", "leading versus lagging", "auditability"),
                     settles="which KPIs appear in every release with a number"),
        DecisionSpec("peer_set", f"Which peer set should {company} be compared with?", peer_opts,
                     evidence_keywords=FAMILIES["competition"], considerations=("business model match", "end market", "scale", "data availability"),
                     settles="the competition section and segment mix"),
        DecisionSpec("pricing_power", f"Does {company} have high or low pricing power?",
                     [Option("high pricing power", "high", "Can raise prices or pass through costs without losing volume: contracts, switching costs, differentiated product, backlog"),
                      Option("low pricing power", "low", "Price taker: commodity output, spot-price exposure, fragmented competition, price used to win share")],
                     evidence_keywords=FAMILIES["pricing"] + FAMILIES["competition"], considerations=("gross margin trend", "price realization language", "contract structure", "commodity exposure"),
                     settles="margin behavior when input costs rose and the filing's own language on pricing"),
        DecisionSpec("market_cycle", f"Where is {company}'s demand cycle now?",
                     [Option("expansion", "expansion", "Demand and orders accelerating, margins expanding, capacity being added"),
                      Option("peak", "peak", "Growth still high but decelerating, capacity catching up, pricing at its best"),
                      Option("contraction", "contraction", "Orders or volumes falling, margins compressing, capacity idle"),
                      Option("trough", "trough", "Demand at a low, cost cuts done, early signs of recovery")],
                     evidence_keywords=FAMILIES["cycle"] + FAMILIES["kpi"], considerations=("company KPIs versus macro", "cyclical versus secular drivers", "what the last two releases changed"),
                     settles="order and backlog trajectory against the macro features", tournament=True),
        DecisionSpec("management_alignment", f"Are {company}'s executive incentives aligned with value-creating capital allocation?",
                     [Option("good alignment (value creating)", "good", "Incentives tied to ROIC, ROCE, free cash flow per share or TSR; allocation record shows returns above the cost of capital"),
                      Option("poor alignment (value destroying)", "poor", "Incentives tied to size (revenue, EBITDA, hashrate) or EPS alone; allocation record shows dilution, empire building or buybacks above value")],
                     evidence_keywords=FAMILIES["incentives"] + FAMILIES["allocation"], considerations=("metrics in the proxy", "dilution history", "returns on capital", "M&A record"),
                     settles="the CD&A metric table and the five-year allocation record"),
        DecisionSpec("normalization_policy", f"Which adjustments should the {company} model apply?",
                     [Option("GAAP as reported", "gaap", "No adjustments; stock comp is an expense, leases as reported, one-offs left in"),
                      Option("normalize one-offs only", "one_offs", "Exclude impairments, restructuring and settlements from margin trends; keep stock comp as an expense"),
                      Option("full non-GAAP policy", "non_gaap", "Follow the company's adjusted metrics: add back stock comp and amortization, exclude one-offs, treat leases as debt")],
                     evidence_keywords=FAMILIES["normalization"], considerations=("size and recurrence of adjustments", "stock comp as a share of revenue", "lease materiality", "peer comparability"),
                     settles="the reconciliation table sizes and how often the same one-off recurs"),
        DecisionSpec("revenue_method", f"How should {company} revenue be built?",
                     [Option("segment build", "segment", "Growth rates by reported segment or product line"),
                      Option("price x volume", "price_volume", "Units times realized price; volume tied to capacity, price to the market"),
                      Option("company-specific driver", "specific", specific_desc)],
                     evidence_keywords=FAMILIES["forecast"], considerations=("what is disclosed quarterly", "link to KPIs", "explains history"),
                     settles="which build reconciles the last three years within a small residual"),
        DecisionSpec("cost_method", f"How should {company} costs be structured?",
                     [Option("fixed vs variable", "fixed_variable", "Split COGS and SG&A into fixed dollars and variable percent of revenue; operating leverage explicit"),
                      Option("percent of revenue", "pct_revenue", "All costs as ratios to revenue; simplest, no leverage"),
                      Option("company-specific cost driver", "specific", "Costs tied to the operating driver the filings disclose" + (f": {', '.join(c.get('name', '') for c in profile['cost_drivers'][:3])}" if (profile or {}).get('cost_drivers') else ""))],
                     evidence_keywords=FAMILIES["forecast"] + FAMILIES["unit_economics"], considerations=("cost disclosure", "margin history versus volume", "input price exposure"),
                     settles="whether margins moved with volume or with input prices in the history"),
        DecisionSpec("capex_method", f"How should {company} capex be forecast?",
                     [Option("maintenance plus growth", "maintenance_growth", "Maintenance capex near depreciation plus growth capex tied to the revenue build's capacity"),
                      Option("percent of revenue", "pct_revenue", "Capex intensity held at its trailing ratio"),
                      Option("committed program", "committed", "Disclosed purchase commitments and expansion programs drive the next years explicitly")],
                     evidence_keywords=FAMILIES["capex"], considerations=("capacity constraint", "disclosed commitments", "depreciation versus capex history"),
                     settles="disclosed commitments and capacity plans"),
        DecisionSpec("industry_template", f"Which driver template fits {company} ({industry_line})?", tmpl,
                     evidence_keywords=FAMILIES["kpi"] + FAMILIES["segments"] + FAMILIES["forecast"], considerations=("driver disclosure", "cost structure", "capital intensity"),
                     settles="the business description and the KPIs disclosed"),
        DecisionSpec("valuation_method", f"Which valuation method should lead for {company}?",
                     [Option("DCF", "dcf", "Discounted cash flow on the driver model"),
                      Option("multiples vs peers", "multiples", "EV/EBITDA, EV/sales or P/E against the peer set"),
                      Option("sum of the parts", "sotp", "Value distinct segments separately"),
                      Option("reverse DCF", "reverse_dcf", "Back out what the price implies and judge it")],
                     evidence_keywords=FAMILIES["valuation"], considerations=("cash flow predictability", "peer availability", "segment distinctness", "volatility of the driver"),
                     settles="whether cash flows are forecastable and whether clean peers exist"),
        DecisionSpec("dcf_method", f"Which DCF variant and terminal method for {company}?",
                     [Option("unlevered FCFF, perpetual growth", "fcff_gordon", "Free cash flow to the firm at WACC with a Gordon growth terminal value"),
                      Option("unlevered FCFF, exit multiple", "fcff_exit", "FCFF at WACC with an EV/EBITDA exit multiple from peers"),
                      Option("levered FCFE at cost of equity", "fcfe", "Cash flow to equity at the cost of equity; for stable leverage or financial businesses")],
                     evidence_keywords=FAMILIES["valuation"] + ("debt", "leverage", "convertible"), considerations=("leverage stability", "terminal sensitivity", "peer multiple availability"),
                     settles="leverage path and whether a defensible peer multiple exists"),
        DecisionSpec("scenarios", f"Which scenario framing should the {company} model carry?",
                     [Option("bull / base / bear driver sets", "three", "Three driver sets (growth, margin, multiple) with probabilities and signposts"),
                      Option("base with sensitivity grid", "grid", "One base case with a WACC by growth grid"),
                      Option("probability-weighted event tree", "tree", "Discrete events (halving, contract wins, financing) with probabilities")],
                     evidence_keywords=FAMILIES["cycle"] + FAMILIES["valuation"], considerations=("what the debate ranges already give", "event risk", "PM usage"),
                     settles="whether value hinges on discrete events or on continuous drivers"),
    ]

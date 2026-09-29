"""LLM mapping proposals for what the rules could not map.

Flow: candidates (unmapped statement captions and unmapped or custom XBRL concepts above a materiality threshold, with
their latest values) -> the LLM proposes a schema target per candidate with a confidence and ranked alternatives ->
proposals are validated (source must be one we offered, target must be a schema key, 'residual' or 'ignore') -> written
to the ledger as *pending*. Nothing changes in the model until an analyst accepts, and an accepted link is reused on
every later run without calling the model again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .ledger import MappingLedger
from .debate import _JSON_RULES, _complete
from .llm import LLM
from .schema import ITEM_BY_KEY, STANDARD_ITEMS, MappingReport, norm_label

TARGETS = [it.key for it in STANDARD_ITEMS] + ["residual", "ignore"]


def _latest_values(df: pd.DataFrame, label: str) -> dict:
    from .mapping import _to_float, period_columns

    cols = period_columns(df)
    lab_col = next((c for c in df.columns if str(c).lower() == "label"), None)
    if lab_col is None:
        return {}
    rows = df[df[lab_col].astype(str) == label]
    if not len(rows):
        return {}
    r = rows.iloc[0]
    out = {}
    for c, y in sorted(cols.items(), key=lambda kv: kv[1])[-3:]:
        v = _to_float(r[c])
        if v is not None:
            out[str(y)] = v
    return out


def candidates(data_dir: Path, ticker: str, reports: dict[str, MappingReport], hist: pd.DataFrame, xbrl: pd.DataFrame | None = None,
               materiality: float = 0.01, ledger: MappingLedger | None = None) -> list[dict]:
    """Unmapped sources worth an analyst's attention: |latest value| >= materiality x (revenue for IS/CF, total assets for BS)."""
    d = Path(data_dir) / ticker.upper()
    last = int(hist.columns[-1])
    base = {"IS": float(hist.loc["revenue", last]) * 1e6, "CF": float(hist.loc["revenue", last]) * 1e6, "BS": float(hist.loc["total_assets", last]) * 1e6}
    decided = set()
    if ledger is not None:
        for st in ("IS", "BS", "CF"):
            for src_type in ("label", "concept"):
                decided |= {(st, s) for s in ledger.analyst_decided(ticker, st, src_type)}
                decided |= {(st, s) for s in ledger.rows(ticker, st, src_type, "pending").source}
    out: list[dict] = []
    for st, name in (("IS", "annual_is.csv"), ("BS", "annual_bs.csv"), ("CF", "annual_cf.csv")):
        p = d / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for label in reports[st].unmapped:
            if (st, label) in decided:
                continue
            vals = _latest_values(df, label)
            if not vals:
                continue
            latest = list(vals.values())[-1]
            if abs(latest) < materiality * base[st]:
                continue
            out.append({"statement": st, "source_type": "label", "source": label, "values": vals, "share_of_base": round(abs(latest) / base[st], 3)})
    if xbrl is not None and len(xbrl):
        known = {f"us-gaap:{c}".lower() for it in STANDARD_ITEMS for c in it.concepts + it.sum_concepts + it.neg_concepts}
        x = xbrl[(xbrl.fiscal_year == xbrl.fiscal_year.max())].copy()
        st_map = {"IncomeStatement": "IS", "BalanceSheet": "BS", "CashFlowStatement": "CF"}
        x["st"] = x.get("statement_type", "").map(lambda v: next((s for k, s in st_map.items() if k.lower() in str(v).lower()), None))
        x = x[x.st.notna() & ~x.concept.str.lower().isin(known) & ~x.concept.str.startswith("dei:")]
        for (concept, st), g in x.groupby(["concept", "st"]):
            if (st, concept) in decided:
                continue
            g = g.sort_values("period_end")
            latest = float(g.numeric_value.iloc[-1])
            if abs(latest) < materiality * base[st]:
                continue
            vals = {str(pd.Timestamp(pe).year): float(v) for pe, v in zip(g.period_end.tail(3), g.numeric_value.tail(3))}
            out.append({"statement": st, "source_type": "concept", "source": concept, "values": vals, "share_of_base": round(abs(latest) / base[st], 3),
                        "label": str(g.get("label", pd.Series([""])).iloc[-1])})
    # drop candidates whose latest value duplicates an already-mapped line (the same line under another caption or tag)
    mapped_vals = {st: set() for st in ("IS", "BS", "CF")}
    for key in hist.index:
        it = ITEM_BY_KEY.get(key)
        if it is None:
            continue
        v = float(hist.loc[key, last]) * 1e6
        if v:
            mapped_vals[it.statement].add(round(v, -3))
    out = [c for c in out if round(list(c["values"].values())[-1], -3) not in mapped_vals[c["statement"]]]
    # a caption and a tag with the same latest value are one line: keep the tag (more precise), drop the caption
    concept_vals = {(c["statement"], round(list(c["values"].values())[-1], -3)) for c in out if c["source_type"] == "concept"}
    out = [c for c in out if not (c["source_type"] == "label" and (c["statement"], round(list(c["values"].values())[-1], -3)) in concept_vals)]
    out.sort(key=lambda c: -c["share_of_base"])
    return out


_STOP = {"and", "of", "the", "in", "for", "to", "net", "loss", "income", "expense", "expenses", "other", "total", "current", "noncurrent",
         "gain", "gains", "losses", "increase", "decrease", "from", "on", "at", "per", "basic", "diluted", "before", "after", "value", "cost",
         "costs", "assets", "liabilities", "activities", "cash", "equivalents", "amount", "related", "us", "gaap", "mara", "riot", "clsk", "cifr",
         "operations", "continuing", "domestic", "attributable", "including", "excluding", "portion", "stockholders", "common", "issued"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _tokens(source: str) -> list[str]:
    """Distinctive words of a caption or tag ('mara:PurchasedEnergyCosts' -> ['purchased', 'energy'])."""
    txt = source.split(":", 1)[-1]
    txt = re.sub(r"([a-z])([A-Z])", r"\1 \2", txt)
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", txt)]
    return [w for w in words if len(w) > 3 and w not in _STOP]


def context_sentences(sections: dict[str, str], source: str, label: str = "", max_sents: int = 3) -> list[dict]:
    """Sentences from Items 1, 7 and 8 that share the candidate's distinctive words: the material the model may cite.
    Chosen deterministically, so a cited sentence is verbatim filing text by construction."""
    toks = set(_tokens(source) + _tokens(label))
    if not toks:
        return []
    hits = []
    for section, text in sections.items():
        if not text or section.lower().startswith("competition"):
            continue
        for sent in _SENT_SPLIT.split(re.sub(r"\s+", " ", text)):
            sent = sent.strip()
            if not (40 <= len(sent) <= 400):
                continue
            low = sent.lower()
            score = sum(1 for t in toks if t in low)
            if score >= max(1, min(2, len(toks))) or (score >= 1 and len(toks) == 1):
                bonus = 1 if any(w in low for w in ("consist", "include", "comprise", "represent", "recognize", "recorded")) else 0
                bonus += 2 if section.startswith("Item 8") else (1 if section.startswith(("Item 7", "Item 1 ")) else 0)  # policies over risk boilerplate
                hits.append((score + bonus, section, sent))
    hits.sort(key=lambda h: -h[0])
    out, seen = [], set()
    for sc, section, sent in hits:
        key = sent.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append({"section": section, "sentence": sent})
        if len(out) >= max_sents:
            break
    return out


_SYSTEM = """ROLE: mapper
You map a company's reported financial statement lines onto a fixed model schema. For each candidate line (a caption from the
standardized statements, or an XBRL concept, custom company concepts included) propose the schema target it belongs to.
Targets: one of the schema keys below FOR THE SAME STATEMENT as the candidate (an IS caption maps to an IS key), or "residual" (leave it in the statement's residual line, e.g. amortization of intangibles
inside other operating expense), or "ignore" (not a statement line, e.g. a per-share figure or a memo).
Give a confidence from 0 to 1 and up to two alternatives with their own confidences. Never invent sources: use the candidates exactly
as given.
Every proposal MUST include all of the following fields (a proposal missing "relation" is rejected):
- "relation": one sentence on what this line is for this company and what drives it (e.g. 'electricity bought for owned mining sites; scales with hashrate and power price'); write 'unclear from the filing' if you cannot tell;
- "citation_ids": the ids of the filing sentences (S-ids offered with the candidate) that support the relation; cite every offered sentence that is relevant; empty only if none fit. Never cite an id that was not offered;
- "questions": up to two questions an analyst must settle to categorize the line (e.g. 'Does cost of revenue exclude depreciation?');
- "duplicates": other candidate indices or schema keys this line would double count if mapped as proposed, each with a short reason (e.g. 'index 0 is a total that already contains this').
Return one JSON object and nothing else, with single quotes inside strings."""


def _schema_text() -> str:
    return "\n".join(f"- {it.key} ({it.statement}): {it.label}" for it in STANDARD_ITEMS if it.in_model)


def suggest(cands: list[dict], llm: LLM, company: str, seed: int = 0, batch: int = 15) -> list[dict]:
    """Ask the model in batches of ``batch`` candidates; keep only proposals whose source is one we offered and whose
    targets belong to the candidate's statement."""
    out: list[dict] = []
    for start in range(0, len(cands), batch):
        out += _suggest_batch(cands[start:start + batch], llm, company, seed + start)
    return out


def _suggest_batch(cands: list[dict], llm: LLM, company: str, seed: int = 0) -> list[dict]:
    if not cands:
        return []
    listing = "\n".join(f"[{i}] {c['statement']} {c['source_type']}: {c['source']}" + (f" (label: {c['label']})" if c.get("label") else "")
                        + f" | values {c['values']} | {c['share_of_base']:.1%} of {'total assets' if c['statement'] == 'BS' else 'revenue'}"
                        for i, c in enumerate(cands))
    user = (f"COMPANY: {company}\n\nSCHEMA TARGETS:\n{_schema_text()}\n\nCANDIDATES:\n{listing}\n\n"
            'Output JSON: {"proposals": [{"index": <candidate index>, "target": "<key|residual|ignore>", "confidence": <0-1>, '
            '"alternatives": [{"target": "...", "confidence": <0-1>}], "rationale": "<under 30 words>"}]}')
    raw = llm.complete_json(_SYSTEM, user, seed=seed, temperature=0.0)
    out = []
    for p in raw.get("proposals", []) if isinstance(raw, dict) else []:
        try:
            i = int(p.get("index"))
            c = cands[i]
        except (TypeError, ValueError, IndexError):
            continue
        tgt = str(p.get("target", "")).strip()
        allowed = {it.key for it in STANDARD_ITEMS if it.statement == c["statement"]} | {"residual", "ignore"}
        if tgt not in allowed:  # a target from another statement is not a mapping, it is a category error
            continue
        try:
            conf = min(max(float(p.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        alts = []
        for a in p.get("alternatives", []) or []:
            if isinstance(a, dict) and str(a.get("target", "")) in allowed and str(a.get("target")) != tgt:
                try:
                    alts.append({"target": str(a["target"]), "confidence": min(max(float(a.get("confidence", 0.3)), 0.0), 1.0)})
                except (TypeError, ValueError):
                    pass
        offered = {f"S{i}-{j}": cs for j, cs in enumerate(c.get("context", []))}
        cites = [offered[x] for x in (p.get("citation_ids") or []) if isinstance(x, str) and x in offered]
        qs = [str(q)[:200] for q in (p.get("questions") or []) if str(q).strip()][:2]
        dups = []
        for dd in p.get("duplicates") or []:
            if isinstance(dd, dict) and str(dd.get("ref", "")).strip():
                ref = str(dd["ref"]).strip()
                if ref.isdigit() and int(ref) < len(cands):
                    ref = cands[int(ref)]["source"]
                dups.append(f"{ref}: {str(dd.get('reason', ''))[:120]}")
            elif isinstance(dd, str) and dd.strip():
                dups.append(dd.strip()[:160])
        relation = str(p.get("relation") or "").strip()
        if not relation and p.get("rationale"):
            relation = f"(from rationale) {str(p['rationale'])[:240]}"
        auto = False
        if not cites and c.get("context"):
            cites, auto = [c["context"][0]], True  # keyword match the tool found; the model did not confirm it
        out.append({**c, "target": tgt, "confidence": conf, "alternatives": alts[:2], "rationale": str(p.get("rationale", ""))[:200],
                    "relation": relation[:300], "citations": cites, "citation_auto": auto, "questions": qs, "duplicates": dups[:4]})
    return out


def value_duplicates(cands: list[dict], hist: pd.DataFrame) -> dict[str, list[str]]:
    """Candidates and mapped lines that carry the same latest value (rounded to thousands) as another: near-certain double counts."""
    last = int(hist.columns[-1])
    by_val: dict[tuple, list[str]] = {}
    for c in cands:
        v = round(list(c["values"].values())[-1], -3)
        by_val.setdefault((c["statement"], v), []).append(c["source"])
    for key in hist.index:
        it = ITEM_BY_KEY.get(key)
        if it is not None and float(hist.loc[key, last]):
            by_val.setdefault((it.statement, round(float(hist.loc[key, last]) * 1e6, -3)), []).append(f"mapped line {key}")
    out: dict[str, list[str]] = {}
    for c in cands:
        v = round(list(c["values"].values())[-1], -3)
        others = [x for x in by_val.get((c["statement"], v), []) if x != c["source"]]
        if others:
            out[c["source"]] = [f"{x}: same value" for x in others]
    return out


def merge_runs(runs: list[list[dict]]) -> tuple[list[dict], dict]:
    """Combine repeated proposal runs per source: agreeing runs average their confidence; disagreeing runs keep the
    higher-confidence target, carry the other as an alternative, and the disagreement is written into the evidence."""
    by_src: dict[tuple, list[dict]] = {}
    for run in runs:
        for p in run:
            by_src.setdefault((p["statement"], p["source"]), []).append(p)
    merged, agree, total = [], 0, 0
    for key, ps in by_src.items():
        total += 1
        targets = {p["target"] for p in ps}
        best = max(ps, key=lambda p: p["confidence"])
        m = dict(best)
        if len(targets) == 1:
            agree += 1
            m["confidence"] = round(sum(p["confidence"] for p in ps) / len(ps), 2)
            m["agreement"] = f"{len(ps)}/{len(ps)} runs agree"
        else:
            alts = {a["target"]: a["confidence"] for a in best.get("alternatives", [])}
            for p in ps:
                if p["target"] != best["target"]:
                    alts[p["target"]] = max(alts.get(p["target"], 0), p["confidence"])
            m["alternatives"] = [{"target": t, "confidence": c} for t, c in sorted(alts.items(), key=lambda kv: -kv[1])][:3]
            m["confidence"] = round(best["confidence"] * (sum(1 for p in ps if p["target"] == best["target"]) / len(ps)), 2)
            m["agreement"] = f"runs disagree: {', '.join(sorted(targets))}"
        for f in ("questions", "duplicates"):
            seen, acc = set(), []
            for p in ps:
                for x in p.get(f, []):
                    if x not in seen:
                        seen.add(x)
                        acc.append(x)
            m[f] = acc[:4]
        if not m.get("relation") or m["relation"].startswith("(from rationale)"):
            better = next((p["relation"] for p in ps if p.get("relation") and not p["relation"].startswith("(from rationale)")), None)
            if better:
                m["relation"] = better
        confirmed = [p for p in ps if p.get("citations") and not p.get("citation_auto")]
        if confirmed:
            seen, acc = set(), []
            for p in confirmed:
                for cs in p["citations"]:
                    if cs["sentence"] not in seen:
                        seen.add(cs["sentence"])
                        acc.append(cs)
            m["citations"], m["citation_auto"] = acc[:3], False
        merged.append(m)
    return merged, {"sources": total, "agree": agree, "agreement_rate": round(agree / total, 2) if total else None, "runs": len(runs)}


_CITER = """ROLE: citer
You confirm citations for proposed financial-statement mappings. For each proposal (a reported line, the schema target it is
proposed to map to, and the reason) you are offered numbered filing sentences. Pick the sentence ids that show what the
line is or where the company records it, so that the mapping can be checked against the filing. A sentence that merely
contains the same words does not count. Give a confidence from 0 to 1 that the mapping is right given the sentences, and
a "why" under 20 words. Return an empty list when no offered sentence supports the mapping.
""" + _JSON_RULES + """
Output JSON: {"citations": {"P1": {"citation_ids": ["P1-S2"], "confidence": 0.8, "why": "..."}, "P2": {"citation_ids": [], "confidence": 0.2, "why": "..."}}}"""


def confirm_citations(props: list[dict], sections: dict[str, str], llm: LLM, batch: int = 8, max_sents: int = 8, seed: int = 0) -> dict:
    """Second pass for proposals whose citation the model did not choose: offer a wider set of sentences (the line's own
    words plus the target's) and let the model pick or decline. Marks citation_model_conf and citation_checked."""
    todo = [p for p in props if p.get("citation_auto") or not p.get("citations")]
    stats: list[dict] = []
    confirmed = declined = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        offered: dict[str, dict] = {}
        lines = []
        for i, p in enumerate(chunk, start=1):
            tgt = p.get("target", "")
            tlabel = ITEM_BY_KEY[tgt].label if tgt in ITEM_BY_KEY else tgt
            sents = context_sentences(sections, p["source"], p.get("label", ""), max_sents=max_sents)
            extra = context_sentences(sections, tlabel, "", max_sents=3) if tlabel and tlabel != tgt else []
            seen = {c["sentence"] for c in sents}
            sents += [c for c in extra if c["sentence"] not in seen]
            if not sents:
                p["citation_checked"], p["citations"], p["citation_auto"] = f"no filing sentence shares words with this line", [], False
                declined += 1
                continue
            lines.append(f"P{i}: line \"{p['source']}\" ({p['statement']}) proposed -> {tgt} ({tlabel}); reason: {p.get('relation', '')[:160]}")
            for j, c in enumerate(sents, start=1):
                sid = f"P{i}-S{j}"
                offered[sid] = c
                lines.append(f"  [{sid}] ({c['section']}) \"{c['sentence'][:300]}\"")
        if not lines:
            continue
        raw = _complete(llm, _CITER, "PROPOSALS AND OFFERED SENTENCES:\n" + "\n".join(lines) + "\n\nPick the supporting sentences.", seed + start, 0.1, stats, "citer")
        picks = raw.get("citations") or {}
        for i, p in enumerate(chunk, start=1):
            if p.get("citation_checked"):
                continue
            pick = picks.get(f"P{i}") or {}
            ids = [x for x in (pick.get("citation_ids") or []) if isinstance(x, str) and x in offered and x.startswith(f"P{i}-")]
            try:
                conf = min(max(float(pick.get("confidence", 0.0)), 0.0), 1.0)
            except (TypeError, ValueError):
                conf = 0.0
            n_off = sum(1 for k in offered if k.startswith(f"P{i}-"))
            if ids:
                p["citations"], p["citation_auto"], p["citation_model_conf"] = [offered[x] for x in ids][:3], False, conf
                p["citation_checked"] = f"model-confirmed ({conf:.2f}): {str(pick.get('why', ''))[:80]}"
                confirmed += 1
            else:
                p["citations"], p["citation_auto"] = [], False
                p["citation_checked"] = f"no supporting sentence found by the model among {n_off} candidates" + (f"; {str(pick.get('why', ''))[:80]}" if pick.get("why") else "")
                declined += 1
    return {"checked": len(todo), "confirmed": confirmed, "declined": declined, "calls": len(stats)}


def write_pending(ledger: MappingLedger, ticker: str, proposals: list[dict], model_name: str, accession: str = "", val_dups: dict | None = None) -> int:
    n = 0
    for p in proposals:
        dups = list(p.get("duplicates", [])) + list((val_dups or {}).get(p["source"], []))
        cite = " || ".join(f"{accession or '10-K'} | {c['section']} | \"{c['sentence']}\"" for c in p.get("citations", []))
        if cite and p.get("citation_auto"):
            cite = "auto (keyword match, not confirmed by the model): " + cite
        elif p.get("citation_checked"):
            cite = f"{p['citation_checked']}" + (f": {cite}" if cite else "")
        elif cite:
            cite = "model-selected: " + cite
        ev = f"values {p['values']}; {p['share_of_base']:.1%} of base; {p['rationale']}" + (f"; {p['agreement']}" if p.get("agreement") else "")
        ledger.upsert(ticker, p["statement"], p["source_type"], p["source"], p["target"], p["confidence"], p["alternatives"], "pending",
                      f"llm:{model_name}", evidence=ev, relation=p.get("relation", ""), citation=cite, questions=" | ".join(p.get("questions", [])),
                      duplicates=" | ".join(dict.fromkeys(dups)))
        n += 1
    return n


class MockMapper:
    """Keyword heuristic standing in for the model in tests and offline runs."""

    name = "mock"
    RULES = [(r"cost|depreciation and amortization|energy", "cogs"), (r"digital|crypto|bitcoin", "residual"), (r"per share|weighted", "ignore"),
             (r"receivable", "receivables"), (r"debt|notes payable", "long_term_debt"), (r"lease", "residual")]

    def complete_json(self, system: str, user: str, seed=None, temperature=0.0) -> dict:
        props = []
        for m in re.finditer(r"\[(\d+)\] (IS|BS|CF) (label|concept): ([^|]+)\|", user):
            i, src = int(m.group(1)), m.group(4).lower()
            tgt, conf = "residual", 0.4
            for pat, t in self.RULES:
                if re.search(pat, src):
                    tgt, conf = t, 0.7
                    break
            props.append({"index": i, "target": tgt, "confidence": conf, "alternatives": [{"target": "ignore", "confidence": 0.2}], "rationale": "mock",
                          "relation": "mock relation", "citation_ids": [f"S{i}-0"], "questions": ["mock: does this line include depreciation?"],
                          "duplicates": [{"ref": "0", "reason": "mock overlap"}] if i else []})
        return {"proposals": props}

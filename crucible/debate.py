"""Structured disagreement over one assumption, with the analyst as the judge.

Protocol (2 rounds by default):
  1. Bull and bear each propose a value with evidence-cited, quote-verified claims, from role-specific briefs.
  2. Each sees the other's surviving claims and rebuts.
  3. A synthesis pass maps the crux (where and why they still disagree), the questions that would settle it, and a
     suggested range. The range is advisory: the analyst accepts, edits or rejects it, and that decision is logged.

Guardrails against the known failure modes of LLM debate:
  * frozen evidence packet: agents cannot introduce facts, only cite ids;
  * verified-quote gate (Khan et al., ICML 2024; Irving, Christiano & Amodei 2018): every claim carries a verbatim
    span that must string-match a cited evidence item, or the claim is dropped before anyone sees it;
  * numbers in a claim must exist in the packet; a citation that omits the item holding a number is completed, not trusted;
  * heterogeneity: each side gets a different evidence brief (growth drivers vs risk and cost lines) and can run on a
    different model family;
  * an explicit concession policy (Smit et al. 2024: agreement intensity is the hyperparameter that matters most);
  * the synthesis is told not to split the difference and to show the arithmetic its range implies.
"""
from __future__ import annotations

import difflib
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .evidence import EvidenceItem, EvidencePacket
from .llm import LLM

PROMPT_VERSION = "0.5"
Policy = Literal["stubborn", "calibrated", "agreeable"]

POLICY_TEXT = {
    "stubborn": "Do not concede any point unless the opposing evidence is decisive. Hold your proposed value unless forced.",
    "calibrated": "Update your proposed value only when the opposing side cites stronger or more recent evidence. Do not concede for politeness. Do not harden for effect.",
    "agreeable": "Move toward the opposing side whenever their evidence is at least as strong as yours.",
}


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    quote: str = ""  # verbatim span from one cited evidence item


class Argument(BaseModel):
    role: str
    round: int
    proposed_value: float
    confidence: float = 0.5
    claims: list[Claim] = Field(default_factory=list)
    rebuttals: list[Claim] = Field(default_factory=list)
    questions_for_management: list[str] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)  # validator report
    completed_citations: list[str] = Field(default_factory=list)
    out_of_bounds: bool = False


class Crux(BaseModel):
    question: str
    bull_position: str = ""
    bear_position: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    what_would_settle_it: str = ""


class Verdict(BaseModel):
    low: float
    base: float
    high: float
    confidence: float = 0.5
    crux: list[Crux] = Field(default_factory=list)
    key_disagreements: list[str] = Field(default_factory=list)
    questions_for_management: list[str] = Field(default_factory=list)
    questions_for_internal_discussion: list[str] = Field(default_factory=list)
    rationale: str = ""
    implied_path: str = ""
    evidence_used: list[str] = Field(default_factory=list)


class DebateResult(BaseModel):
    company: str
    assumption: str
    packet_hash: str
    evidence_ids: list[str]
    model: str
    models: dict[str, str] = Field(default_factory=dict)
    seed: int | None
    temperature: float
    policy: Policy
    rounds: int
    prompt_version: str = PROMPT_VERSION
    arguments: list[Argument]
    verdict: Verdict
    call_stats: list[dict] = Field(default_factory=list)
    elapsed_s: float


# ----------------------------------------------------------------------------- validation

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    text = re.sub(r"\bE\d+\b", " ", text)                                  # evidence ids are not numbers
    text = re.sub(r"\b\d+-[KQ]s?\b", " ", text)                             # form names: 10-K, 10-Qs
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:-|\s)?(?:year|yr|month|quarter|week|day|x|pt|pp)s?\b", " ", text)  # 3-year, 2x, 8pt
    text = re.sub(r"\b(?:year|years|yr|y|fy|q|quarter)\s?\d(?:\s*(?:-|and|to|/)\s*\d)?\b", " ", text, flags=re.I)  # year 2, Y1, years 2-3, Q3
    text = re.sub(r"(?<=\d)\s*(?:-|to|–)\s*(?=\d)", " ", text)              # ranges: 12-18 -> 12 18
    for m in _NUM.finditer(text):
        s = m.group(0).replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if 1990 <= v <= 2100 and float(v).is_integer():                     # years are not claims
            continue
        tail = text[m.end():m.end() + 2]
        head = text[max(0, m.start() - 1):m.start()]
        if v.is_integer() and abs(v) <= 12 and "%" not in tail and "$" not in head and "." not in s:
            continue                                                         # counts ("3 years", "2 peers"), not figures
        out.append(v)
    return out


def _norm(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[•\u2022]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_verified(quote: str, items: list[EvidenceItem], min_ratio: float = 0.92) -> str | None:
    """Return the id of the cited item that contains the quote (verbatim after normalization, or near-verbatim), else None."""
    q = _norm(quote).strip('"\' ')
    if len(q) < 8:
        return None
    for it in items:
        c = _norm(it.content)
        if q in c:
            return it.id
    # tolerate punctuation slips: best window of the same length in each item
    for it in items:
        c = _norm(it.content)
        if len(c) < len(q):
            continue
        best = 0.0
        step = max(1, len(q) // 4)
        for start in range(0, len(c) - len(q) + 1, step):
            r = difflib.SequenceMatcher(None, q, c[start:start + len(q)]).ratio()
            if r > best:
                best = r
                if best >= min_ratio:
                    return it.id
    return None


def validate_claims(claims: list[Claim], packet: EvidencePacket, tol: float = 0.02, require_quote: bool = True) -> tuple[list[Claim], list[str], list[str]]:
    """Hard gate. Keep a claim only if: it cites real ids; its quote string-matches a cited item; every number in it
    exists in the packet (citations are completed with the items holding the numbers). Returns (kept, dropped, completed)."""
    ids = set(packet.ids())
    kept, dropped, completed = [], [], []
    all_pool = {i.id: _numbers(i.content) for i in packet.items}
    for c in claims:
        cited = [i for i in c.evidence_ids if i in ids]
        if not cited:
            dropped.append(f"uncited or unknown ids: {c.text[:80]}")
            continue
        if require_quote:
            src = quote_verified(c.quote, [packet.get(i) for i in cited])
            if src is None:
                dropped.append(f"quote not found in cited evidence {cited}: {c.text[:80]}")
                continue
        pool = [n for i in cited for n in all_pool[i]]
        pool += [abs(a - b) for a in pool for b in pool if a != b]  # differences of cited numbers are legitimate arithmetic
        extra_ids: list[str] = []
        bad = None
        for v in _numbers(c.text):
            cands = (v, v * 100.0, v / 100.0)
            if any(abs(cv - p) <= tol * max(1.0, abs(p)) for p in pool for cv in cands):
                continue
            holder = next((i for i, nums in all_pool.items() if i not in cited and any(abs(cv - p) <= tol * max(1.0, abs(p)) for p in nums for cv in cands)), None)
            if holder:
                extra_ids.append(holder)
                pool += all_pool[holder]
                continue
            bad = v
            break
        if bad is not None:
            dropped.append(f"number {bad} not in the packet: {c.text[:80]}")
            continue
        if extra_ids:
            completed.append(f"{c.text[:60]} -> added {extra_ids}")
        kept.append(Claim(text=c.text, evidence_ids=cited + extra_ids, quote=c.quote))
    return kept, dropped, completed


# ----------------------------------------------------------------------------- heterogeneity: role briefs

_BULL_HINTS = ("guidance", "backlog", "orders", "book-to-bill", "expect", "growth", "capacity", "demand", "pipeline", "segment", "acquisition")
_BEAR_HINTS = ("risk", "may not", "could", "cost", "margin", "supplier", "constraint", "cancel", "decline", "decelerat", "impairment", "dilut", "debt", "capex", "competition")


def role_brief(packet: EvidencePacket, role: str) -> list[str]:
    """Which evidence ids each side should start from. Both sides see the whole packet; the brief creates heterogeneity."""
    ids = []
    for it in packet.items:
        c = it.content.lower()
        if role == "bull":
            forward = it.kind in ("filing_text", "derived_metric") and (it.source.startswith("8-K") or it.source.startswith("derived: guidance") or any(h in c for h in ("backlog", "orders", "guidance")))
            trend = it.kind == "historical_metric"
            hit = forward or trend or (it.kind == "filing_text" and any(h in c for h in _BULL_HINTS[4:]))
        else:
            hit = (it.kind == "peer_metric") or ("risk" in it.source.lower()) or any(h in c for h in _BEAR_HINTS) or (it.kind == "derived_metric" and "cost" in c)
        if hit:
            ids.append(it.id)
    return ids


# ----------------------------------------------------------------------------- prompts

_JSON_RULES = ("Return one JSON object and nothing else. Inside JSON strings use single quotes for any inner quotation, never unescaped double quotes.")


def _advocate_system(role: str, packet: EvidencePacket, policy: Policy) -> str:
    spec = packet.assumption
    direction = spec.bull_direction if role == "bull" else ("down" if spec.bull_direction == "up" else "up")
    brief = role_brief(packet, role)
    return f"""ROLE: {role}
You are the {role} advocate in a structured debate over one financial-model assumption for {packet.company}.
Assumption: {spec.key} = {spec.description}. Horizon: {spec.horizon}.
Your mandate: argue for the {direction}-side value that the evidence can support.
Start from your brief (items most relevant to your side): {', '.join(brief) if brief else 'none flagged; use the whole packet'}. You may cite any item in the packet.
Rules:
- Use only the numbered evidence items provided. Every claim must cite evidence ids and carry a "quote": a verbatim span (5 to 25 words) copied exactly from ONE cited item. Claims whose quote does not match are deleted before anyone sees them.
- Never introduce numbers that are not in the packet.
- Evidence hierarchy for a forward-looking assumption: (1) the company's own forward indicators (guidance, orders, backlog, book-to-bill) outrank (2) the company's recent trajectory, which outranks (3) peer history, which outranks (4) generic risk-factor language that appears in every 10-K.
- If the packet contains guidance or backlog that implies a year-one figure, reconcile your proposed value with it: state what your number implies for the remaining years.
- Plausible range for this assumption: {spec.lower_bound} to {spec.upper_bound} (decimal).
- {POLICY_TEXT[policy]}
- Be selective: at most 6 claims and 4 rebuttals, each under 40 words, and at most 3 questions for management.
- State your confidence (0 to 1) that the true value is on your side of the opposing proposal.
{_JSON_RULES}
Output JSON: {{"proposed_value": <decimal>, "confidence": <0-1>, "claims": [{{"text": "...", "evidence_ids": ["E1"], "quote": "..."}}], "rebuttals": [{{"text": "...", "evidence_ids": ["E2"], "quote": "..."}}], "questions_for_management": ["..."]}}"""


def _judge_system(packet: EvidencePacket) -> str:
    spec = packet.assumption
    return f"""ROLE: judge
You are the synthesis pass in a structured debate over one financial-model assumption for {packet.company}. The analyst, not you, makes the final call:
your job is to map where the two sides still disagree and why, what would settle it, and to suggest a defensible range.
Assumption: {spec.key} = {spec.description}. Horizon: {spec.horizon}.
Weigh the surviving claims by evidence quality, in this order: (1) the company's own forward indicators (guidance, orders, backlog, book-to-bill),
(2) the company's recent trajectory, (3) peer history, (4) generic risk-factor language that appears in every 10-K. A peer base rate never overrides
a company-specific forward indicator unless a claim shows why that indicator is unreliable.
Arithmetic discipline: if any evidence implies a year-one figure (guidance, backlog conversion), compute what your base implies for the remaining years
of the horizon and put it in "implied_path"; if that path is not defensible, move the base.
Do not split the difference by default; the base must follow the stronger evidence and say why.
Range width: the packet states the standard deviation of the company's own annual growth over all years available. Unless you argue
why the coming years will be calmer, your high minus low should be at least twice that number, centered on the base. Ranges narrower
than the company's own history of surprises have been wrong in every backtest.
Crux: list 2 to 4 cruxes. Each names the question, each side's position in one sentence, the evidence ids in play, and what would settle it.
Questions: management questions must be specific and answerable (numbers, dates, shares of backlog); internal questions are for the team's own work
(what to model, what to check, what data to pull).
Confidence: 0.3 or below when primary sources conflict or are absent, 0.5 when evidence is mixed, 0.7 or above only when the company's own forward
indicators and its trajectory agree. Do not default to a middle value. Keep every text field under 60 words.
{_JSON_RULES}
Output JSON: {{"low": <decimal>, "base": <decimal>, "high": <decimal>, "confidence": <0-1>, "implied_path": "...",
"crux": [{{"question": "...", "bull_position": "...", "bear_position": "...", "evidence_ids": ["E1"], "what_would_settle_it": "..."}}],
"questions_for_management": ["..."], "questions_for_internal_discussion": ["..."], "rationale": "...", "evidence_used": ["E1"]}}"""


def _render_argument(a: Argument) -> str:
    lines = [f"{a.role.upper()} (round {a.round}) proposed_value {a.proposed_value:.4f} | confidence {a.confidence:.2f}"]
    for c in a.claims:
        lines.append(f"- claim [{','.join(c.evidence_ids)}]: {c.text}" + (f'  |  quote: "{c.quote}"' if c.quote else ""))
    for c in a.rebuttals:
        lines.append(f"- rebuttal [{','.join(c.evidence_ids)}]: {c.text}" + (f'  |  quote: "{c.quote}"' if c.quote else ""))
    return "\n".join(lines)


def _as_claims(raw_list) -> list[Claim]:
    out = []
    for c in raw_list or []:
        if not isinstance(c, dict):
            continue
        text = c.get("text") or c.get("claim") or c.get("argument")
        if not text:
            continue
        ids = c.get("evidence_ids") or c.get("evidence") or c.get("ids") or []
        if isinstance(ids, str):
            ids = re.findall(r"E\d+", ids)
        out.append(Claim(text=str(text), evidence_ids=[str(i).strip() for i in ids], quote=str(c.get("quote") or "")))
    return out


def _parse_argument(raw: dict, role: str, rnd: int, packet: EvidencePacket, require_quote: bool) -> Argument:
    try:
        pv = float(raw.get("proposed_value"))
    except (TypeError, ValueError):
        pv = float("nan")
    try:
        conf = min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    claims = _as_claims(raw.get("claims") or raw.get("arguments") or raw.get("points"))
    rebuttals = _as_claims(raw.get("rebuttals"))
    kept_c, drop_c, comp_c = validate_claims(claims, packet, require_quote=require_quote)
    kept_r, drop_r, comp_r = validate_claims(rebuttals, packet, require_quote=require_quote)
    spec = packet.assumption
    oob = not (spec.lower_bound <= pv <= spec.upper_bound) if pv == pv else True
    return Argument(role=role, round=rnd, proposed_value=pv, confidence=conf, claims=kept_c, rebuttals=kept_r,
                    questions_for_management=[str(q) for q in raw.get("questions_for_management", [])][:5],
                    dropped_claims=drop_c + drop_r, completed_citations=comp_c + comp_r, out_of_bounds=oob)


def _complete(llm: LLM, system: str, user: str, seed: int | None, temperature: float, stats: list[dict], tag: str) -> dict:
    """One retry on malformed JSON, with an explicit reminder; then fail loudly. Records call statistics."""
    t0 = time.time()
    try:
        raw = llm.complete_json(system, user, seed=seed, temperature=temperature)
    except ValueError:
        try:
            raw = llm.complete_json(system, user + "\n\nYour previous answer was not valid JSON. Return exactly one JSON object and nothing else, with single quotes inside strings.",
                                    seed=(None if seed is None else seed + 1000), temperature=temperature)
            tag += " (retry)"
        except ValueError as e2:
            raise ValueError(f"model returned unparsable JSON twice: {e2}") from e2
    meta = dict(getattr(llm, "last_meta", {}) or {})
    stats.append({"call": tag, "model": getattr(llm, "name", "?"), "seconds": round(time.time() - t0, 1), **meta})
    return raw


def run_debate(packet: EvidencePacket, llm: LLM, rounds: int = 2, seed: int | None = None, temperature: float = 0.7,
               policy: Policy = "calibrated", evidence_order: list[str] | None = None, role_llms: dict[str, LLM] | None = None,
               require_quote: bool = True) -> DebateResult:
    """``role_llms`` maps 'bull' / 'bear' / 'judge' to different LLM clients for model-family heterogeneity."""
    t0 = time.time()
    role_llms = role_llms or {}
    L = {r: role_llms.get(r, llm) for r in ("bull", "bear", "judge")}
    ev_text = packet.render(evidence_order)
    args: list[Argument] = []
    latest: dict[str, Argument] = {}
    stats: list[dict] = []
    for rnd in range(1, rounds + 1):
        for role in ("bull", "bear"):
            other = latest.get("bear" if role == "bull" else "bull")
            user = f"EVIDENCE PACKET (as of {packet.as_of or 'latest filing'}):\n{ev_text}\n"
            if rnd == 1:
                user += "\nRound 1: state your proposed value and your strongest evidence-cited, quote-verified claims."
            else:
                user += f"\nOpposing side's surviving claims:\n{_render_argument(other) if other else '(none)'}\n"
                user += f"\nYour previous position:\n{_render_argument(latest[role])}\n\nRound {rnd}: rebut, then restate your proposed value (updated only if warranted by the evidence)."
            raw = _complete(L[role], _advocate_system(role, packet, policy), user, (None if seed is None else seed * 10 + rnd), temperature, stats, f"{role} r{rnd}")
            a = _parse_argument(raw, role, rnd, packet, require_quote)
            if not a.claims and not a.rebuttals:
                hint = ("\nYour previous answer had no claim that survived validation (each claim needs real evidence ids and a verbatim quote copied "
                        "from one cited item). Try again and copy quotes exactly.")
                raw = _complete(L[role], _advocate_system(role, packet, policy), user + hint, (None if seed is None else seed * 10 + rnd + 100), temperature, stats, f"{role} r{rnd} (revalidate)")
                a = _parse_argument(raw, role, rnd, packet, require_quote)
            args.append(a)
            latest[role] = a
    judge_user = (f"EVIDENCE PACKET:\n{ev_text}\n\nFINAL POSITIONS:\n{_render_argument(latest['bull'])}\n\n{_render_argument(latest['bear'])}\n"
                  f"\nValidator dropped {sum(len(a.dropped_claims) for a in args)} claims (uncited, unverified quote, or numbers not in the packet) before you saw this.")
    raw = _complete(L["judge"], _judge_system(packet), judge_user, (None if seed is None else seed * 10 + 9), temperature, stats, "judge")
    try:
        crux = [Crux(**{k: v for k, v in c.items() if k in Crux.model_fields}) for c in raw.get("crux", []) if isinstance(c, dict) and c.get("question")]
        verdict = Verdict(**{k: raw.get(k) for k in Verdict.model_fields if k in raw and k != "crux"}, crux=crux)
    except ValidationError as e:
        raise ValueError(f"judge returned malformed JSON: {e}") from e
    lo, hi = min(verdict.low, verdict.high), max(verdict.low, verdict.high)
    verdict.low, verdict.high = lo, hi
    verdict.base = min(max(verdict.base, lo), hi)
    if not verdict.key_disagreements:
        verdict.key_disagreements = [c.question for c in verdict.crux]
    verdict.questions_for_management = list(dict.fromkeys(
        [str(q) for q in verdict.questions_for_management] + [q for a in args for q in a.questions_for_management]))[:8]
    models = {r: getattr(L[r], "name", "?") for r in L}
    return DebateResult(company=packet.company, assumption=packet.assumption.key, packet_hash=packet.hash(),
                        evidence_ids=packet.ids() if evidence_order is None else evidence_order, model=getattr(llm, "name", "?"), models=models,
                        seed=seed, temperature=temperature, policy=policy, rounds=rounds, arguments=args, verdict=verdict,
                        call_stats=stats, elapsed_s=round(time.time() - t0, 2))


class Estimate(BaseModel):
    """One independent, non-adversarial estimate: the control arm for the harness."""
    low: float
    base: float
    high: float
    confidence: float = 0.5
    claims: list[Claim] = Field(default_factory=list)
    rationale: str = ""
    dropped_claims: list[str] = Field(default_factory=list)
    model: str = "?"
    seed: int | None = None
    call_stats: list[dict] = Field(default_factory=list)


def _analyst_system(packet: EvidencePacket) -> str:
    spec = packet.assumption
    return f"""ROLE: analyst
You are an independent analyst estimating one financial-model assumption for {packet.company}. There is no debate: give your own best estimate.
Assumption: {spec.key} = {spec.description}. Horizon: {spec.horizon}.
Rules:
- Use only the numbered evidence items provided. Every claim must cite evidence ids and carry a "quote": a verbatim span (5 to 25 words) copied exactly from ONE cited item.
- Never introduce numbers that are not in the packet.
- Weigh evidence in this order: the company's own forward indicators (guidance, orders, backlog), the company's recent trajectory, peer history, generic risk-factor language.
- If any evidence implies a year-one figure, state what your base implies for the remaining years.
- Plausible range: {spec.lower_bound} to {spec.upper_bound} (decimal). At most 6 claims, each under 40 words.
- Confidence: 0.3 or below when primary sources conflict or are absent, 0.5 when mixed, 0.7 or above only when forward indicators and trajectory agree.
{_JSON_RULES}
Output JSON: {{"low": <decimal>, "base": <decimal>, "high": <decimal>, "confidence": <0-1>, "claims": [{{"text": "...", "evidence_ids": ["E1"], "quote": "..."}}], "rationale": "..."}}"""


def estimate(packet: EvidencePacket, llm: LLM, seed: int | None = None, temperature: float = 0.7, evidence_order: list[str] | None = None,
             require_quote: bool = True) -> Estimate:
    stats: list[dict] = []
    user = f"EVIDENCE PACKET (as of {packet.as_of or 'latest filing'}):\n{packet.render(evidence_order)}\n\nGive your estimate."
    raw = _complete(llm, _analyst_system(packet), user, (None if seed is None else seed * 10 + 5), temperature, stats, "analyst")
    claims = _as_claims(raw.get("claims"))
    kept, dropped, _ = validate_claims(claims, packet, require_quote=require_quote)
    try:
        lo, base, hi = float(raw["low"]), float(raw["base"]), float(raw["high"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"analyst returned malformed estimate: {e}") from e
    lo, hi = min(lo, hi), max(lo, hi)
    base = min(max(base, lo), hi)
    try:
        conf = min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    return Estimate(low=lo, base=base, high=hi, confidence=conf, claims=kept, rationale=str(raw.get("rationale", "")), dropped_claims=dropped,
                    model=getattr(llm, "name", "?"), seed=seed, call_stats=stats)


def transcript(result: DebateResult) -> str:
    v = result.verdict
    lines = [f"# {result.company}: {result.assumption} (models {result.models}, seed {result.seed}, policy {result.policy}, prompt v{result.prompt_version})"]
    lines.append("\n## CRUX (where the sides still disagree)")
    for i, c in enumerate(v.crux, 1):
        lines.append(f"{i}. {c.question}\n   bull: {c.bull_position}\n   bear: {c.bear_position}\n   evidence: {', '.join(c.evidence_ids)}\n   settles it: {c.what_would_settle_it}")
    if not v.crux:
        lines.append("- " + "\n- ".join(v.key_disagreements))
    lines.append("\n## QUESTIONS FOR MANAGEMENT\n- " + "\n- ".join(v.questions_for_management))
    if v.questions_for_internal_discussion:
        lines.append("\n## QUESTIONS FOR INTERNAL DISCUSSION\n- " + "\n- ".join(v.questions_for_internal_discussion))
    lines.append(f"\n## SUGGESTED RANGE (advisory; the analyst decides)\nlow {v.low:.4f} | base {v.base:.4f} | high {v.high:.4f} | self-reported confidence {v.confidence:.2f}")
    if v.implied_path:
        lines.append("Implied path: " + v.implied_path)
    lines.append("Rationale: " + v.rationale)
    lines.append("\n## DEBATE PATH")
    for a in result.arguments:
        lines.append("\n" + _render_argument(a))
        if a.completed_citations:
            lines.append("  citations completed: " + " | ".join(a.completed_citations))
        if a.dropped_claims:
            lines.append("  validator dropped: " + " | ".join(a.dropped_claims))
    calls = result.call_stats
    if calls:
        trunc = sum(1 for c in calls if c.get("stop_reason") == "max_tokens")
        lines.append(f"\n({len(calls)} model calls, {sum(c.get('seconds', 0) for c in calls):.0f}s, {trunc} truncated)")
    return "\n".join(lines)


def review_markdown(result: DebateResult, stability: dict | None = None, attribution_rows: list[dict] | None = None, arms: dict | None = None,
                    calibration: dict | None = None) -> str:
    """Analyst review sheet: crux, questions, evidence map, suggested range, decision template."""
    v = result.verdict
    md = [f"# Review: {result.company} / {result.assumption}", "",
          f"Packet `{result.packet_hash}` | models {result.models} | policy {result.policy} | prompt v{result.prompt_version}", "",
          "## Crux", ""]
    for c in v.crux:
        md += [f"**{c.question}**", f"- Bull: {c.bull_position}", f"- Bear: {c.bear_position}", f"- Evidence: {', '.join(c.evidence_ids)}", f"- Settles it: {c.what_would_settle_it}", ""]
    md += ["## Questions for management", ""] + [f"- {q}" for q in v.questions_for_management] + [""]
    if v.questions_for_internal_discussion:
        md += ["## Questions for internal discussion", ""] + [f"- {q}" for q in v.questions_for_internal_discussion] + [""]
    md += ["## Suggested range (advisory)", "", f"| low | base | high | self-reported confidence |", "|---|---|---|---|",
           f"| {v.low:.1%} | {v.base:.1%} | {v.high:.1%} | {v.confidence:.2f} |", ""]
    if v.implied_path:
        md += [f"Implied path: {v.implied_path}", ""]
    if calibration and calibration.get("n_points"):
        lo, hi = v.base + calibration["error_min"], v.base + calibration["error_max"]
        md += [f"Empirical check from {calibration['n_points']} backtest(s) ({', '.join(calibration['as_of_dates'])}): realized minus base ranged "
               f"{calibration['error_min']:+.1%} to {calibration['error_max']:+.1%} (median {calibration['error_median']:+.1%}). Applied to this base: {lo:.1%} to {hi:.1%}. "
               f"Advisory range half-width {((v.high - v.low) / 2):.1%} vs past half-width {calibration['mean_debate_half_width']:.1%}.", ""]
    if stability:
        md += ["## Robustness (measured, not self-reported)", "", f"- Base across {stability.get('n')} runs: median {stability.get('base_median')}, min {stability.get('base_min')}, max {stability.get('base_max')}, agreement {stability.get('agreement_rate')}", ""]
    if arms:
        md += ["## Debate vs vote (control arm: independent estimates, no debate)", "",
               f"- Debate base {arms['debate_base_median']:.1%} vs vote base {arms['vote_base_median']:.1%} (delta {arms['delta_base']:+.1%}, noise floor {arms['noise_floor_abs']:.1%})",
               f"- Debate range width {arms['debate_width']:.1%} vs vote {arms['vote_width']:.1%}; evidence cited: debate {arms['debate_evidence_ids']}, vote {arms['vote_evidence_ids']}",
               f"- {arms['verdict']}", ""]
    if attribution_rows:
        md += ["## Evidence attribution (base without the group minus base with it)", "", "| group | delta base | delta width |", "|---|---|---|"]
        md += [f"| {r['evidence_id']} | {r['delta_base']:+.3f} | {r['delta_width']:+.3f} |" for r in attribution_rows] + [""]
    md += ["## Evidence map", ""]
    for a in result.arguments:
        for c in a.claims + a.rebuttals:
            md.append(f"- {a.role} r{a.round} [{', '.join(c.evidence_ids)}]: {c.text}")
    md += ["", "## Decision", "", "`crucible decide <TICKER> --assumption " + result.assumption + " --action accept|edit|reject --value <decimal> --reason \"...\"`", ""]
    return "\n".join(md)

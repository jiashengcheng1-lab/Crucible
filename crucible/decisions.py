"""Decisions: every "which scheme / which method / which state" question an analyst must settle, run the same way.

A decision has options. Each option gets an advocate who argues from the shared evidence packet with verified quotes.
Two options debate directly; more than two run a single-elimination tournament (pairwise, the winner advances) and the
final order is the ranking. The judge returns, per option, a confidence, the reasoning, the citations it relied on,
the questions that would settle the choice, and the considerations (trade-offs, conflicts with other decisions).

Every option becomes a row in the decision ledger, with the same columns as the mapping ledger so one Excel format
serves both: ticker | statement = decision key | source_type = option | source = option label | target = option value |
confidence | alternatives | status (pending, accepted, rejected, unsure) | proposed_by | evidence | relation = reasoning |
citation = verbatim filing text | questions | duplicates = considerations and conflicts | decided_by | decided_at | note | version.

Token discipline: one shared packet per company, one round per pairing, results cached by (packet hash, options, seed),
and `--cheap` skips the advocates and asks the judge to rank directly (one call instead of three per pairing).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from .debate import _JSON_RULES, Claim, _as_claims, _complete, validate_claims
from .evidence import EvidencePacket
from .ledger import LEDGER_COLUMNS, MappingLedger
from .llm import LLM

PROMPT_VERSION = "d0.1"


@dataclass
class Option:
    label: str
    value: str
    description: str


@dataclass
class DecisionSpec:
    key: str
    question: str
    options: list[Option]
    evidence_kinds: tuple[str, ...] = ()          # packet item kinds to keep (empty = all)
    evidence_keywords: tuple[str, ...] = ()       # keep filing_text items that mention any of these (empty = all)
    considerations: tuple[str, ...] = ()          # categories the judge must address
    settles: str = ""                             # what kind of information settles it
    notes: str = ""
    tournament: bool = False                      # always pairwise (winner advances), even with --cheap


class OptionVerdict(BaseModel):
    label: str
    confidence: float = 0.5
    reasoning: str = ""
    citations: list[str] = Field(default_factory=list)   # evidence ids
    questions: list[str] = Field(default_factory=list)
    considerations: list[str] = Field(default_factory=list)


class DecisionResult(BaseModel):
    key: str
    company: str
    question: str
    ranking: list[OptionVerdict]
    pairings: list[dict] = Field(default_factory=list)
    packet_hash: str = ""
    model: str = "?"
    call_stats: list[dict] = Field(default_factory=list)


# ----------------------------------------------------------------------------- prompts

def _packet_for(spec: DecisionSpec, packet: EvidencePacket) -> EvidencePacket:
    items = packet.items
    if spec.evidence_kinds:
        items = [i for i in items if i.kind in spec.evidence_kinds or i.kind != "filing_text"]
    if spec.evidence_keywords:
        kw = tuple(k.lower() for k in spec.evidence_keywords)
        items = [i for i in items if i.kind != "filing_text" or any(k in i.content.lower() for k in kw)]
    return packet.model_copy(update={"items": items})


def _advocate_system(company: str, spec: DecisionSpec, mine: Option, other: Option) -> str:
    return f"""ROLE: advocate
You argue that the option "{mine.label}" is the right answer to this question for {company}: {spec.question}
Option you defend: {mine.label}: {mine.description}
Option you argue against: {other.label}: {other.description}
Rules:
- Use only the numbered evidence items. Every claim must cite evidence ids and carry a "quote": a verbatim span (5 to 25 words) copied exactly from ONE cited item.
- Never introduce numbers that are not in the packet. Prefer the company's own filings over peer or generic evidence.
- At most 5 claims and 3 rebuttals, each under 40 words. Say plainly what would make your option wrong.
{_JSON_RULES}
Output JSON: {{"claims": [{{"text": "...", "evidence_ids": ["E1"], "quote": "..."}}], "rebuttals": [{{"text": "...", "evidence_ids": ["E2"], "quote": "..."}}], "what_would_change_my_mind": "..."}}"""


def _judge_system(company: str, spec: DecisionSpec, a: Option, b: Option) -> str:
    cons = "; ".join(spec.considerations) if spec.considerations else "data availability, comparability, what the filings actually disclose, stability over time"
    return f"""ROLE: chooser
You decide between two options for {company}. Question: {spec.question}
A: {a.label}: {a.description}
B: {b.label}: {b.description}
You have each side's argued case (claims with verified quotes) and the evidence packet.
Judge on evidence, not rhetoric. Address these considerations explicitly: {cons}. What settles this kind of question: {spec.settles or 'company disclosure'}.
For each option give a confidence (the two need not sum to 1), reasoning in 2 to 4 sentences that names the evidence ids you relied on,
"citations": the evidence ids whose text supports your reasoning, "questions": 1 to 3 questions that would settle the choice,
"considerations": the trade-offs or conflicts with other modeling decisions you weighed.
{_JSON_RULES}
Output JSON: {{"winner": "A"|"B", "options": {{"A": {{"confidence": <0-1>, "reasoning": "...", "citations": ["E1"], "questions": ["..."], "considerations": ["..."]}},
"B": {{"confidence": <0-1>, "reasoning": "...", "citations": ["E3"], "questions": ["..."], "considerations": ["..."]}}}}}}"""


def _rank_system(company: str, spec: DecisionSpec) -> str:
    opts = "\n".join(f"- {o.label}: {o.description}" for o in spec.options)
    cons = "; ".join(spec.considerations) if spec.considerations else "data availability, comparability, what the filings disclose"
    return f"""ROLE: ranker
Rank the options for {company}. Question: {spec.question}
Options:
{opts}
Use only the numbered evidence items. Address these considerations: {cons}. What settles this kind of question: {spec.settles or 'company disclosure'}.
For every option give a confidence (0 to 1), reasoning in 2 to 4 sentences naming evidence ids, "citations" (evidence ids that support it),
"questions" (1 to 3 that would settle it) and "considerations" (trade-offs, conflicts with other modeling decisions).
{_JSON_RULES}
Output JSON: {{"ranking": [{{"label": "<option label>", "confidence": <0-1>, "reasoning": "...", "citations": ["E1"], "questions": ["..."], "considerations": ["..."]}}]}}"""


# ----------------------------------------------------------------------------- engine

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def match_option(text: str, options: list[Option]) -> Option | None:
    """The model's label for an option, matched leniently. A trailing ': description' is dropped first; then the best
    scoring option wins (exact label or value 4, prefix 3, whole-label substring 2, fuzzy > 0.6 1), longest label on ties,
    so "reverse DCF: back out..." matches reverse DCF and not DCF."""
    import difflib

    raw = str(text)
    head = raw.split(":")[0] if ":" in raw else raw
    best, best_score = None, 0
    for cand in (head, raw):
        t = _norm(cand)
        if not t:
            continue
        for o in options:
            lab, val = _norm(o.label), _norm(o.value)
            if t in (lab, val):
                score = 4
            elif t.startswith(lab + " ") or lab.startswith(t + " ") or t.startswith(lab):
                score = 3
            elif re.search(r"(?:^| )" + re.escape(lab) + r"(?: |$)", t) or re.search(r"(?:^| )" + re.escape(t) + r"(?: |$)", lab):
                score = 2
            elif difflib.SequenceMatcher(None, t, lab).ratio() > 0.6:
                score = 1
            else:
                continue
            if score > best_score or (score == best_score and best is not None and len(lab) > len(_norm(best.label))):
                best, best_score = o, score
        if best_score >= 3:
            break
    return best


def _cache_key(packet: EvidencePacket, spec: DecisionSpec, labels: tuple[str, ...], seed: int, model: str, cheap: bool) -> str:
    payload = json.dumps({"p": packet.hash(), "ids": packet.ids(), "k": spec.key, "o": labels, "seed": seed, "m": model, "v": PROMPT_VERSION, "cheap": cheap}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _verdict_from(raw: dict, opt: Option, packet: EvidencePacket) -> OptionVerdict:
    ids = set(packet.ids())
    try:
        conf = min(max(float(raw.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    return OptionVerdict(label=opt.label, confidence=conf, reasoning=str(raw.get("reasoning", ""))[:900],
                         citations=[c for c in (raw.get("citations") or []) if isinstance(c, str) and c in ids][:6],
                         questions=[str(q)[:200] for q in (raw.get("questions") or []) if str(q).strip()][:3],
                         considerations=[str(c)[:200] for c in (raw.get("considerations") or []) if str(c).strip()][:4])


def pair_debate(packet: EvidencePacket, spec: DecisionSpec, a: Option, b: Option, llm: LLM, seed: int = 1, temperature: float = 0.5,
                cheap: bool = False, cache_dir: Path | None = None) -> dict:
    """One pairing: two advocates (skipped when cheap) then the judge. Returns {winner, verdicts{A,B}, arguments, stats}."""
    key = _cache_key(packet, spec, (a.label, b.label), seed, getattr(llm, "name", "?"), cheap)
    if cache_dir:
        p = Path(cache_dir) / f"decision_{key}.json"
        if p.exists():
            try:
                cached = json.loads(p.read_text())
                cached["stats"], cached["cached"] = [], True  # no calls were made
                return cached
            except Exception:
                pass
    stats: list[dict] = []
    rendered = packet.render()
    args: dict[str, dict] = {}
    if not cheap:
        for side, mine, other in (("A", a, b), ("B", b, a)):
            raw = _complete(llm, _advocate_system(packet.company, spec, mine, other), f"EVIDENCE PACKET:\n{rendered}\n\nMake the case for {mine.label}.",
                            seed, temperature, stats, f"advocate-{side}")
            kept, dropped, _ = validate_claims(_as_claims(raw.get("claims")), packet, require_quote=True)
            reb, dropped2, _ = validate_claims(_as_claims(raw.get("rebuttals")), packet, require_quote=True)
            args[side] = {"label": mine.label, "claims": [c.model_dump() for c in kept], "rebuttals": [c.model_dump() for c in reb],
                          "dropped": dropped + dropped2, "change_mind": str(raw.get("what_would_change_my_mind", ""))[:300]}
    case_text = ""
    for side in ("A", "B"):
        if side in args:
            lines = [f"- [{','.join(c['evidence_ids'])}] {c['text']} | quote: \"{c.get('quote', '')}\"" for c in args[side]["claims"] + args[side]["rebuttals"]]
            case_text += f"\nCASE FOR {side} ({args[side]['label']}):\n" + ("\n".join(lines) if lines else "(no verified claims)") + "\n"
    raw = _complete(llm, _judge_system(packet.company, spec, a, b), f"EVIDENCE PACKET:\n{rendered}\n{case_text}\nDecide.", seed, 0.2, stats, "chooser")
    opts = raw.get("options") or {}
    va, vb = _verdict_from(opts.get("A") or {}, a, packet), _verdict_from(opts.get("B") or {}, b, packet)
    winner = "A" if str(raw.get("winner", "A")).strip().upper().startswith("A") else "B"
    if va.confidence != vb.confidence and ((va.confidence > vb.confidence) != (winner == "A")):
        winner = "A" if va.confidence > vb.confidence else "B"  # the stated winner must be the higher-confidence side
    out = {"winner": winner, "verdicts": {"A": va.model_dump(), "B": vb.model_dump()}, "arguments": args, "stats": stats, "labels": {"A": a.label, "B": b.label}}
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"decision_{key}.json").write_text(json.dumps(out))
    return out


def decide(packet: EvidencePacket, spec: DecisionSpec, llm: LLM, seed: int = 1, cheap: bool = False, cache_dir: Path | None = None,
           progress=None) -> DecisionResult:
    """Two options: one pairing. More: single-elimination tournament; losers are ranked by their losing-round confidence."""
    pk = _packet_for(spec, packet)
    opts = list(spec.options)
    stats: list[dict] = []
    pairings: list[dict] = []
    if cheap and len(opts) > 2 and not spec.tournament:
        ranking: list[OptionVerdict] = []
        returned: list[str] = []
        for attempt in range(2):
            system = _rank_system(packet.company, spec)
            if attempt:
                system += "\nUse these labels exactly, one entry each: " + " | ".join(o.label for o in opts)
            raw = _complete(llm, system, f"EVIDENCE PACKET:\n{pk.render()}\n\nRank the options.", seed + attempt, 0.0 if attempt else 0.2, stats, "ranker")
            ranking, seen = [], set()
            for r in raw.get("ranking") or []:
                returned.append(str(r.get("label", "")))
                o = match_option(str(r.get("label", "")), opts)
                if o and o.label not in seen:
                    seen.add(o.label)
                    ranking.append(_verdict_from(r, o, pk))
            if len(ranking) >= max(2, len(opts) - 1):
                break
        for o in opts:
            if o.label not in {v.label for v in ranking}:
                ranking.append(OptionVerdict(label=o.label, confidence=0.0,
                                             reasoning=f"model ranking unmatched (labels returned: {', '.join(returned) or 'none'}); rerun `choose --keys {spec.key}`"))
        ranking.sort(key=lambda v: -v.confidence)
        return DecisionResult(key=spec.key, company=packet.company, question=spec.question, ranking=ranking, packet_hash=pk.hash(),
                              model=getattr(llm, "name", "?"), call_stats=stats)
    alive = opts[:]
    eliminated: list[OptionVerdict] = []
    rnd = 0
    while len(alive) > 1:
        rnd += 1
        nxt = []
        for i in range(0, len(alive) - 1, 2):
            a, b = alive[i], alive[i + 1]
            if progress:
                progress(f"[{spec.key}] round {rnd}: {a.label} vs {b.label}")
            res = pair_debate(pk, spec, a, b, llm, seed=seed + rnd, cheap=cheap, cache_dir=cache_dir)
            stats += res.get("stats", [])
            pairings.append({"round": rnd, "a": a.label, "b": b.label, "winner": res["labels"][res["winner"]], "arguments": res.get("arguments", {})})
            win, lose = (a, b) if res["winner"] == "A" else (b, a)
            nxt.append(win)
            eliminated.append(OptionVerdict(**res["verdicts"]["B" if res["winner"] == "A" else "A"]))
            final_v = OptionVerdict(**res["verdicts"][res["winner"]])
            if len(alive) == 2:
                eliminated.append(final_v)
        if len(alive) % 2 == 1:
            nxt.append(alive[-1])  # bye
        alive = nxt
    winner = alive[0]
    ranking = [v for v in eliminated if v.label == winner.label][-1:] or [OptionVerdict(label=winner.label, confidence=0.5)]
    losers = [v for v in eliminated if v.label != winner.label]
    seen = set()
    for v in sorted(losers, key=lambda v: -v.confidence):
        if v.label not in seen:
            seen.add(v.label)
            ranking.append(v)
    return DecisionResult(key=spec.key, company=packet.company, question=spec.question, ranking=ranking, pairings=pairings, packet_hash=pk.hash(),
                          model=getattr(llm, "name", "?"), call_stats=stats)


# ----------------------------------------------------------------------------- ledger

class DecisionLedger(MappingLedger):
    """Same columns and Excel round trip as the mapping ledger; rows are decision options."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.path = Path(data_dir) / "decision_ledger.csv"
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype=str).fillna("")
            for c in LEDGER_COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = ""
        else:
            self.df = pd.DataFrame(columns=LEDGER_COLUMNS)

    def reset(self, ticker: str, key: str) -> int:
        """Drop every row of one decision (a rerun after new evidence; the analyst's earlier choice is retired on purpose)."""
        mask = (self.df.ticker == ticker.upper()) & (self.df.statement == key)
        n = int(mask.sum())
        if n:
            self.df = self.df[~mask].reset_index(drop=True)
            self._dirty = True
        return n

    def record(self, ticker: str, result: DecisionResult, packet: EvidencePacket, spec: DecisionSpec, cheap: bool = False, reset: bool = False) -> int:
        n = 0
        if reset:
            self.reset(ticker, spec.key)
        by_id = {i.id: i for i in packet.items}
        for rank, v in enumerate(result.ranking, start=1):
            opt = next((o for o in spec.options if o.label == v.label), None)
            alts = [{"target": o.label, "confidence": round(w.confidence, 2)} for o, w in
                    ((next((x for x in spec.options if x.label == w.label), None), w) for w in result.ranking) if o and w.label != v.label][:3]
            cites = " || ".join(f"{by_id[c].source} | \"{by_id[c].content[:300]}\"" for c in v.citations if c in by_id)
            ev = f"rank {rank}/{len(result.ranking)}; judge confidence {v.confidence:.2f}; pairings: " + \
                 "; ".join(f"R{p['round']} {p['a']} vs {p['b']} -> {p['winner']}" for p in result.pairings if v.label in (p["a"], p["b"]))
            self.upsert(ticker, spec.key, "option", v.label, (opt.value if opt else v.label), v.confidence, alts, "pending",
                        f"judge:{result.model}{' cheap' if cheap else ''}", evidence=ev, relation=v.reasoning, citation=cites,
                        questions=" | ".join(v.questions), duplicates=" | ".join(v.considerations))
            n += 1
        return n

    def selected(self, ticker: str, key: str) -> list[str]:
        d = self.rows(ticker, key, "option", "accepted")
        return list(d.target)


_VALUES_RE = re.compile(r"values (\{.*?\})")
_MATERIALITY_RE = re.compile(r"(-?\d+(?:\.\d+)?)% of base")


def _values_key(evidence: str) -> str:
    m = _VALUES_RE.search(str(evidence))
    return m.group(1) if m else ""


def _materiality(evidence: str) -> float:
    m = _MATERIALITY_RE.search(str(evidence))
    return abs(float(m.group(1))) if m else 0.0


def _current_mapping(ledger: MappingLedger, ticker: str, statement: str, target: str) -> str:
    """What the rules currently map the proposed target from (so the analyst knows a link will act as a component)."""
    if not target or target in ("residual", "ignore"):
        return ""
    try:
        d = ledger.rows(ticker, statement, status="accepted")
    except Exception:
        return ""
    d = d[(d.target == target) & d.proposed_by.astype(str).str.startswith("rule")]
    if not len(d):
        return "nothing yet: an accepted link becomes the source for this line"
    srcs = [f"{r.source} ({r.evidence})" if r.evidence else str(r.source) for _, r in d.head(3).iterrows()]
    return "; ".join(srcs) + " — an accepted link is a component: it fills years this total lacks (summed with other links) and is checked against the total elsewhere"


def review_terminal(ledger: MappingLedger, ticker: str, statement: str | None = None, analyst: str = "analyst", decisions: dict | None = None,
                    io_in=None, io_out=None, ask_fn=None) -> dict:
    """Walk pending rows, most material first. Interactive keys: a accept, r reject, u unsure, t<key> retarget, n<note>, s skip, q quit.
    A decision is copied to other pending rows that carry the same values and the same proposed target (a label and its
    concept twin), with a note saying so. ``decisions`` (source -> (status, target, note)) drives the loop without a terminal."""
    io_in, io_out = io_in or sys.stdin, io_out or sys.stdout
    pend = ledger.rows(ticker, statement, status="pending").copy()
    counts = {"accepted": 0, "rejected": 0, "unsure": 0, "skipped": 0, "auto": 0}
    pend["_mat"] = pend.evidence.map(_materiality)
    pend["_conf"] = pd.to_numeric(pend.confidence, errors="coerce").fillna(0.0)
    pend = pend.sort_values(["statement", "_mat", "_conf"], ascending=[True, False, False])
    done: set[str] = set()

    def _apply(r, status, target, note):
        ledger.decide(ticker, r.source, status, target=target, decided_by=analyst, note=note or "", statement=r.statement)
        counts[status] += 1
        done.add((r.statement, r.source))
        vk = _values_key(r.evidence)
        if vk:
            twins = pend[(pend.statement == r.statement) & (pend.source != r.source) & (pend.target == r.target) & (pend.evidence.map(_values_key) == vk)]
            for _, t in twins.iterrows():
                if (t.statement, t.source) in done:
                    continue
                ledger.decide(ticker, t.source, status, target=target, decided_by=analyst, note=f"same values as {r.source}; {note}".strip("; "), statement=t.statement)
                done.add((t.statement, t.source))
                counts["auto"] += 1

    for _, r in pend.iterrows():
        if (r.statement, r.source) in done:
            continue
        if decisions is not None:
            d = decisions.get(r.source)
            if not d:
                counts["skipped"] += 1
                continue
            status, target, note = d
            _apply(r, status, target, note)
            continue
        io_out.write(f"\n{'=' * 100}\n[{r.statement}] {r.source}\n  proposed target: {r.target}   confidence {r.confidence}   by {r.proposed_by}\n")
        io_out.write(f"  alternatives: {r.alternatives}\n  evidence: {str(r.evidence)[:400]}\n")
        cur = _current_mapping(ledger, ticker, r.statement, r.target) if r.source_type != "option" else ""
        if cur:
            io_out.write(f"  target now mapped from: {cur[:400]}\n")
        if r.relation:
            io_out.write(f"  reasoning: {str(r.relation)[:600]}\n")
        if r.citation:
            io_out.write(f"  citation: {str(r.citation)[:600]}\n")
        if r.questions:
            io_out.write(f"  questions: {r.questions}\n")
        if r.duplicates:
            io_out.write(f"  considerations/duplicates: {r.duplicates}\n")
        if not r.relation and str(r.proposed_by).startswith("judge") and float(r._conf) == 0:
            io_out.write("  WARNING: the model's ranking did not match this option (rerun `choose --keys <decision>`); accepting it records nothing useful\n")
        io_out.write("  [a]ccept  [r]eject  [u]nsure  [t<target>] retarget+accept  [n<note>] note  [?<question>] ask the filings  [s]kip  [q]uit > ")
        io_out.flush()
        note = ""
        qa_notes: list[str] = []
        while True:
            line = io_in.readline()
            if not line:
                ledger.save()
                return counts
            cmd = line.strip()
            if cmd.startswith("n") and cmd != "n":
                note = cmd[1:].strip()
                io_out.write("  note recorded; now decide > ")
                io_out.flush()
                continue
            if cmd.startswith("?") and len(cmd) > 1:
                if ask_fn is None:
                    io_out.write("  (no model configured for questions; pass --provider) > ")
                else:
                    from .ask import format_answer, note_line

                    ans = ask_fn(r.to_dict(), cmd[1:].strip())
                    io_out.write(format_answer(ans) + "\n  decide, or ask again > ")
                    qa_notes.append(note_line(ans))
                io_out.flush()
                continue
            break
        if qa_notes:
            note = " | ".join([note] + qa_notes) if note else " | ".join(qa_notes)
        if cmd == "q":
            break
        if cmd == "s" or not cmd:
            counts["skipped"] += 1
        elif cmd == "a":
            _apply(r, "accepted", None, note)
        elif cmd == "r":
            _apply(r, "rejected", None, note)
        elif cmd == "u":
            _apply(r, "unsure", None, note or "marked unsure")
        elif cmd.startswith("t") and len(cmd) > 1:
            _apply(r, "accepted", cmd[1:].strip(), note)
        else:
            counts["skipped"] += 1
    ledger.save()
    return counts


REVIEW_COLUMNS = ["area", "statement_or_decision", "source_or_option", "source_type", "proposed_target", "confidence", "alternatives", "evidence",
                  "target_now_mapped_from", "reasoning", "citation", "questions", "considerations_duplicates", "proposed_by", "status",
                  "DECISION", "TARGET_OVERRIDE", "NOTE", "FOLLOW_UP_QUESTION", "ANSWER"]
DECISION_CHOICES = ["accepted", "rejected", "unsure", "skip"]


def review_frame(ticker: str, ledger: MappingLedger | None, area: str, include_decided: bool = False) -> pd.DataFrame:
    """The review columns for one ledger (mapping or decisions), most material first."""
    d = ledger.rows(ticker) if ledger is not None else pd.DataFrame(columns=LEDGER_COLUMNS)
    if not include_decided and len(d):
        d = d[d.status == "pending"]
    rows = []
    if len(d):
        d = d.copy()
        d["_mat"] = d.evidence.map(_materiality)
        d = d.sort_values(["statement", "_mat"], ascending=[True, False])
    for _, r in d.iterrows():
        cur = _current_mapping(ledger, ticker, r.statement, r.target) if area == "mapping" else ""
        try:
            conf = float(r.confidence)
        except (TypeError, ValueError):
            conf = None
        rows.append({"area": area, "statement_or_decision": r.statement, "source_or_option": r.source, "source_type": r.source_type, "proposed_target": r.target,
                     "confidence": conf, "alternatives": r.alternatives, "evidence": r.evidence, "target_now_mapped_from": cur, "reasoning": r.relation,
                     "citation": r.citation, "questions": r.questions, "considerations_duplicates": r.duplicates, "proposed_by": r.proposed_by, "status": r.status,
                     "DECISION": "", "TARGET_OVERRIDE": "", "NOTE": r.note if include_decided else "", "FOLLOW_UP_QUESTION": "", "ANSWER": ""})
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def export_review(path: Path, ticker: str, mapping: MappingLedger | None, decisions: MappingLedger | None, include_decided: bool = False) -> Path:
    """One workbook the analyst can take away: every pending mapping proposal and decision option with all the fields the
    terminal shows (proposed target, confidence, alternatives, evidence, what the target is currently mapped from, reasoning,
    citation, questions, considerations) plus DECISION / TARGET_OVERRIDE / NOTE / FOLLOW_UP_QUESTION columns to fill in and
    import back with `review-import`."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    wb.remove(wb.active)
    lists = wb.create_sheet("lists")
    for i, c in enumerate(DECISION_CHOICES, start=1):
        lists.cell(row=i, column=1, value=c)
    lists.sheet_state = "hidden"
    fill = PatternFill("solid", fgColor="FFF2CC")

    def _sheet(name, ledger, area):
        ws = wb.create_sheet(name)
        ws.append(REVIEW_COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for _, r in review_frame(ticker, ledger, area, include_decided).iterrows():
            ws.append([r[c] for c in REVIEW_COLUMNS])
        n = ws.max_row
        if n >= 2:
            dv = DataValidation(type="list", formula1=f"=lists!$A$1:$A${len(DECISION_CHOICES)}", allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"P2:P{n}")
            for row in ws.iter_rows(min_row=2, max_row=n, min_col=16, max_col=19):
                for c in row:
                    c.fill = fill
        widths = {"A": 10, "B": 18, "C": 44, "D": 10, "E": 22, "F": 10, "G": 30, "H": 50, "I": 50, "J": 60, "K": 70, "L": 60, "M": 50, "N": 22, "O": 10,
                  "P": 12, "Q": 18, "R": 40, "S": 50, "T": 60}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "D2"
        return ws

    _sheet("mapping", mapping, "mapping")
    _sheet("decisions", decisions, "decision")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return Path(path)


def import_review(path: Path, ticker: str, mapping: MappingLedger | None, decisions: MappingLedger | None, analyst: str = "analyst",
                  ask_fn=None, answered_path: Path | None = None) -> dict:
    """Apply the DECISION / TARGET_OVERRIDE / NOTE columns of a review workbook; answer every FOLLOW_UP_QUESTION with
    ``ask_fn(row, question)`` (citations and confidence go into the ledger note and into an answered copy of the workbook)."""
    from openpyxl import load_workbook

    from .ask import note_line

    wb = load_workbook(path)
    counts = {"accepted": 0, "rejected": 0, "unsure": 0, "skipped": 0, "answered": 0}
    answers: list[dict] = []
    for name, ledger in (("mapping", mapping), ("decisions", decisions)):
        if name not in wb.sheetnames or ledger is None:
            continue
        ws = wb[name]
        header = [c.value for c in ws[1]]
        col = {h: i for i, h in enumerate(header)}
        for row in ws.iter_rows(min_row=2):
            vals = [c.value for c in row]
            rec = {h: vals[col[h]] for h in header if h in col}
            statement, source = str(rec.get("statement_or_decision") or ""), str(rec.get("source_or_option") or "")
            if not source:
                continue
            q = str(rec.get("FOLLOW_UP_QUESTION") or "").strip()
            note = str(rec.get("NOTE") or "").strip()
            if q and ask_fn is not None:
                lrow = {"statement": statement, "source": source, "target": rec.get("proposed_target", ""), "relation": rec.get("reasoning", ""),
                        "citation": rec.get("citation", ""), "questions": rec.get("questions", ""), "duplicates": rec.get("considerations_duplicates", "")}
                ans = ask_fn(lrow, q)
                answers.append({"sheet": name, "source": source, **ans})
                row[col["ANSWER"]].value = f"{ans['answer']} [confidence {ans['confidence']:.2f}] " + " | ".join(f"{c['source']}: {c['text'][:120]}" for c in ans.get("citations", []))
                note = (note + " | " if note else "") + note_line(ans)
                counts["answered"] += 1
            dec = str(rec.get("DECISION") or "").strip().lower()
            if dec in ("accepted", "rejected", "unsure"):
                target = str(rec.get("TARGET_OVERRIDE") or "").strip() or None
                ledger.decide(ticker, source, dec, target=target, decided_by=analyst, note=note or ("marked unsure" if dec == "unsure" else ""), statement=statement)
                counts[dec] += 1
            else:
                counts["skipped"] += 1
                if note and not dec:
                    pass  # a note without a decision is kept only in the answered workbook
        ledger.save()
    if answered_path or counts["answered"]:
        ap = answered_path or Path(path).with_name(Path(path).stem + "_answered.xlsx")
        wb.save(ap)
        counts["answered_workbook"] = str(ap)
    counts["answers"] = answers
    return counts


def export_workbook(path: Path, ticker: str, mapping: MappingLedger | None, decisions: MappingLedger | None, brief: dict | None = None,
                    unsure_path: Path | None = None) -> tuple[Path, Path | None]:
    """Final workbook: mapping_ledger, decision_ledger, brief. Unsure rows from both ledgers go to a second workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    def _sheet(wb, name, df):
        ws = wb.create_sheet(name)
        ws.append(LEDGER_COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for _, r in df.iterrows():
            row = []
            for c in LEDGER_COLUMNS:
                v = r.get(c, "")
                if c == "confidence":
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = None
                row.append(v)
            ws.append(row)
        ws.freeze_panes = "A2"
        return ws

    wb = Workbook()
    wb.remove(wb.active)
    unsure_frames = []
    if mapping is not None:
        d = mapping.rows(ticker)
        _sheet(wb, "mapping_ledger", d[d.status != "unsure"])
        unsure_frames.append(d[d.status == "unsure"])
    if decisions is not None:
        d = decisions.rows(ticker)
        _sheet(wb, "decision_ledger", d[d.status != "unsure"])
        unsure_frames.append(d[d.status == "unsure"])
    if brief:
        ws = wb.create_sheet("brief")
        ws.append(["section", "text", "citations"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for sec in ("what_it_does", "why_now", "what_the_debate_is"):
            b = brief.get(sec) or {}
            ws.append([sec, b.get("text", ""), " || ".join(b.get("citations", []))])
        ws.column_dimensions["B"].width = 100
        ws.column_dimensions["C"].width = 120
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    up = None
    if unsure_frames:
        u = pd.concat(unsure_frames, ignore_index=True) if unsure_frames else pd.DataFrame(columns=LEDGER_COLUMNS)
        up = unsure_path or Path(path).with_name(Path(path).stem + "_unsure.xlsx")
        wb2 = Workbook()
        wb2.remove(wb2.active)
        _sheet(wb2, "unsure", u)
        wb2.save(up)
    return Path(path), up

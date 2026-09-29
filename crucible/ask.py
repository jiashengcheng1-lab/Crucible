"""Follow-up questions while reviewing: the analyst asks, the model answers from the filings only, with citations and a
confidence. Retrieval is deterministic (sentences that share the question's words, from Items 1, 1A, 7 and 8 and the
release KPI sentences in the dossier), the answer may only cite offered context ids, and unsupported questions are
answered as unsupported rather than guessed. Every exchange is appended to logs/qa.jsonl."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .debate import _JSON_RULES, _complete
from .evidence import EvidencePacket
from .llm import LLM

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "this", "that", "it", "as", "at", "by", "with",
         "from", "does", "do", "did", "how", "what", "which", "why", "when", "where", "who", "than", "into", "its", "their", "company", "any", "not",
         "have", "has", "had", "been", "will", "would", "should", "could", "can", "may", "might", "more", "most", "such", "these", "those", "there"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9&\-]{2,}", str(text).lower()) if t not in _STOP]


_STATEMENT_TRIGGERS = {"BS": ("balance sheet", "balance", "assets", "liabilities", "equity", "debt", "cash position"),
                       "IS": ("income statement", "revenue", "margin", "cogs", "cost of revenue", "operating income", "net income", "sg&a", "sga", "r&d", "eps"),
                       "CF": ("cash flow", "capex", "free cash", "buyback", "dividend", "financing", "investing", "operating cash"),
                       "MODEL": ("model", "forecast", "dcf", "valuation", "excel", "sheet", "workbook", "driver", "scenario", "wacc", "per share", "target price")}


def _state_context(question: str, state: list[dict], row: dict | None, k: int = 40) -> list[dict]:
    """Model-state items relevant to the question: every line of a statement the question or the row names, plus items
    that share the question's words."""
    q = question.lower()
    wanted = {st for st, trig in _STATEMENT_TRIGGERS.items() if any(t in q for t in trig)}
    if row and row.get("statement") in ("IS", "BS", "CF") and any(w in q for w in ("this", "line", "now", "mapped", "under", "current")):
        wanted.add(row["statement"])
    toks = set(_tokens(question))
    picked = [it for it in state if it.get("statement") in wanted]
    if len(picked) < k:
        for it in state:
            if it in picked:
                continue
            low = it["text"].lower()
            if sum(1 for t in toks if t in low) >= 2:
                picked.append(it)
    return picked[:k]


def retrieve(question: str, sections: dict[str, str], packet: EvidencePacket | None = None, k_sent: int = 10, k_items: int = 6,
             extra_terms: str = "", state: list[dict] | None = None, row: dict | None = None) -> list[dict]:
    """Numbered context: the model's own state (statements as mapped, forecast and DCF as built) where the question asks for
    it, then the filing sentences and evidence items that share the question's distinctive words."""
    ctx: list[dict] = []
    for it in _state_context(question, state or [], row):
        ctx.append({"id": f"C{len(ctx) + 1}", "source": it["source"], "text": it["text"][:600], "kind": "state"})
    toks = set(_tokens(question) + _tokens(extra_terms))
    scored: list[tuple[int, str, str]] = []
    if toks:
        for name, text in (sections or {}).items():
            if not text:
                continue
            for s in _SENT.split(re.sub(r"\s+", " ", text)):
                s = s.strip()
                if not (40 <= len(s) <= 420):
                    continue
                low = s.lower()
                hit = sum(1 for t in toks if t in low)
                if hit >= max(1, min(2, len(toks) // 3)):
                    bonus = 2 if name.startswith("Item 8") else (1 if name.startswith(("Item 7", "Item 1")) else 0)
                    if re.search(r"\d", s):
                        bonus += 1
                    scored.append((hit * 2 + bonus, name, s))
    scored.sort(key=lambda x: -x[0])
    seen: set = set()
    for sc, name, s in scored:
        key = s.lower()[:70]
        if key in seen:
            continue
        seen.add(key)
        ctx.append({"id": f"C{len(ctx) + 1}", "source": f"10-K {name}", "text": s, "kind": "filing"})
        if sum(1 for c in ctx if c.get("kind") == "filing") >= k_sent:
            break
    if packet is not None and toks:
        items = []
        for it in packet.items:
            low = it.content.lower()
            hit = sum(1 for t in toks if t in low)
            if hit:
                items.append((hit, it))
        items.sort(key=lambda x: -x[0])
        for _, it in items[:k_items]:
            ctx.append({"id": f"C{len(ctx) + 1}", "source": it.source, "text": it.content[:400], "evidence_id": it.id, "kind": "evidence"})
    return ctx


_SYSTEM = """ROLE: answerer
You answer an analyst's follow-up question about a company from the numbered context only. The context has three kinds of
item: the model's own state (the three statements as mapped now, the forecast and DCF as last built, the workbook's sheets),
filing sentences, and computed evidence. Questions about what the model or a statement currently holds are answered from the
state items (list the lines and values asked for); questions about the company are answered from the filings. Rules: cite the
context ids that support each claim; do not use knowledge outside the context; if the context does not answer the question,
say so in one sentence and set "unsupported": true. Give a confidence from 0 to 1 that the answer is correct given the
context. Keep the answer under 160 words; a requested list of lines may be longer.
""" + _JSON_RULES + """
Output JSON: {"answer": "...", "citations": ["C1", "C3"], "confidence": <0-1>, "unsupported": false}"""


def ask(question: str, row: dict | None, packet: EvidencePacket | None, sections: dict[str, str], llm: LLM, seed: int = 1,
        log_dir: Path | None = None, ticker: str = "", state: list[dict] | None = None) -> dict:
    """Answer with citations and a confidence. ``row`` is the ledger row being reviewed (source, target, reasoning, citation...);
    ``state`` is the model's own state from crucible.state.state_items."""
    row = row or {}
    extra = " ".join(str(row.get(k, "")) for k in ("source", "target", "relation", "statement"))
    ctx = retrieve(question, sections, packet, extra_terms=extra, state=state, row=row)
    row_txt = ""
    if row:
        row_txt = "ROW UNDER REVIEW:\n" + "\n".join(f"  {k}: {str(row.get(k, ''))[:300]}" for k in ("statement", "source", "target", "confidence", "relation", "citation", "questions", "duplicates") if row.get(k)) + "\n\n"
    ctx_txt = "\n".join(f"[{c['id']}] ({c['source']}) {c['text']}" for c in ctx) or "(no context matched the question)"
    stats: list[dict] = []
    raw = _complete(llm, _SYSTEM, f"{row_txt}CONTEXT:\n{ctx_txt}\n\nQUESTION: {question}", seed, 0.2, stats, "answerer")
    by_id = {c["id"]: c for c in ctx}
    ids = [c for c in (raw.get("citations") or []) if isinstance(c, str) and c in by_id]
    try:
        conf = min(max(float(raw.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    unsupported = bool(raw.get("unsupported")) or not ids
    if unsupported:
        conf = min(conf, 0.3)
    out = {"question": question, "answer": str(raw.get("answer", ""))[:2400], "confidence": round(conf, 2), "unsupported": unsupported,
           "citations": [{"id": i, "source": by_id[i]["source"], "text": by_id[i]["text"][:300]} for i in ids],
           "context_offered": len(ctx), "model": getattr(llm, "name", "?"), "row_source": row.get("source", ""), "ticker": ticker,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir) / "qa.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


def format_answer(a: dict) -> str:
    lines = [f"  answer (confidence {a['confidence']:.2f}{', unsupported by the filings or the model state' if a.get('unsupported') else ''}): {a['answer']}"]
    for c in a.get("citations", []):
        lines.append(f"    [{c['id']}] {c['source']} | {c['text'][:200]}")
    return "\n".join(lines)


def note_line(a: dict) -> str:
    """One line for the ledger note: the question, the answer, the confidence and the sources."""
    src = "; ".join(c["source"][:40] for c in a.get("citations", [])[:2])
    return f"Q: {a['question'][:120]} A: {a['answer'][:200]} [conf {a['confidence']:.2f}{'; unsupported' if a.get('unsupported') else ''}; {src}]"

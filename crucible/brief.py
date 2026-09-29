"""Company brief: what it does, why now, what the debate is. Each section carries evidence ids from the dossier, which
are verbatim filing text or computed metrics; the export writes the cited text beside each section."""
from __future__ import annotations

from .debate import _JSON_RULES, _complete
from .evidence import EvidencePacket
from .llm import LLM

_SYSTEM = """ROLE: brief
Write a one-page brief on the company from the numbered evidence only. Three sections, each 60 to 120 words:
- what_it_does: the business, what it sells, to whom, how it makes money, its segments and KPIs as disclosed;
- why_now: what changed in the last two filings or releases (guidance, orders, capacity, pricing, balance sheet) that makes the name timely;
- what_the_debate_is: the two or three questions on which the value depends, phrased as a bull and a bear would.
Every section must list the evidence ids it rests on in "citations" (only ids that exist). No numbers that are not in the packet.
""" + _JSON_RULES + """
Output JSON: {"what_it_does": {"text": "...", "citations": ["E1"]}, "why_now": {"text": "...", "citations": ["E2"]}, "what_the_debate_is": {"text": "...", "citations": ["E3"]}}"""


def write_brief(packet: EvidencePacket, llm: LLM, seed: int = 1) -> dict:
    stats: list[dict] = []
    raw = _complete(llm, _SYSTEM, f"EVIDENCE PACKET:\n{packet.render()}\n\nWrite the brief.", seed, 0.3, stats, "brief")
    by_id = {i.id: i for i in packet.items}
    out = {"company": packet.company, "as_of": packet.as_of, "model": getattr(llm, "name", "?"), "call_stats": stats}
    for sec in ("what_it_does", "why_now", "what_the_debate_is"):
        b = raw.get(sec) or {}
        ids = [c for c in (b.get("citations") or []) if isinstance(c, str) and c in by_id]
        out[sec] = {"text": str(b.get("text", ""))[:1500], "citation_ids": ids,
                    "citations": [f"[{c}] {by_id[c].source} | {by_id[c].content[:240]}" for c in ids]}
    return out


def brief_markdown(b: dict) -> str:
    md = [f"# {b.get('company')} brief (as of {b.get('as_of') or 'the latest filings on disk'})", ""]
    for sec, title in (("what_it_does", "What it does"), ("why_now", "Why now"), ("what_the_debate_is", "What the debate is")):
        s = b.get(sec) or {}
        md += [f"## {title}", "", s.get("text", ""), ""]
        for c in s.get("citations", []):
            md.append(f"- {c}")
        md.append("")
    return "\n".join(md)

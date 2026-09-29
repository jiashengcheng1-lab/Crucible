"""Peer discovery.

The 10-K's own "Competition" discussion is the primary source for who the
company thinks it competes with. The LLM's only job is to list the company
names that appear in that excerpt; the analyst confirms tickers. SIC code
neighbours are a second, deterministic candidate list when the analyst has
network access (edgartools can search by SIC).
"""
from __future__ import annotations

import re

from .llm import LLM

_SYSTEM = """ROLE: extractor
You extract the names of competitor companies that are explicitly named in a 10-K competition excerpt.
Return JSON: {"competitors": [{"name": "...", "context": "<= 15 words from the excerpt"}]}.
Only list companies named in the text. Do not add companies from your own knowledge."""


def extract_competitors(competition_excerpt: str, llm: LLM) -> list[dict]:
    if not competition_excerpt:
        return []
    raw = llm.complete_json(_SYSTEM, f"EXCERPT:\n{competition_excerpt}", seed=0, temperature=0.0)
    out = []
    for c in raw.get("competitors", []) if isinstance(raw, dict) else []:
        name = str(c.get("name", "")).strip()
        if name and re.search(re.escape(name.split()[0]), competition_excerpt, re.I):  # must appear in the text
            out.append({"name": name, "context": str(c.get("context", ""))[:120]})
    return out


def sic_candidates(sic: str | int | None, limit: int = 25) -> list[dict]:
    """Companies sharing the SIC code (requires network; best effort)."""
    if not sic:
        return []
    try:
        from edgar import get_filings  # noqa: F401  (only to confirm edgartools present)
        from edgar.reference import industries  # type: ignore
    except Exception:
        return []
    try:
        rows = industries.get_companies_by_sic(str(sic))  # may not exist in all versions
        return [{"name": r.get("name"), "cik": r.get("cik")} for r in rows[:limit]]
    except Exception:
        return []

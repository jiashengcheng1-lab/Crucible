"""LLM clients behind one tiny interface: ``complete_json(system, user, seed, temperature) -> dict``.

Providers are imported lazily so the package works with none installed. The
MockLLM is deterministic and reacts to the numeric evidence it is shown, which
lets the harness and tests exercise stability and ablation logic offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Protocol


def _dump_raw(text: str, tag: str) -> str:
    from datetime import datetime
    from pathlib import Path

    d = Path("logs") / "raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


class LLM(Protocol):
    name: str

    def complete_json(self, system: str, user: str, seed: int | None = None, temperature: float = 0.7) -> dict: ...


def _scan_object(text: str, start: int) -> int | None:
    """Index just past the object that starts at ``start``, ignoring braces inside strings."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _repair_truncated(text: str) -> str:
    """Trim a truncated JSON tail back to the last structurally complete point and close what is open.

    Walks the text once recording, at every ',' '}' ']' outside strings, the container stack at that point;
    then tries the latest cut first, appending closers, until json.loads succeeds.
    """
    cuts: list[tuple[int, list[str]]] = []
    stack: list[str] = []
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
            cuts.append((i + 1, list(stack)))
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, list(stack)))
        elif ch == ",":
            cuts.append((i, list(stack)))
    for pos, st in reversed(cuts[-200:]):
        candidate = text[:pos].rstrip().rstrip(",") + "".join("}" if c == "{" else "]" for c in reversed(st))
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return text


def _fix_inner_quotes(text: str) -> str:
    """Replace unescaped double quotes inside JSON string values with single quotes.

    A quote closes a string only if the next non-space character is structural (, : } ]); otherwise it is an inner
    quotation mark the model forgot to escape, which is the most common way a long argument breaks JSON parsing.
    """
    out, in_str, esc = [], False, False
    n = len(text)
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
                out.append(ch)
            elif ch == "\\":
                esc = True
                out.append(ch)
            elif ch == '"':
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                closing = j >= n or text[j] in ":}]"
                if not closing and j < n and text[j] == ",":
                    k = j + 1
                    while k < n and text[k] in " \t\r\n":
                        k += 1
                    closing = k >= n or text[k] in '"{[-0123456789tfn'  # a JSON token must follow a real closing quote
                if closing:
                    in_str = False
                    out.append(ch)
                else:
                    out.append("'")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model response (tolerates fences, prose, unescaped inner quotes, and a truncated tail)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*)", text, re.S)
    if m:
        text = m.group(1).split("```")[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    for candidate in (text, _fix_inner_quotes(text)):
        end = _scan_object(candidate, start)
        if end is not None:
            try:
                return json.loads(candidate[start:end])
            except json.JSONDecodeError:
                pass
    repaired = _repair_truncated(_fix_inner_quotes(text)[start:])
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse JSON from response ({e})") from e


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"  # override with CRUCIBLE_MODEL or --model; current ids: https://docs.claude.com/en/api/overview


@dataclass
class AnthropicLLM:
    model: str = field(default_factory=lambda: os.environ.get("CRUCIBLE_MODEL") or DEFAULT_ANTHROPIC_MODEL)
    max_tokens: int = 6000
    name: str = "anthropic"

    def __post_init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set in this shell. Run:  export ANTHROPIC_API_KEY=sk-ant-...   "
                             "(or: set -a; source .env; set +a). Optional: export CRUCIBLE_MODEL=<model id> or pass --model.")
        try:
            import anthropic  # lazy
        except ImportError as e:
            raise SystemExit("anthropic SDK missing: pip install -e '.[anthropic]'") from e
        self._client = anthropic.Anthropic()
        self.name = f"anthropic:{self.model}"

    def complete_json(self, system: str, user: str, seed: int | None = None, temperature: float | None = 0.7) -> dict:
        # Seed is recorded for provenance only: the API does not accept one, and the SDK (>=1.0) removed the
        # temperature/top_p/top_k keyword arguments. Sampling parameters go through extra_body, which the SDK
        # merges into the request JSON; current models may ignore them, which is exactly why the harness
        # measures run-to-run variation instead of assuming it can be switched off.
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=[{"role": "user", "content": user + "\n\nReturn a single JSON object and nothing else."}])
        if temperature is not None:
            kwargs["extra_body"] = {"temperature": float(temperature)}
        msg = self._create(kwargs)
        if getattr(msg, "stop_reason", None) == "max_tokens":  # truncated: one retry with a bigger budget
            kwargs["max_tokens"] = self.max_tokens * 2
            msg = self._create(kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        usage = getattr(msg, "usage", None)
        self.last_meta = {"stop_reason": getattr(msg, "stop_reason", None), "chars": len(text),
                          "input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}
        if os.environ.get("CRUCIBLE_DEBUG"):
            _dump_raw(text, "anthropic")
        try:
            return extract_json(text)
        except ValueError as e:
            path = _dump_raw(text, "anthropic_failed")
            raise ValueError(f"{e}; raw response saved to {path} (stop_reason={getattr(msg, 'stop_reason', None)})") from e

    def _create(self, kwargs: dict):
        try:
            return self._client.messages.create(**kwargs)
        except Exception as e:  # if this model rejects sampling parameters, retry without them
            if "extra_body" in kwargs and "temperature" in str(e).lower():
                kwargs.pop("extra_body")
                return self._client.messages.create(**kwargs)
            raise


@dataclass
class OpenAICompatLLM:
    model: str = field(default_factory=lambda: os.environ.get("CRUCIBLE_MODEL", ""))
    base_url: str | None = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL"))
    max_tokens: int = 2000
    name: str = "openai"

    def __post_init__(self) -> None:
        if not self.model:
            raise SystemExit("Set CRUCIBLE_MODEL (or pass --model) to the model id for the OpenAI-compatible endpoint")
        if not os.environ.get("OPENAI_API_KEY") and not self.base_url:
            raise SystemExit("OPENAI_API_KEY is not set in this shell (or set OPENAI_BASE_URL for a local server)")
        from openai import OpenAI  # lazy

        self._client = OpenAI(base_url=self.base_url) if self.base_url else OpenAI()
        self.name = f"openai:{self.model}"

    def complete_json(self, system: str, user: str, seed: int | None = None, temperature: float = 0.7) -> dict:
        kwargs: dict[str, Any] = dict(model=self.model, temperature=temperature, max_tokens=self.max_tokens,
                                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      response_format={"type": "json_object"})
        if seed is not None:
            kwargs["seed"] = seed
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or "{}"
        self.last_meta = {"stop_reason": getattr(resp.choices[0], "finish_reason", None), "chars": len(text)}
        if os.environ.get("CRUCIBLE_DEBUG"):
            _dump_raw(text, "openai")
        try:
            return extract_json(text)
        except ValueError as e:
            raise ValueError(f"{e}; raw response saved to {_dump_raw(text, 'openai')}") from e


_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


@dataclass
class MockLLM:
    """Deterministic stand-in. It reads the evidence block in the prompt, takes
    every '<id>: ... X%' value tagged as directly comparable, and answers with a
    role-biased summary statistic plus a little seeded noise. Removing an
    evidence item therefore moves the answer, which is what the ablation
    harness needs to demonstrate offline."""

    noise: float = 0.004
    name: str = "mock"

    def complete_json(self, system: str, user: str, seed: int | None = None, temperature: float = 0.7) -> dict:
        rng = random.Random(seed if seed is not None else int(hashlib.md5(user.encode()).hexdigest(), 16) % 10_000)
        role = "judge" if "ROLE: judge" in system else ("bull" if "ROLE: bull" in system else ("analyst" if "ROLE: analyst" in system else "bear"))
        for r in ("advocate", "chooser", "ranker", "brief", "profiler", "citer", "answerer"):
            if f"ROLE: {r}" in system:
                role = r
        ev = self._evidence(user)
        vals = [v for _, v, direct in ev if direct]
        ids = [i for i, _, direct in ev if direct]
        all_ids = re.findall(r"^\s*\[(E\d+)\]", user, flags=re.M)
        if role == "advocate":
            claims = [{"text": f"Evidence {i} supports this option", "evidence_ids": [i], "quote": self._quote_for(user, i)} for i in all_ids[:3]]
            return {"claims": claims, "rebuttals": [], "what_would_change_my_mind": "mock: contrary disclosure"}
        if role == "chooser":
            ca = 0.6 + (rng.random() - 0.5) * 0.1
            return {"winner": "A", "options": {"A": {"confidence": round(ca, 2), "reasoning": f"mock: A fits the evidence {', '.join(all_ids[:2])}", "citations": all_ids[:2],
                                                   "questions": ["mock: what does the segment note disclose?"], "considerations": ["mock: data availability"]},
                                             "B": {"confidence": round(1 - ca, 2), "reasoning": "mock: B weaker", "citations": all_ids[2:3],
                                                   "questions": ["mock: is the unit disclosed quarterly?"], "considerations": ["mock: comparability"]}}}
        if role == "ranker":
            labels = re.findall(r"^- ([^:\n]+):", system, flags=re.M)
            return {"ranking": [{"label": lab, "confidence": round(0.8 - 0.15 * k, 2), "reasoning": f"mock rank {k + 1}", "citations": all_ids[:1],
                                 "questions": ["mock question"], "considerations": ["mock consideration"]} for k, lab in enumerate(labels)]}
        if role == "profiler":
            m = re.search(r"ITEM 1 \(BUSINESS\) EXCERPT:\n(.*?)\n\nEVIDENCE ITEMS", user, flags=re.S)
            words = (m.group(1) if m else "").split()
            quote = " ".join(words[:8]) if len(words) >= 8 else ""
            fam = re.search(r"family guess from SIC: ([^.\n]+)", user)
            return {"industry": (fam.group(1) if fam else "mock industry")[:60], "business_model": "mock: sells things to customers.",
                    "revenue_drivers": [{"name": "units", "unit": "units", "disclosed": True, "frequency": "quarterly", "quote": quote}],
                    "cost_drivers": [{"name": "input cost", "unit": "USD per unit", "quote": quote}],
                    "kpis": [{"name": "units shipped", "unit": "units", "frequency": "quarterly", "quote": quote}],
                    "kpi_scheme_candidates": [{"label": "volume KPIs", "value": "volume", "description": "units and utilization", "quote": quote},
                                              {"label": "pricing KPIs", "value": "pricing", "description": "price per unit", "quote": quote}],
                    "unit_economics_candidates": [{"label": "per unit shipped", "value": "per_unit", "description": "revenue and cost per unit", "quote": quote}],
                    "template_candidates": [{"label": "units x price", "value": "units_price", "formula": "units x ASP", "drivers": ["units", "ASP"], "quote": quote}],
                    "competitors_named": ["Mock Rival Inc"]}
        if role == "citer":
            ids = re.findall(r"\b(P\d+)\b", user)
            sids = re.findall(r"\b(P\d+-S\d+)\b", user)
            picks = {}
            for pid in dict.fromkeys(ids):
                mine = [x for x in sids if x.startswith(pid + "-")]
                picks[pid] = {"citation_ids": mine[:1], "confidence": 0.7 if mine else 0.0, "why": "mock: the sentence names the line"}
            return {"citations": picks}
        if role == "answerer":
            ids = re.findall(r"^\s*\[(C\d+)\]", user, flags=re.M)
            return {"answer": "mock answer from the filings.", "citations": ids[:2], "confidence": 0.6 if ids else 0.2, "unsupported": not ids}
        if role == "brief":
            return {"what_it_does": {"text": "mock: the company does things.", "citations": all_ids[:2]},
                    "why_now": {"text": "mock: because of recent guidance.", "citations": all_ids[2:4]},
                    "what_the_debate_is": {"text": "mock: growth durability.", "citations": all_ids[:1]}}
        if role == "analyst":
            base = (statistics.mean(vals) if vals else 0.05) + rng.gauss(0, self.noise * temperature)
            spread = (statistics.pstdev(vals) if len(vals) > 1 else 0.01)
            claims = [{"text": f"Evidence {i} shows {v * 100:.1f}%", "evidence_ids": [i], "quote": self._quote_for(user, i)} for i, v in zip(ids, vals)][:3]
            return {"low": round(base - spread, 4), "base": round(base, 4), "high": round(base + spread, 4), "confidence": 0.5,
                    "claims": claims, "rationale": "mock analyst: mean of direct evidence"}
        if role == "judge":
            props = [float(x) for x in re.findall(r"proposed_value\W+(-?\d+\.\d+)", user)]
            if props:
                lo, hi = min(props), max(props)
                base = statistics.median(props) + rng.gauss(0, self.noise * temperature)
            elif vals:
                lo, hi = min(vals), max(vals)
                base = statistics.mean(vals)
            else:
                lo, base, hi = 0.0, 0.05, 0.10
            return {"low": round(min(lo, base), 4), "base": round(base, 4), "high": round(max(hi, base), 4),
                    "confidence": round(0.6 if len(vals) >= 3 else 0.4, 2),
                    "crux": [{"question": "mock: trailing growth vs guidance", "bull_position": "trend continues", "bear_position": "reverts",
                              "evidence_ids": ids[:2], "what_would_settle_it": "backlog conversion"}],
                    "key_disagreements": ["mock: weight of trailing growth vs guidance"],
                    "questions_for_management": ["mock: what is embedded in the guide?"],
                    "questions_for_internal_discussion": ["mock: model the backlog conversion"],
                    "implied_path": "mock", "rationale": "mock judge: median of advocate proposals", "evidence_used": ids[:6]}
        if not vals:
            return {"proposed_value": 0.05, "claims": [], "rebuttals": [], "questions_for_management": []}
        # The mandate direction in the system prompt decides the bias, not the role label.
        bias = -0.6 if "argue for the down-side" in system else +0.6
        mean = statistics.mean(vals)
        spread = (statistics.pstdev(vals) if len(vals) > 1 else abs(mean) * 0.1)
        prop = mean + bias * spread + rng.gauss(0, self.noise * temperature)
        claims = [{"text": f"Evidence {i} shows {v * 100:.1f}%, which supports my case", "evidence_ids": [i],
                   "quote": self._quote_for(user, i)} for i, v in zip(ids, vals)][:4]
        return {"proposed_value": round(prop, 4), "confidence": 0.6, "claims": claims, "rebuttals": [],
                "questions_for_management": [f"mock {role}: ask about durability of {ids[0]}"] if ids else []}

    @staticmethod
    def _quote_for(user: str, eid: str) -> str:
        for line in user.splitlines():
            if line.strip().startswith(f"[{eid}]"):
                body = line.split(")", 1)[-1].split("<source", 1)[0].strip()
                body = re.sub(r"\[FY[^\]]*\]\s*$", "", body).strip()
                return " ".join(body.split()[:12])
        return ""

    @staticmethod
    def _evidence(user: str) -> list[tuple[str, float, bool]]:
        out = []
        for line in user.splitlines():
            m = re.match(r"\s*\[(E\d+)\]\s*\((\w+)(?:,\s*direct)?\)\s*(.*)", line)
            if not m:
                continue
            eid, kind, rest = m.group(1), m.group(2), m.group(3)
            direct = ", direct)" in line.split(rest)[0] if rest else "direct" in line
            nums = _NUM.findall(rest)
            if nums:
                out.append((eid, float(nums[0]) / 100.0, direct))
        return out


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """``provider`` may be 'anthropic', 'openai', 'mock', or a spec 'provider:model' (e.g. 'anthropic:claude-opus-5-5')."""
    provider = (provider or os.environ.get("CRUCIBLE_PROVIDER", "mock")).lower()
    if ":" in provider:
        provider, model = provider.split(":", 1)
    if provider == "anthropic":
        return AnthropicLLM(model=model) if model else AnthropicLLM()
    if provider in ("openai", "openai-compat"):
        return OpenAICompatLLM(model=model) if model else OpenAICompatLLM()
    return MockLLM()

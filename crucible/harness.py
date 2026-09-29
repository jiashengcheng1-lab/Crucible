"""Robustness harness for assumption debates.

Three questions, three experiments:
  * stability: same packet, repeated runs -> does the base move?
  * permutation: same packet, shuffled evidence order -> position bias?
  * attribution: leave one evidence group out -> which information moves the answer?

Debates are independent API-bound jobs, so they run in a thread pool (``workers``) and every finished debate is
cached on disk keyed by its inputs, so an interrupted harness resumes instead of restarting.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .debate import PROMPT_VERSION, DebateResult, Estimate, Policy, estimate, run_debate
from .evidence import EvidencePacket
from .llm import LLM


class DebateCache:
    """logs/cache/<key>.json; key = hash of everything that determines a debate's inputs."""

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(packet: EvidencePacket, seed: int | None, order: list[str] | None, policy: str, rounds: int, temperature: float, models: dict) -> str:
        payload = json.dumps({"p": packet.hash(), "ids": packet.ids(), "seed": seed, "order": order, "policy": policy, "rounds": rounds,
                              "t": temperature, "models": models, "v": PROMPT_VERSION}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def get(self, key: str) -> DebateResult | None:
        if not self.root:
            return None
        p = self.root / f"{key}.json"
        if p.exists():
            try:
                return DebateResult.model_validate_json(p.read_text())
            except Exception:
                return None
        return None

    def put(self, key: str, result: DebateResult) -> None:
        if self.root:
            (self.root / f"{key}.json").write_text(result.model_dump_json())


def _models(llm: LLM, role_llms: dict | None) -> dict:
    return {r: getattr((role_llms or {}).get(r, llm), "name", "?") for r in ("bull", "bear", "judge")}


def run_debate_cached(packet: EvidencePacket, llm: LLM, seed: int, rounds: int, temperature: float, policy: Policy,
                      order: list[str] | None = None, role_llms: dict | None = None, cache: DebateCache | None = None) -> tuple[DebateResult, bool]:
    models = _models(llm, role_llms)
    k = DebateCache.key(packet, seed, order, policy, rounds, temperature, models) if cache else None
    if cache:
        hit = cache.get(k)
        if hit is not None:
            return hit, True
    res = run_debate(packet, llm, rounds=rounds, seed=seed, temperature=temperature, policy=policy, evidence_order=order, role_llms=role_llms)
    if cache:
        cache.put(k, res)
    return res, False


def run_jobs(jobs: list[tuple[str, Callable[[], tuple[DebateResult, bool]]]], workers: int = 1,
             progress: Callable[[str], None] | None = None) -> list[DebateResult]:
    """Run debate jobs concurrently, preserving order; ``progress`` gets one line per finished job."""
    results: list[DebateResult | None] = [None] * len(jobs)
    total = len(jobs)
    done = 0
    t0 = time.time()

    def _one(i: int):
        label, fn = jobs[i]
        res, cached = fn()
        return i, label, res, cached

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, label, res, cached in ex.map(_one, range(total)):
            results[i] = res
            done += 1
            if progress:
                progress(f"[{done}/{total}] {label}: base {res.verdict.base:.3f} ({'cached' if cached else f'{res.elapsed_s:.0f}s'}, {time.time() - t0:.0f}s elapsed)")
    return results  # type: ignore[return-value]


@dataclass
class StabilityReport:
    kind: str
    n: int
    bases: list[float]
    lows: list[float]
    highs: list[float]
    confidences: list[float]
    results: list[DebateResult] = field(default_factory=list)

    @property
    def base_median(self) -> float:
        return statistics.median(self.bases)

    @property
    def base_std(self) -> float:
        return statistics.pstdev(self.bases) if len(self.bases) > 1 else 0.0

    @property
    def mean_width(self) -> float:
        return statistics.mean(h - l for h, l in zip(self.highs, self.lows))

    def agreement_rate(self, tol: float = 0.01) -> float:
        """Share of runs whose base is within ``tol`` (absolute, in decimal units) of the median."""
        m = self.base_median
        return sum(1 for b in self.bases if abs(b - m) <= tol) / len(self.bases)

    def judge_lean(self) -> float | None:
        """Where the base sits between the final bear and bull proposals: 0 = at the bear, 1 = at the bull, median over runs."""
        pos = []
        for r in self.results:
            bull = [a for a in r.arguments if a.role == "bull"][-1].proposed_value
            bear = [a for a in r.arguments if a.role == "bear"][-1].proposed_value
            if bull == bull and bear == bear and bull != bear:
                pos.append((r.verdict.base - min(bull, bear)) / abs(bull - bear))
        return round(statistics.median(pos), 2) if pos else None

    def summary(self, tol: float = 0.01) -> dict:
        return {"kind": self.kind, "n": self.n, "base_median": round(self.base_median, 4), "base_std": round(self.base_std, 4),
                "base_min": round(min(self.bases), 4), "base_max": round(max(self.bases), 4),
                "mean_range_width": round(self.mean_width, 4), "agreement_rate": round(self.agreement_rate(tol), 2),
                "mean_confidence": round(statistics.mean(self.confidences), 2), "judge_lean_bear0_bull1": self.judge_lean()}


def _collect(kind: str, results: list[DebateResult]) -> StabilityReport:
    return StabilityReport(kind=kind, n=len(results), bases=[r.verdict.base for r in results], lows=[r.verdict.low for r in results],
                           highs=[r.verdict.high for r in results], confidences=[r.verdict.confidence for r in results], results=results)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def stability(packet: EvidencePacket, llm: LLM, n: int = 5, rounds: int = 2, temperature: float = 0.7, policy: Policy = "calibrated",
              seed0: int = 1, workers: int = 1, cache: DebateCache | None = None, role_llms: dict | None = None,
              progress: Callable[[str], None] | None = _stderr, label: str = "stability") -> StabilityReport:
    jobs = [(f"{label} seed {seed0 + i}", (lambda s=seed0 + i: run_debate_cached(packet, llm, s, rounds, temperature, policy, None, role_llms, cache)))
            for i in range(n)]
    return _collect("seed_stability", run_jobs(jobs, workers, progress))


def permutation(packet: EvidencePacket, llm: LLM, n: int = 5, rounds: int = 2, temperature: float = 0.7, policy: Policy = "calibrated",
                seed0: int = 1, workers: int = 1, cache: DebateCache | None = None, role_llms: dict | None = None,
                progress: Callable[[str], None] | None = _stderr) -> StabilityReport:
    ids = packet.ids()
    jobs = []
    for i in range(n):
        order = ids[:]
        random.Random(seed0 + i).shuffle(order)
        jobs.append((f"permutation seed {seed0 + i}", (lambda s=seed0 + i, o=order: run_debate_cached(packet, llm, s, rounds, temperature, policy, o, role_llms, cache))))
    return _collect("order_permutation", run_jobs(jobs, workers, progress))


def group_key(item) -> str:
    """Evidence groups an analyst would recognise: target history, target derived metrics, earnings releases, each peer, each 10-K section, analyst notes."""
    if item.kind == "peer_metric":
        return "peer: " + item.source.split(" 10-K")[0].split(" ")[0]
    if item.kind == "filing_text":
        return "earnings releases (8-K)" if item.source.startswith("8-K") else "filing text: " + item.source.split("10-K ")[-1]
    if item.kind == "derived_metric" and item.source.startswith("derived: guidance"):
        return "earnings releases (8-K)"
    if item.kind == "historical_metric":
        return "target history"
    if item.kind == "derived_metric":
        return "target derived metrics"
    return "analyst inputs"


def leave_one_out(packet: EvidencePacket, llm: LLM, k: int = 2, rounds: int = 2, temperature: float = 0.7, policy: Policy = "calibrated",
                  seed0: int = 1, full: StabilityReport | None = None, mode: str = "groups", workers: int = 1,
                  cache: DebateCache | None = None, role_llms: dict | None = None, progress: Callable[[str], None] | None = _stderr) -> pd.DataFrame:
    """Evidence attribution: rerun the debate without each group (default) or item, ``k`` seeds each, all jobs in one pool."""
    # Baseline must use exactly the same seeds as each ablation, otherwise seed noise masquerades as attribution.
    if full is None or full.n < k or [r.seed for r in full.results[:k]] != [seed0 + i for i in range(k)]:
        full = stability(packet, llm, n=k, rounds=rounds, temperature=temperature, policy=policy, seed0=seed0, workers=workers, cache=cache,
                         role_llms=role_llms, progress=progress, label="baseline")
    else:
        full = _collect(full.kind, full.results[:k])
    base_full, width_full = full.base_median, full.mean_width
    if mode == "groups":
        groups: dict[str, list] = {}
        for item in packet.items:
            groups.setdefault(group_key(item), []).append(item)
        units = list(groups.items())
    else:
        units = [(item.id, [item]) for item in packet.items]
    jobs = []
    for name, items in units:
        sub = packet
        for item in items:
            sub = sub.without(item.id)
        for i in range(k):
            jobs.append((f"without {name} seed {seed0 + i}", (lambda p=sub, s=seed0 + i: run_debate_cached(p, llm, s, rounds, temperature, policy, None, role_llms, cache))))
    results = run_jobs(jobs, workers, progress)
    rows = []
    for u, (name, items) in enumerate(units):
        rep = _collect("ablation", results[u * k:(u + 1) * k])
        rows.append({"evidence_id": name if mode == "groups" else items[0].id, "kind": items[0].kind if mode == "items" else f"{len(items)} items",
                     "source": items[0].source, "content": (items[0].content[:90] if mode == "items" else ", ".join(i.id for i in items)),
                     "base_full": base_full, "base_without": rep.base_median, "delta_base": rep.base_median - base_full,
                     "width_full": width_full, "width_without": rep.mean_width, "delta_width": rep.mean_width - width_full,
                     "confidence_without": statistics.mean(rep.confidences)})
    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta_base"].abs()
    return df.sort_values("abs_delta", ascending=False).drop(columns="abs_delta").reset_index(drop=True)


def pooled_noise_floor(*reports: StabilityReport) -> float:
    """2 x pooled std of the base across every full-packet run (stability and permutation together)."""
    bases = [b for r in reports for b in r.bases]
    return 2 * (statistics.pstdev(bases) if len(bases) > 1 else 0.0)


def attribution_summary(df: pd.DataFrame, noise_floor: float) -> dict:
    """Split evidence into 'moves the answer' vs 'noise'. Also flags systematic shrinkage: if every removal moves the
    base the same way, the judge is reacting to the amount of evidence, not its content, and single deltas overstate."""
    moves = df[df["delta_base"].abs() > max(noise_floor, 1e-9)]
    noise = df[df["delta_base"].abs() <= max(noise_floor, 1e-9)]
    signs = set(1 if d > 0 else -1 for d in df["delta_base"] if abs(d) > 1e-9)
    return {"noise_floor_abs": round(noise_floor, 4), "n_items": int(len(df)), "n_move_answer": int(len(moves)), "n_noise": int(len(noise)),
            "top_movers": moves[["evidence_id", "kind", "delta_base"]].head(5).to_dict("records"),
            "removable_without_effect": noise["evidence_id"].tolist(),
            "systematic_shift": (len(signs) == 1 and len(df) >= 4), "median_delta": round(float(df["delta_base"].median()), 4)}


# ----------------------------------------------------------------------------- control arm: independent estimates + vote

@dataclass
class VoteReport:
    n: int
    estimates: list[Estimate]

    @property
    def bases(self) -> list[float]:
        return [e.base for e in self.estimates]

    @property
    def base_median(self) -> float:
        return statistics.median(self.bases)

    @property
    def base_std(self) -> float:
        return statistics.pstdev(self.bases) if len(self.bases) > 1 else 0.0

    @property
    def low_median(self) -> float:
        return statistics.median(e.low for e in self.estimates)

    @property
    def high_median(self) -> float:
        return statistics.median(e.high for e in self.estimates)

    @property
    def mean_width(self) -> float:
        return statistics.mean(e.high - e.low for e in self.estimates)

    def evidence_ids(self) -> set[str]:
        return {i for e in self.estimates for c in e.claims for i in c.evidence_ids}

    def summary(self) -> dict:
        return {"kind": "vote", "n": self.n, "base_median": round(self.base_median, 4), "base_std": round(self.base_std, 4),
                "base_min": round(min(self.bases), 4), "base_max": round(max(self.bases), 4),
                "low_median": round(self.low_median, 4), "high_median": round(self.high_median, 4), "mean_range_width": round(self.mean_width, 4),
                "mean_confidence": round(statistics.mean(e.confidence for e in self.estimates), 2), "evidence_ids_cited": len(self.evidence_ids())}


def estimate_cached(packet: EvidencePacket, llm: LLM, seed: int, temperature: float, cache: DebateCache | None = None) -> tuple[Estimate, bool]:
    k = None
    if cache:
        payload = json.dumps({"arm": "vote", "p": packet.hash(), "ids": packet.ids(), "seed": seed, "t": temperature,
                              "model": getattr(llm, "name", "?"), "v": PROMPT_VERSION}, sort_keys=True)
        k = hashlib.sha256(payload.encode()).hexdigest()[:20]
        pth = cache.root / f"{k}.json"
        if pth.exists():
            try:
                return Estimate.model_validate_json(pth.read_text()), True
            except Exception:
                pass
    e = estimate(packet, llm, seed=seed, temperature=temperature)
    if cache and k:
        (cache.root / f"{k}.json").write_text(e.model_dump_json())
    return e, False


def vote(packet: EvidencePacket, llm: LLM, n: int = 5, temperature: float = 0.7, seed0: int = 1, workers: int = 1,
         cache: DebateCache | None = None, progress: Callable[[str], None] | None = _stderr) -> VoteReport:
    """N independent estimates, no debate, same evidence: the control (Choi et al. 2025 find voting explains most of
    debate's gains; Zhu et al. 2026 show homogeneous debate cannot reliably beat it). The harness reports whether
    the debate moved the assumption beyond this."""
    results: list[Estimate | None] = [None] * n
    t0 = time.time()

    def _one(i: int):
        e, cached = estimate_cached(packet, llm, seed0 + i, temperature, cache)
        return i, e, cached

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for done, (i, e, cached) in enumerate(ex.map(_one, range(n)), start=1):
            results[i] = e
            if progress:
                progress(f"[{done}/{n}] vote seed {seed0 + i}: base {e.base:.3f} ({'cached' if cached else 'fresh'}, {time.time() - t0:.0f}s elapsed)")
    return VoteReport(n=n, estimates=results)  # type: ignore[arg-type]


def compare_arms(debate: StabilityReport, votes: VoteReport, noise_floor: float) -> dict:
    """Did the debate change the assumption, the range, or the evidence used, beyond a plain vote?"""
    debate_ids = {i for r in debate.results for a in r.arguments for c in a.claims + a.rebuttals for i in c.evidence_ids}
    vote_ids = votes.evidence_ids()
    delta = debate.base_median - votes.base_median
    return {"debate_base_median": round(debate.base_median, 4), "vote_base_median": round(votes.base_median, 4), "delta_base": round(delta, 4),
            "noise_floor_abs": round(noise_floor, 4), "moved_beyond_noise": abs(delta) > noise_floor,
            "debate_width": round(debate.mean_width, 4), "vote_width": round(votes.mean_width, 4),
            "debate_evidence_ids": len(debate_ids), "vote_evidence_ids": len(vote_ids), "evidence_only_in_debate": sorted(debate_ids - vote_ids),
            "verdict": ("debate moved the base beyond the vote's noise" if abs(delta) > noise_floor else "debate did not move the base beyond a plain vote")
                       + (f"; debate cited {len(debate_ids - vote_ids)} evidence items the vote never used" if debate_ids - vote_ids else "; no extra evidence surfaced")}

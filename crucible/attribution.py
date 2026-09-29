"""Evidence attribution by regression instead of one debate per removal.

Two estimators, both recorded with their metrics:

* Random-subset design (``attribution_random``): each debate gets a random subset of evidence groups (each group in with
  probability p). Regressing the base on the inclusion indicators estimates every group's marginal effect at once, with a
  shared noise estimate, instead of k debates per group. Sixteen runs cover ten groups better than ten leave-one-out pairs.
* Fit on logs (``attribution_from_logs``): the cached runs (stability, permutation, ablations, backtests) already vary the
  packet. OLS of the base on group inclusion costs nothing; when backtest runs carry a realized outcome, a logistic fit of
  hit-or-miss on the same design says which groups made the range cover reality. Observational, so confounded by whatever
  else varied; the random design is the clean one.

Coefficients are in decimal growth units per group included; bootstrap intervals over runs.
"""
from __future__ import annotations

import json
import random
import statistics

import numpy as np
import pandas as pd

from .debate import DebateResult


def group_ids(packet) -> dict[str, list[str]]:
    from .harness import group_key

    out: dict[str, list[str]] = {}
    for it in packet.items:
        out.setdefault(group_key(it), []).append(it.id)
    return out


def design_random_subsets(groups: list[str], k: int, seed: int = 1, p: float = 0.5, always: tuple[str, ...] = ("target history",)) -> list[dict[str, int]]:
    """k inclusion vectors; groups in ``always`` are never dropped (a growth debate with no history is not the same question)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(k):
        row = {g: (1 if g in always else int(rng.random() < p)) for g in groups}
        if sum(row.values()) == 0:
            row[groups[0]] = 1
        rows.append(row)
    # make sure every optional group is both present and absent at least once
    for g in groups:
        if g in always:
            continue
        col = [r[g] for r in rows]
        if all(col):
            rows[0][g] = 0
        if not any(col):
            rows[-1][g] = 1
    return rows


def inclusion_matrix(results: list[DebateResult], groups: dict[str, list[str]]) -> np.ndarray:
    """Fraction of each group's ids present in each run's packet (1 = fully present, 0 = removed)."""
    X = np.zeros((len(results), len(groups)))
    for i, r in enumerate(results):
        ids = set(r.evidence_ids)
        for j, (g, members) in enumerate(groups.items()):
            X[i, j] = sum(1 for m in members if m in ids) / max(1, len(members))
    return X


def fit_ols(X: np.ndarray, y: np.ndarray, names: list[str], n_boot: int = 300, seed: int = 1) -> dict:
    """Least squares with an intercept; bootstrap-over-runs 90% intervals; R2 and n. Columns with no variation get NaN."""
    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 0]
    A = np.column_stack([np.ones(len(y)), X[:, keep]])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            b, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
            boots.append(b)
        except np.linalg.LinAlgError:
            continue
    B = np.array(boots) if boots else coef[None, :]
    lo, hi = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)
    rows = []
    for j, name in enumerate(names):
        if j in keep:
            pos = keep.index(j) + 1
            rows.append({"group": name, "effect": round(float(coef[pos]), 4), "ci90_low": round(float(lo[pos]), 4), "ci90_high": round(float(hi[pos]), 4),
                         "significant": bool(lo[pos] > 0 or hi[pos] < 0)})
        else:
            rows.append({"group": name, "effect": float("nan"), "ci90_low": float("nan"), "ci90_high": float("nan"), "significant": False})
    return {"model": "ols", "n": int(len(y)), "r2": round(float(r2), 3), "intercept": round(float(coef[0]), 4), "rows": rows,
            "resid_std": round(float(np.sqrt(ss_res / max(1, len(y) - A.shape[1]))), 4)}


def fit_logistic(X: np.ndarray, y: np.ndarray, names: list[str], ridge: float = 1e-2, iters: int = 30) -> dict:
    """Logistic regression by IRLS with a small ridge penalty (few runs, separable data). y in {0, 1}."""
    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 0]
    A = np.column_stack([np.ones(len(y)), X[:, keep]])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        z = A @ w
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1])
        g = A.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-6:
            break
    p = 1 / (1 + np.exp(-(A @ w)))
    ll = float((y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum())
    base_p = y.mean() if 0 < y.mean() < 1 else 0.5
    ll0 = float((y * np.log(base_p) + (1 - y) * np.log(1 - base_p)).sum())
    rows = []
    for j, name in enumerate(names):
        rows.append({"group": name, "log_odds": (round(float(w[keep.index(j) + 1]), 3) if j in keep else float("nan"))})
    return {"model": "logistic", "n": int(len(y)), "hit_rate": round(float(y.mean()), 3), "mcfadden_r2": round(1 - ll / ll0, 3) if ll0 < 0 else float("nan"),
            "intercept": round(float(w[0]), 3), "rows": rows}


def attribution_random(packet, llm, k: int = 16, rounds: int = 2, temperature: float = 0.7, policy: str = "calibrated", seed0: int = 1,
                       workers: int = 1, cache=None, role_llms=None, p: float = 0.5, progress=None) -> tuple[pd.DataFrame, dict, list[DebateResult]]:
    """Run k random-subset debates and fit OLS of the base on group inclusion."""
    from .harness import _stderr, run_debate_cached, run_jobs

    progress = _stderr if progress is None else progress
    groups = group_ids(packet)
    names = list(groups)
    design = design_random_subsets(names, k, seed=seed0, p=p)
    jobs = []
    for i, row in enumerate(design):
        sub = packet
        for g, inc in row.items():
            if not inc:
                for eid in groups[g]:
                    sub = sub.without(eid)
        kept = sum(row.values())
        jobs.append((f"random subset {i + 1}/{k} ({kept}/{len(names)} groups) seed {seed0 + i}",
                     (lambda pk=sub, s=seed0 + i: run_debate_cached(pk, llm, s, rounds, temperature, policy, None, role_llms, cache))))
    results = run_jobs(jobs, workers, progress)
    X = inclusion_matrix(results, groups)
    y = np.array([r.verdict.base for r in results])
    fit = fit_ols(X, y, names)
    fit["design"] = "random subsets"
    fit["p_include"] = p
    df = pd.DataFrame(fit["rows"]).sort_values("effect", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df, fit, results


def attribution_from_logs(runs: list[dict], packet, min_runs: int = 8) -> dict:
    """Fit on logged runs for this packet family (same company and assumption, prompt version, evidence-id space)."""
    groups = group_ids(packet)
    names = list(groups)
    if len(runs) < min_runs:
        return {"error": f"only {len(runs)} runs logged for this packet family; need {min_runs}"}
    X = np.zeros((len(runs), len(names)))
    y = np.zeros(len(runs))
    hits, errs = [], []
    for i, r in enumerate(runs):
        ids = set(r["evidence_ids"])
        for j, g in enumerate(names):
            X[i, j] = sum(1 for m in groups[g] if m in ids) / max(1, len(groups[g]))
        y[i] = r["verdict"]["base"]
        if r.get("realized") is not None:
            hits.append(1.0 if r["verdict"]["low"] <= r["realized"] <= r["verdict"]["high"] else 0.0)
            errs.append(r["realized"] - r["verdict"]["base"])
    out = {"n_runs": len(runs), "ols_base": fit_ols(X, y, names)}
    if len(hits) >= min_runs:
        mask = np.array([r.get("realized") is not None for r in runs])
        out["logistic_hit"] = fit_logistic(X[mask], np.array(hits), names)
        out["ols_error"] = fit_ols(X[mask], np.array(errs), names)
        out["note"] = "ols_error effect > 0 means including the group moved the base toward what was realized"
    return out


def attribution_table(fit: dict) -> str:
    rows = fit.get("rows", [])
    lines = [f"{'group':34s} {'effect':>8s} {'ci90 low':>9s} {'ci90 high':>9s}  sig"]
    for r in sorted(rows, key=lambda r: -abs(r["effect"]) if r["effect"] == r["effect"] else 0):
        e = r["effect"]
        lines.append(f"{r['group'][:34]:34s} {e:>+8.3f} {r['ci90_low']:>+9.3f} {r['ci90_high']:>+9.3f}  {'*' if r.get('significant') else ''}"
                     if e == e else f"{r['group'][:34]:34s} {'n/a':>8s}")
    lines.append(f"n {fit.get('n')} | R2 {fit.get('r2')} | residual std {fit.get('resid_std')} | intercept {fit.get('intercept')}")
    return "\n".join(lines)

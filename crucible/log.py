"""Append-only JSONL logs.

runs.jsonl        every debate run (inputs hash, model, outputs)          -> reproducibility
decisions.jsonl   every analyst accept / edit / reject with a reason     -> the dataset that compounds
outcomes.jsonl    realized values recorded later (backtest)              -> calibration
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .debate import DebateResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, rec: dict) -> None:
        with (self.root / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def record_run(self, result: DebateResult, tag: str = "", extra: dict | None = None) -> None:
        rec = {"ts": _now(), "tag": tag, **(extra or {}), **result.model_dump()}
        self._append("runs.jsonl", rec)

    def record_decision(self, company: str, assumption: str, packet_hash: str, tool_low: float, tool_base: float, tool_high: float,
                        action: str, analyst_value: float | None, reason: str, analyst: str = "analyst") -> dict:
        assert action in ("accept", "edit", "reject")
        rec = {"ts": _now(), "analyst": analyst, "company": company, "assumption": assumption, "packet_hash": packet_hash,
               "tool_low": tool_low, "tool_base": tool_base, "tool_high": tool_high, "action": action,
               "analyst_value": analyst_value if action != "accept" else tool_base, "reason": reason}
        self._append("decisions.jsonl", rec)
        return rec

    def record_outcome(self, company: str, assumption: str, period: str, realized: float, source: str) -> dict:
        rec = {"ts": _now(), "company": company, "assumption": assumption, "period": period, "realized": realized, "source": source}
        self._append("outcomes.jsonl", rec)
        return rec

    def read(self, name: str) -> list[dict]:
        p = self.root / name
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def decision_stats(self) -> dict:
        rows = self.read("decisions.jsonl")
        if not rows:
            return {"n": 0}
        n = len(rows)
        by_action = {a: sum(1 for r in rows if r["action"] == a) for a in ("accept", "edit", "reject")}
        edits = [r for r in rows if r["action"] == "edit" and r.get("analyst_value") is not None]
        mean_abs_edit = (sum(abs(r["analyst_value"] - r["tool_base"]) for r in edits) / len(edits)) if edits else 0.0
        in_range = sum(1 for r in edits if r["tool_low"] <= r["analyst_value"] <= r["tool_high"])
        return {"n": n, **by_action, "mean_abs_edit_vs_base": round(mean_abs_edit, 4),
                "edits_inside_tool_range": (round(in_range / len(edits), 2) if edits else None)}

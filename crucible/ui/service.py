"""Plain-Python services behind the UI: no Streamlit here, so every function is testable and the CLI stays the source of
truth. Long steps (ingest, map-suggest, choose, model) run the CLI in a subprocess and return its output; review actions
write the ledgers directly."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..decisions import DecisionLedger, review_frame
from ..ledger import MappingLedger

STEPS = [
    ("ingest", "Ingest filings (10-K, 10-Q, 8-K, proxy, XBRL)", ["ingest", "{T}", "--years", "7"]),
    ("map-suggest", "Propose mappings for unmapped lines (model)", ["map-suggest", "{T}", "--provider", "{P}", "--runs", "2"]),
    ("profile", "Profile the company from the filings (industry, drivers, KPIs)", ["profile", "{T}", "--provider", "{P}"]),
    ("choose", "Run the modeling decisions (cheap)", ["choose", "{T}", "--provider", "{P}", "--cheap"]),
    ("brief", "Write the brief (what it does, why now, the debate)", ["brief", "{T}", "--provider", "{P}"]),
    ("inputs", "Derive analyst inputs (rf, beta, ERP series, cost of debt, shares)", ["inputs", "{T}"]),
    ("model", "Build the three-statement model and DCF", ["model", "{T}", "--audit-links"]),
    ("kpis", "Fit the industry KPI table from releases", ["kpis", "{T}"]),
    ("quarterly", "Quarterly history", ["quarterly", "{T}"]),
    ("export", "Final workbook + unsure workbook", ["export", "{T}"]),
]


def tickers(data_dir: Path) -> list[str]:
    d = Path(data_dir)
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "meta.json").exists()) if d.exists() else []


def run_cli(args: list[str], cwd: Path | None = None, timeout: int = 3600) -> dict:
    """Run `python -m crucible <args>` and return code, stdout and stderr."""
    proc = subprocess.run([sys.executable, "-m", "crucible", *args], capture_output=True, text=True, cwd=str(cwd) if cwd else None, timeout=timeout)
    return {"code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": " ".join(["python", "-m", "crucible", *args])}


def step_args(step: str, ticker: str, provider: str, extra: list[str] | None = None) -> list[str]:
    for key, _, template in STEPS:
        if key == step:
            return [a.replace("{T}", ticker.upper()).replace("{P}", provider) for a in template] + list(extra or [])
    raise KeyError(step)


def pending(data_dir: Path, ticker: str, area: str, include_decided: bool = False) -> pd.DataFrame:
    ledger = MappingLedger(Path(data_dir)) if area == "mapping" else DecisionLedger(Path(data_dir))
    return review_frame(ticker, ledger, area, include_decided)


def decide(data_dir: Path, ticker: str, area: str, statement: str, source: str, status: str, target: str | None = None, note: str = "",
           analyst: str = "analyst") -> int:
    ledger = MappingLedger(Path(data_dir)) if area == "mapping" else DecisionLedger(Path(data_dir))
    n = ledger.decide(ticker, source, status, target=target or None, decided_by=analyst, note=note, statement=statement)
    ledger.save()
    return n


def counts(data_dir: Path, ticker: str) -> dict:
    out = {}
    for area, cls in (("mapping", MappingLedger), ("decisions", DecisionLedger)):
        d = cls(Path(data_dir)).rows(ticker)
        out[area] = d.status.value_counts().to_dict() if len(d) else {}
    return out


def ask_question(data_dir: Path, ticker: str, provider: str, row: dict | None, question: str, model: str | None = None, logs: Path = Path("logs")) -> dict:
    from ..ask import ask
    from ..evidence import EvidencePacket
    from ..ingest import load_sections
    from ..llm import get_llm

    sections = load_sections(Path(data_dir), ticker)
    pk = None
    dp = Path("outputs") / f"{ticker.upper()}_dossier.json"
    if dp.exists():
        try:
            pk = EvidencePacket.load(dp)
        except Exception:
            pk = None
    from ..state import state_items

    try:
        state = state_items(Path(data_dir), ticker)
    except Exception:
        state = []
    return ask(question, row, pk, sections, get_llm(provider, model), log_dir=Path(logs), ticker=ticker.upper(), state=state)


def qa_log(logs: Path, ticker: str, n: int = 50) -> list[dict]:
    p = Path(logs) / "qa.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("ticker") in ("", ticker.upper())][-n:]


def artifacts(ticker: str, outputs: Path = Path("outputs")) -> dict:
    """The files the UI shows or offers for download."""
    t = ticker.upper()
    o = Path(outputs)
    f = {"brief_md": o / f"{t}_brief.md", "profile_md": o / f"{t}_profile.md", "model": o / f"{t}_model.xlsx", "final": o / f"{t}_final.xlsx",
         "unsure": o / f"{t}_final_unsure.xlsx", "review": o / f"{t}_review.xlsx", "quarterly": o / f"{t}_quarterly.csv", "scenarios": o / f"{t}_scenarios.json",
         "elasticity": o / f"{t}_elasticity.json", "driver_build": o / f"{t}_driver_build.csv"}
    return {k: v for k, v in f.items() if v.exists()}


def debate_reviews(ticker: str, outputs: Path = Path("outputs")) -> list[Path]:
    return sorted(Path(outputs).glob(f"{ticker.upper()}_*_review.md"))


def statements(data_dir: Path, ticker: str) -> dict:
    """The three statements as mapped now (USD m) with the source of each line, and the forecast as last built if any."""
    from ..schema import STANDARD_ITEMS
    from ..state import statements_state

    st = statements_state(Path(data_dir), ticker)
    hist, reports = st["hist"], st["reports"]
    out = {}
    for code, name in (("IS", "Income statement"), ("BS", "Balance sheet"), ("CF", "Cash flow")):
        keys = [i.key for i in STANDARD_ITEMS if i.statement == code and i.key in hist.index]
        rep = reports.get(code)
        df = hist.loc[keys].copy()
        df.columns = [f"FY{c}" for c in df.columns]
        df.insert(0, "line", [next(i.label for i in STANDARD_ITEMS if i.key == k) for k in keys])
        df["source"] = [(rep.mapped.get(k, "") if rep else "") for k in keys]
        out[name] = df.round(1)
        resid = [k for k in hist.index if k not in {i.key for i in STANDARD_ITEMS}]
        if code == "CF" and resid:
            r = hist.loc[resid].copy()
            r.columns = [f"FY{c}" for c in r.columns]
            out["Residual lines (all statements)"] = r.round(1)
    fc = Path("outputs") / f"{ticker.upper()}_forecast.csv"
    if fc.exists():
        f = pd.read_csv(fc, index_col=0)
        f.columns = [f"FY{c}" for c in f.columns]
        out["Model as last built (history + forecast)"] = f.round(1)
    return out


def workbooks(ticker: str, outputs: Path = Path("outputs")) -> dict:
    arts = artifacts(ticker, outputs)
    return {k: v for k, v in arts.items() if str(v).endswith(".xlsx")}


def sheet_names(path: Path) -> list[str]:
    from ..state import sheet_names as _names

    return _names(Path(path))


def sheet_frame(path: Path, sheet: str) -> pd.DataFrame:
    from ..state import sheet_frame as _frame

    return _frame(Path(path), sheet)

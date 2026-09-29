"""Mapping ledger: one row per (ticker, statement, source) saying what it maps to in the schema and who decided.

This is the data structure that replaces hard-coding. Rules still produce the first guess, but every link a company
ends up with is written here with its confidence and the alternatives that were considered, and an analyst can accept,
reject or redirect any of them. Accepted rows override rules on every later run; rejected rows are never used again.

Columns
  ticker, statement (IS|BS|CF), source_type (concept|label), source (tag or caption), target (schema key | residual | ignore),
  confidence (0..1), alternatives (json: [{target, confidence}]), status (accepted|pending|rejected),
  proposed_by (rule:concept | rule:label | rule:sum | llm:<model> | analyst), evidence (accession / FY / value seen),
  decided_by, decided_at, note, version

System of record: data/mapping_ledger.csv (git-friendly). Analysts can also edit an Excel export and import it back.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEDGER_COLUMNS = ["ticker", "statement", "source_type", "source", "target", "confidence", "alternatives", "status", "proposed_by",
                  "evidence", "relation", "citation", "questions", "duplicates", "decided_by", "decided_at", "note", "version"]
KEY = ["ticker", "statement", "source_type", "source"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MappingLedger:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "mapping_ledger.csv"
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype=str).fillna("")
            for c in LEDGER_COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = ""
        else:
            self.df = pd.DataFrame(columns=LEDGER_COLUMNS)
        self._dirty = False

    # ------------------------------------------------------------------ queries
    def rows(self, ticker: str, statement: str | None = None, source_type: str | None = None, status: str | None = None) -> pd.DataFrame:
        d = self.df[self.df.ticker == ticker.upper()]
        if statement:
            d = d[d.statement == statement]
        if source_type:
            d = d[d.source_type == source_type]
        if status:
            d = d[d.status == status]
        return d

    def accepted(self, ticker: str, statement: str, source_type: str) -> dict[str, str]:
        """source -> target for accepted rows (targets 'residual' and 'ignore' are returned too; callers treat them as no-map)."""
        d = self.rows(ticker, statement, source_type, "accepted")
        return dict(zip(d.source, d.target))

    def rejected(self, ticker: str, statement: str, source_type: str) -> set[str]:
        return set(self.rows(ticker, statement, source_type, "rejected").source)

    def decided_links(self, ticker: str, statement: str, source_type: str) -> dict[str, str]:
        """source -> target for rows an analyst accepted (decided_by set). Only these override the rules: rule-recorded
        rows are a record of what happened, and feeding them back would let the ledger's row order reorder rule priority."""
        d = self.rows(ticker, statement, source_type, "accepted")
        d = d[d.decided_by.str.len() > 0]
        return dict(zip(d.source, d.target))

    def decided_rejects(self, ticker: str, statement: str, source_type: str) -> set[str]:
        d = self.rows(ticker, statement, source_type, "rejected")
        return set(d[d.decided_by.str.len() > 0].source)

    def analyst_decided(self, ticker: str, statement: str, source_type: str) -> set[str]:
        d = self.rows(ticker, statement, source_type)
        return set(d[d.decided_by.str.len() > 0].source)

    # ------------------------------------------------------------------ writes
    def upsert(self, ticker: str, statement: str, source_type: str, source: str, target: str, confidence: float, alternatives: list[dict] | None,
               status: str, proposed_by: str, evidence: str = "", decided_by: str = "", note: str = "", keep_decisions: bool = True,
               relation: str = "", citation: str = "", questions: str = "", duplicates: str = "") -> None:
        """Insert or update one row. With keep_decisions (default) a row an analyst has decided is left untouched."""
        ticker = ticker.upper()
        mask = (self.df.ticker == ticker) & (self.df.statement == statement) & (self.df.source_type == source_type) & (self.df.source == source)
        row = {"ticker": ticker, "statement": statement, "source_type": source_type, "source": source, "target": target,
               "confidence": f"{float(confidence):.2f}", "alternatives": json.dumps(alternatives or []), "status": status, "proposed_by": proposed_by,
               "evidence": evidence, "relation": relation, "citation": citation, "questions": questions, "duplicates": duplicates,
               "decided_by": decided_by, "decided_at": _now() if decided_by else "", "note": note, "version": "1"}
        if mask.any():
            idx = self.df.index[mask][0]
            if keep_decisions and str(self.df.at[idx, "decided_by"]):
                return
            same = all(str(self.df.at[idx, k]) == str(row[k]) for k in ("target", "status", "confidence", "proposed_by", "relation", "citation", "questions", "duplicates"))
            if same:
                return
            row["version"] = str(int(self.df.at[idx, "version"] or 1) + 1)
            for k, v in row.items():
                self.df.at[idx, k] = v
        else:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._dirty = True

    def decide(self, ticker: str, source: str, status: str, target: str | None = None, decided_by: str = "analyst", note: str = "",
               statement: str | None = None) -> int:
        """Analyst decision on a source (any statement unless given). Returns rows changed."""
        mask = (self.df.ticker == ticker.upper()) & (self.df.source == source)
        if statement:
            mask &= self.df.statement == statement
        n = int(mask.sum())
        if n == 0 and target and statement:  # a brand-new link typed by the analyst
            self.upsert(ticker, statement, "label" if not source.startswith(("us-gaap:", "dei:")) and ":" not in source else "concept", source, target,
                        1.0, [], status, "analyst", "", decided_by, note, keep_decisions=False)
            return 1
        for idx in self.df.index[mask]:
            self.df.at[idx, "status"] = status
            if target:
                self.df.at[idx, "target"] = target
            self.df.at[idx, "decided_by"] = decided_by
            self.df.at[idx, "decided_at"] = _now()
            if note:
                self.df.at[idx, "note"] = note
            self.df.at[idx, "version"] = str(int(self.df.at[idx, "version"] or 1) + 1)
        self._dirty = True
        return n

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df[LEDGER_COLUMNS].to_csv(self.path, index=False)
        self._dirty = False
        return self.path

    # ------------------------------------------------------------------ Excel round trip
    def to_excel(self, path: Path, ticker: str | None = None) -> Path:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation

        from .schema import STANDARD_ITEMS

        d = self.rows(ticker) if ticker else self.df
        d = d.sort_values(["ticker", "statement", "status", "confidence"], ascending=[True, True, True, False])
        wb = Workbook()
        ws = wb.active
        ws.title = "mapping_ledger"
        ws.append(LEDGER_COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        fill = {"pending": PatternFill("solid", fgColor="FFF2CC"), "rejected": PatternFill("solid", fgColor="F8CBAD"), "accepted": PatternFill("solid", fgColor="E2EFDA")}
        for _, r in d.iterrows():
            vals = []
            for c in LEDGER_COLUMNS:
                v = r.get(c, "")
                if c == "confidence":
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = None
                elif c == "version":
                    try:
                        v = int(float(v))
                    except (TypeError, ValueError):
                        v = None
                vals.append(v)
            ws.append(vals)
            for cell in ws[ws.max_row]:
                cell.fill = fill.get(str(r.get("status", "")), fill["accepted"])
            ws.cell(row=ws.max_row, column=LEDGER_COLUMNS.index("confidence") + 1).number_format = "0.00"
        lists = wb.create_sheet("lists")
        keys = [it.key for it in STANDARD_ITEMS] + ["residual", "ignore"]
        for i, k in enumerate(keys, start=1):
            lists.cell(row=i, column=1, value=k)
        for i, st in enumerate(("accepted", "pending", "rejected"), start=1):
            lists.cell(row=i, column=2, value=st)
        lists.sheet_state = "hidden"
        dv_t = DataValidation(type="list", formula1=f"=lists!$A$1:$A${len(keys)}", allow_blank=True)
        dv_s = DataValidation(type="list", formula1="=lists!$B$1:$B$3", allow_blank=True)
        ws.add_data_validation(dv_t)
        ws.add_data_validation(dv_s)
        col_t, col_s = LEDGER_COLUMNS.index("target") + 1, LEDGER_COLUMNS.index("status") + 1
        for row in range(2, max(3, ws.max_row + 1)):
            dv_t.add(ws.cell(row=row, column=col_t))
            dv_s.add(ws.cell(row=row, column=col_s))
        ws.freeze_panes = "A2"
        for col, width in (("A", 8), ("B", 6), ("C", 10), ("D", 56), ("E", 22), ("F", 10), ("G", 34), ("H", 10), ("I", 16), ("J", 36), ("K", 60), ("L", 70),
                           ("M", 50), ("N", 44), ("O", 14), ("P", 20), ("Q", 36), ("R", 8)):
            ws.column_dimensions[col].width = width
        readme = wb.create_sheet("README")
        readme["A1"] = "Edit target and status (dropdowns), add a note, then: crucible map-approve TICKER --file <this file>. Yellow = pending LLM proposals. Do not edit source."
        readme["A3"] = "evidence: values seen and share of revenue/assets. relation: what the line is and what drives it. citation: accession | section | verbatim filing sentence(s)."
        readme["A4"] = "questions: what must be settled to categorize the line. duplicates: other lines this one may double count (same value, or a total containing it)."
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
        return Path(path)

    def from_excel(self, path: Path, decided_by: str = "analyst (excel)") -> int:
        """Apply status/target/note edits from an exported sheet. Returns rows changed."""
        from openpyxl import load_workbook

        ws = load_workbook(path, data_only=True)["mapping_ledger"]
        headers = [c.value for c in ws[1]]
        changed = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            r = dict(zip(headers, row))
            if not r.get("source"):
                continue
            mask = ((self.df.ticker == str(r["ticker"]).upper()) & (self.df.statement == r["statement"]) & (self.df.source_type == r["source_type"])
                    & (self.df.source == str(r["source"])))
            if not mask.any():
                if r.get("target") and r.get("status") == "accepted":
                    self.upsert(str(r["ticker"]), r["statement"], r["source_type"], str(r["source"]), str(r["target"]), 1.0, [], "accepted", "analyst", "",
                                decided_by, str(r.get("note") or ""), keep_decisions=False)
                    changed += 1
                continue
            idx = self.df.index[mask][0]
            new_status, new_target, new_note = str(r.get("status") or ""), str(r.get("target") or ""), str(r.get("note") or "")
            if new_status != self.df.at[idx, "status"] or new_target != self.df.at[idx, "target"] or (new_note and new_note != self.df.at[idx, "note"]):
                self.df.at[idx, "status"], self.df.at[idx, "target"] = new_status or self.df.at[idx, "status"], new_target or self.df.at[idx, "target"]
                if new_note:
                    self.df.at[idx, "note"] = new_note
                self.df.at[idx, "decided_by"], self.df.at[idx, "decided_at"] = decided_by, _now()
                self.df.at[idx, "version"] = str(int(self.df.at[idx, "version"] or 1) + 1)
                changed += 1
        self._dirty = changed > 0
        return changed

    def summary(self, ticker: str) -> str:
        d = self.rows(ticker)
        if not len(d):
            return f"{ticker.upper()}: ledger empty (run `crucible model {ticker}` to record rule matches, `crucible map-suggest` for proposals)"
        by = d.groupby(["status", "proposed_by"]).size().to_dict()
        lines = [f"{ticker.upper()}: {len(d)} ledger rows: " + ", ".join(f"{s}/{p}: {n}" for (s, p), n in sorted(by.items()))]
        pend = d[d.status == "pending"]
        for _, r in pend.head(15).iterrows():
            lines.append(f"  pending  {r.statement} {r.source[:60]:60s} -> {r.target:22s} conf {r.confidence}  alts {r.alternatives[:60]}")
        return "\n".join(lines)

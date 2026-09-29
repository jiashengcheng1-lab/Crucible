"""Standard line items and the rules that map filing data onto them.

Two sources feed the schema, in this order:
  1. concept-level XBRL facts (``facts.py``): exact us-gaap concept names, per-year
     coalescing in the order listed, with optional sum-of-components fallbacks;
  2. edgartools' standardized statements (``mapping.py``): exact standard labels,
     then regex patterns with exclusions.

Design rule: the LLM never invents numbers. Anything that does not match is
reported so an analyst, or the LLM mapping step, can resolve it explicitly via
``data/<TICKER>/mapping_overrides.json``.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Item:
    key: str
    statement: str  # IS | BS | CF
    label: str
    concepts: tuple[str, ...] = ()          # us-gaap local names, coalesced per year in this order
    sum_concepts: tuple[str, ...] = ()      # if no concept above has a value for a year, sum these components
    neg_concepts: tuple[str, ...] = ()      # concepts reported with the opposite sign convention (value is negated)
    exact_labels: tuple[str, ...] = ()      # edgartools standardized labels (case-insensitive exact)
    labels: tuple[str, ...] = ()            # regex fallbacks on the lowercased label
    exclude: str = ""                        # regex; a label matching this never maps here
    sign: str = "as_reported"                # "as_reported" | "abs" (store magnitude)
    in_model: bool = True                    # False = memo item kept in hist but not forecast


RULES_PATH = Path(os.environ.get("CRUCIBLE_SCHEMA_RULES", Path(__file__).parent / "rules" / "schema_rules.csv"))


LIST_SEP = ";;"  # regex patterns contain '|', so lists in the CSV use ';;'


def _split(v: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(v or "").split(LIST_SEP) if x.strip())


def load_rules(path: Path = RULES_PATH) -> list[Item]:
    """The mapping rules live in a CSV an analyst can edit (CRUCIBLE_SCHEMA_RULES points at a custom copy).
    Columns: key, statement, label, concepts, sum_concepts, neg_concepts, exact_labels, regex_labels, exclude, sign, in_model.
    Order matters: the first matching rule wins, so specific totals come before partial lines."""
    items: list[Item] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("key"):
                continue
            items.append(Item(key=r["key"].strip(), statement=r["statement"].strip(), label=r.get("label", "").strip(),
                              concepts=_split(r.get("concepts")), sum_concepts=_split(r.get("sum_concepts")), neg_concepts=_split(r.get("neg_concepts")),
                              exact_labels=_split(r.get("exact_labels")), labels=_split(r.get("regex_labels")), exclude=(r.get("exclude") or "").strip(),
                              sign=(r.get("sign") or "as_reported").strip(), in_model=str(r.get("in_model", "1")).strip() not in ("0", "false", "False", "")))
    return items


STANDARD_ITEMS: list[Item] = load_rules()

ITEM_BY_KEY: dict[str, Item] = {it.key: it for it in STANDARD_ITEMS}

# Derived (residual) items computed after mapping so historical totals tie exactly.
DERIVED_ITEMS: list[tuple[str, str, str]] = [
    ("other_opex", "IS", "Other operating expense (residual)"),
    ("other_income", "IS", "Other income/(expense), net (residual)"),
    ("ebitda", "IS", "EBITDA (memo)"),
    ("nci_and_other", "IS", "NCI, discontinued ops & other below tax (residual, memo)"),
    ("other_current_assets", "BS", "Other current assets (residual)"),
    ("other_noncurrent_assets", "BS", "Other non-current assets (residual)"),
    ("other_current_liabilities", "BS", "Other current liabilities (residual)"),
    ("other_noncurrent_liabilities", "BS", "Other non-current liabilities (residual)"),
    ("d_nwc", "CF", "Change in NWC & other (residual)"),
    ("other_investing", "CF", "Other investing (residual)"),
    ("net_debt_issuance", "CF", "Net debt issued/(repaid) & other financing (residual)"),
]

_concept_index: dict[str, str] = {}
for _it in STANDARD_ITEMS:
    for _c in _it.concepts + _it.sum_concepts + _it.neg_concepts:
        _concept_index.setdefault(_c.lower(), _it.key)


def _local_name(concept: str) -> str:
    c = str(concept or "")
    for sep in (":", "_"):
        if sep in c:
            c = c.split(sep)[-1]
    return c.strip().lower()


def norm_label(label) -> str:
    return re.sub(r"\s+", " ", str(label or "").strip().lower()).replace("\u2019", "'")


CONFIDENCE = {"concept": 0.95, "exact_label": 0.85, "regex_label": 0.60, "sum": 0.70, "ledger": 1.00}


def match_candidates(label: str | None, concept: str | None, statement: str) -> list[tuple[str, float, str]]:
    """Every rule that fires for a filing row, as (key, confidence, how), best first. The first entry is the mapping;
    the rest are the alternatives recorded in the ledger. Restricted to the statement and honouring ``exclude``."""
    out: list[tuple[str, float, str]] = []
    if concept:
        key = _concept_index.get(_local_name(concept))
        if key and ITEM_BY_KEY[key].statement == statement:
            out.append((key, CONFIDENCE["concept"], "rule:concept"))
    lab = norm_label(label)
    if lab:
        for it in items_for(statement):
            if lab in it.exact_labels and not (it.exclude and re.search(it.exclude, lab)) and it.key not in [k for k, _, _ in out]:
                out.append((it.key, CONFIDENCE["exact_label"], "rule:label"))
        for it in items_for(statement):
            if it.exclude and re.search(it.exclude, lab):
                continue
            if any(re.search(pat, lab) for pat in it.labels) and it.key not in [k for k, _, _ in out]:
                out.append((it.key, CONFIDENCE["regex_label"], "rule:label"))
    return out


def match_row(label: str | None, concept: str | None, statement: str) -> str | None:
    """Standard key for a filing row, or None: the best candidate."""
    c = match_candidates(label, concept, statement)
    return c[0][0] if c else None


def items_for(statement: str) -> list[Item]:
    return [it for it in STANDARD_ITEMS if it.statement == statement]


@dataclass
class MappingReport:
    statement: str
    mapped: dict[str, str] = field(default_factory=dict)          # key -> source description
    unmapped: list[str] = field(default_factory=list)              # statement labels with numbers that matched nothing
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    residuals: dict[str, float] = field(default_factory=dict)      # residual key -> max abs share of its base
    missing: list[str] = field(default_factory=list)               # model keys with no data at all
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"[{self.statement}] mapped {len(self.mapped)} | missing {len(self.missing)} | unmapped labels {len(self.unmapped)}"]
        if self.missing:
            lines.append("  missing (filled with 0): " + ", ".join(self.missing))
        for k, v in self.residuals.items():
            flag = "  <-- check" if v > 0.10 else ""
            lines.append(f"  residual {k}: {v:.1%} of base{flag}")
        if self.unmapped:
            lines.append("  unmapped labels: " + "; ".join(self.unmapped[:20]) + (" ..." if len(self.unmapped) > 20 else ""))
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)

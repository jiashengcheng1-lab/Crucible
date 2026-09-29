import pandas as pd

from crucible.evidence import long_history_evidence
from crucible.synthetic import synthetic_history


def test_long_history_item_summarizes_all_years_and_skips_gaps():
    h = synthetic_history()  # FY2021..FY2025
    items = long_history_evidence(h, "Test Co", "10-K")
    assert len(items) == 1 and "FY2022 14.0%" in items[0].content and "CAGR FY2021-FY2025" in items[0].content and "nan" not in items[0].content
    gap = h.drop(columns=[2023])  # a missing year breaks pairs; with fewer than three usable years the item is omitted
    assert long_history_evidence(gap, "Test Co", "10-K") == []

from crucible.evidence import build_packet, guidance_track_record, parse_guidance_table
from crucible.harness import DebateCache, compare_arms, pooled_noise_floor, stability, vote
from crucible.llm import MockLLM
from crucible.synthetic import synthetic_history


def test_vote_arm_and_comparison(tmp_path):
    pk = build_packet("revenue_growth_3y", "Test Co", synthetic_history())
    llm = MockLLM()
    st = stability(pk, llm, n=3, workers=2, cache=DebateCache(tmp_path / "c"), progress=None)
    vt = vote(pk, llm, n=4, workers=2, cache=DebateCache(tmp_path / "c"), progress=None)
    assert vt.n == 4 and all(e.claims for e in vt.estimates)
    arms = compare_arms(st, vt, pooled_noise_floor(st))
    assert set(arms) >= {"debate_base_median", "vote_base_median", "delta_base", "moved_beyond_noise", "verdict"}
    again = vote(pk, llm, n=4, workers=1, cache=DebateCache(tmp_path / "c"), progress=None)
    assert again.bases == vt.bases  # cached


def test_guidance_table_two_column_layout_takes_full_year_column():
    fy, sales, organic = parse_guidance_table(
        "First Quarter 2022 Guidance Full Year 2022 Guidance Net sales $1,100M - $1,150M $5,500M - $5,800M "
        "Organic net sales growth(2) (4%) – 1% 5.0% - 11.0% Adjusted operating profit -$30M) – ($10M $500M - $550M")
    assert fy == 2022 and sales == (5500.0, 5800.0) and organic == (0.05, 0.11)


def test_guidance_track_record_compares_initial_guides_with_actuals():
    hist = synthetic_history()  # FY2021..FY2025, growth 14.0/20.0/17.0/15.0 for FY2022..FY2025
    releases = [
        {"filing_date": "2023-02-20", "text": "Full Year 2023 Guidance Net sales $5,900M - $6,100M Organic net sales growth 10% - 12%"},
        {"filing_date": "2024-02-20", "text": "We anticipate 2024 net sales growth of 20%, supported by our backlog."},
        {"filing_date": "2024-08-01", "text": "We now expect 2024 net sales growth of 25%."},   # updated, must not replace the initial guide
    ]
    items = guidance_track_record(releases, hist, "Test Co")
    assert len(items) == 1
    c = items[0].content
    assert "FY2023 initial guide 11.0% vs actual 20.0%" in c and "FY2024 initial guide 20.0% vs actual 17.0%" in c
    assert "beat the guide in 1 of 2 years" in c

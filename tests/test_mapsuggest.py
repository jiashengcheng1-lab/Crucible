from crucible.mapsuggest import MockMapper, context_sentences, merge_runs, suggest, value_duplicates
from crucible.synthetic import synthetic_history


def test_context_sentences_prefer_notes_and_match_distinctive_words():
    sections = {"Item 1A Risk Factors": "Energy prices may rise. Our operations depend on power.",
                "Item 8 Notes": "Purchased energy costs consist of electricity purchased for our owned mining sites and are recognized as incurred. Unrelated sentence about leases here."}
    cs = context_sentences(sections, "mara:PurchasedEnergyCosts")
    assert cs and cs[0]["section"].startswith("Item 8") and "electricity" in cs[0]["sentence"]


def test_merge_runs_reports_agreement_and_keeps_disagreement_as_alternative():
    a = [{"statement": "IS", "source": "x", "target": "cogs", "confidence": 0.8, "alternatives": [], "questions": ["q1"], "duplicates": []},
         {"statement": "IS", "source": "y", "target": "residual", "confidence": 0.6, "alternatives": [], "questions": [], "duplicates": []}]
    b = [{"statement": "IS", "source": "x", "target": "cogs", "confidence": 0.6, "alternatives": [], "questions": ["q2"], "duplicates": []},
         {"statement": "IS", "source": "y", "target": "sga", "confidence": 0.5, "alternatives": [], "questions": [], "duplicates": []}]
    merged, agr = merge_runs([a, b])
    m = {p["source"]: p for p in merged}
    assert agr["agreement_rate"] == 0.5 and m["x"]["confidence"] == 0.7 and m["x"]["questions"] == ["q1", "q2"]
    assert m["y"]["target"] == "residual" and any(alt["target"] == "sga" for alt in m["y"]["alternatives"]) and m["y"]["confidence"] < 0.6


def test_value_duplicates_flag_equal_values():
    hist = synthetic_history()
    rev = float(hist.loc["revenue", 2025]) * 1e6
    cands = [{"statement": "IS", "source": "Total revenue (dup)", "values": {"2025": rev}},
             {"statement": "IS", "source": "Something else", "values": {"2025": 123456.0}}]
    vd = value_duplicates(cands, hist)
    assert "Total revenue (dup)" in vd and "Something else" not in vd


def test_suggest_with_mock_fills_new_fields():
    cands = [{"statement": "IS", "source_type": "concept", "source": "mara:PurchasedEnergyCosts", "values": {"2025": 1.0}, "share_of_base": 0.3,
              "context": [{"section": "Item 8 Notes", "sentence": "Purchased energy costs consist of electricity."}]}]
    props = suggest(cands, MockMapper(), "MARA")
    assert props[0]["target"] == "cogs" and props[0]["citations"][0]["sentence"].startswith("Purchased") and props[0]["questions"]


def test_merge_takes_relation_and_confirmed_citations_from_any_run():
    a = [{"statement": "IS", "source": "x", "target": "cogs", "confidence": 0.8, "alternatives": [], "questions": [], "duplicates": [],
          "relation": "(from rationale) something", "citations": [{"section": "Item 7 MD&A", "sentence": "auto pick"}], "citation_auto": True}]
    b = [{"statement": "IS", "source": "x", "target": "cogs", "confidence": 0.6, "alternatives": [], "questions": [], "duplicates": [],
          "relation": "Electricity for owned sites.", "citations": [{"section": "Item 8 Notes", "sentence": "Purchased energy consists of electricity."}], "citation_auto": False}]
    merged, _ = merge_runs([a, b])
    m = merged[0]
    assert m["relation"] == "Electricity for owned sites." and m["citations"][0]["section"].startswith("Item 8") and not m["citation_auto"]

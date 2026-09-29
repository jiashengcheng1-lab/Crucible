from crucible.debate import Claim, validate_claims
from crucible.evidence import EvidenceItem, EvidencePacket, ASSUMPTIONS, guidance_evidence

RELEASE = {"filing_date": "2026-02-11", "date_of_report": "2026-02-11", "text": (
    "Vertiv Reports Strong Fourth Quarter with Organic Orders Growth of 252%. Full Year 2026: Expect net sales of "
    "$13,250 to $13,750 million, with organic sales growth of 27% to 29% compared to 2025. Fourth quarter 2025 "
    "book-to-bill ratio was ~2.9x and backlog increased to $15.0 billion, up 109% compared to the same period last year. "
    "Liquidity remained strong at $2.6 billion.")}


def test_guidance_evidence_extracts_guide_orders_backlog_and_implied_growth():
    items = guidance_evidence([RELEASE], "Vertiv", 10229.9)
    texts = " ".join(i.content for i in items)
    assert "13,250" in texts and "book-to-bill" in texts
    implied = [i for i in items if i.source.startswith("derived: guidance midpoint")]
    stated = [i for i in items if i.source.startswith("derived: guidance stated")]
    assert len(implied) == 1 and implied[0].direct and abs(implied[0].value - 0.320) < 0.002
    assert len(stated) == 1 and abs(stated[0].value - 0.28) < 0.001
    assert all(i.source.startswith("8-K") or i.source.startswith("derived: guidance") for i in items)


def test_validator_accepts_arithmetic_on_cited_numbers_and_form_names():
    pk = EvidencePacket(company="X", assumption=ASSUMPTIONS["revenue_growth_3y"], items=[
        EvidenceItem(id="E1", kind="derived_metric", source="s", content="gross margin FY2022 28.4%, FY2025 36.3%"),
        EvidenceItem(id="E2", kind="filing_text", source="s", content='"backlog $15.0 billion, shipped within 12 to 18 months"'),
    ])
    kept, dropped, _ = validate_claims([
        Claim(text="An 8-point margin improvement (28.4% to 36.3%)", evidence_ids=["E1"]),   # difference of cited numbers
        Claim(text="Boilerplate common to all 10-Ks; backlog $15.0B ships in 12-18 months over 3 years", evidence_ids=["E2"]),
        Claim(text="Margins improved 20 points", evidence_ids=["E1"]),                        # not supported
    ], pk, require_quote=False)
    assert len(kept) == 2 and len(dropped) == 1


FEB_2023 = {"filing_date": "2023-02-22", "date_of_report": "2023-02-22", "text": (
    "Vertiv Reports Record Fourth Quarter 2022 Net Sales\n• Record high backlog of $4.8 billion provides good visibility into "
    "top-line growth projections for 2023\n• Fourth quarter financial performance provides momentum for strong 2023. Expect 2023 net "
    "sales growth of 15%, operating profit of $568 million to $618 million COLUMBUS, Ohio [February 22, 2023] – Vertiv Holdings Co (NYSE: "
    "VRT) today reported results\nWe anticipate 2023 organic net sales growth of 15%, supported by our backlog.\n"
    "First Quarter 2023 Guidance   Net sales $1,350M - $1,450M   Organic net sales growth(2) 21% - 29%\n"
    "Full Year 2023 Guidance   Net sales $6,450M - $6,600M   Organic net sales growth(2) 14% - 17%   Adjusted operating profit $750M - $800M")}


def test_guidance_extraction_handles_bullets_tables_and_stated_growth():
    items = guidance_evidence([FEB_2023], "Vertiv", 5691.5)
    texts = " ".join(i.content for i in items)
    assert "anticipate 2023 organic net sales growth of 15%" in texts
    table = [i for i in items if i.source.startswith("derived: guidance table")]
    implied = [i for i in items if i.source.startswith("derived: guidance midpoint")]
    assert table and abs(table[0].value - 0.155) < 0.001
    assert implied and abs(implied[0].value - 0.146) < 0.002   # 6,525 / 5,691.5 - 1
    assert not any("1,350" in i.content for i in items if i.kind == "derived_metric")  # quarterly guide never becomes the full-year derivation

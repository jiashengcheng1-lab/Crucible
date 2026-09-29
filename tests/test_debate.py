import pytest

from crucible.debate import Claim, run_debate, validate_claims
from crucible.evidence import build_packet
from crucible.llm import MockLLM
from crucible.synthetic import synthetic_history


@pytest.fixture(scope="module")
def packet():
    return build_packet("revenue_growth_3y", "Test Co", synthetic_history())


def test_validator_drops_uncited_and_fabricated_numbers(packet):
    kept, dropped, completed = validate_claims([
        Claim(text="Growth was 20.0% in FY2023", evidence_ids=["E2"]),          # number present in E2
        Claim(text="Growth was 0.20 in FY2023", evidence_ids=["E2"]),           # decimal equivalent accepted
        Claim(text="Management guided 40% growth", evidence_ids=["E2"]),         # 40 nowhere in the packet -> drop
        Claim(text="Peers are growing faster", evidence_ids=["E99"]),            # unknown id -> drop
        Claim(text="Margins expanded", evidence_ids=[]),                         # uncited -> drop
        Claim(text="FY2024 growth was 17.0%", evidence_ids=["E2"]),              # 17.0 lives in E3 -> citation completed
    ], packet, require_quote=False)
    assert len(kept) == 3 and len(dropped) == 3 and len(completed) == 1
    assert "E3" in kept[2].evidence_ids


def test_quote_gate_drops_claims_without_a_verbatim_span(packet):
    e2 = packet.get("E2").content
    kept, dropped, _ = validate_claims([
        Claim(text="Growth was 20.0% in FY2023", evidence_ids=["E2"], quote=e2[:40]),                 # verbatim -> keep
        Claim(text="Growth was 20.0% in FY2023", evidence_ids=["E2"], quote="growth of twenty percent"),  # paraphrase -> drop
        Claim(text="Growth was 20.0% in FY2023", evidence_ids=["E2"], quote=""),                       # missing -> drop
    ], packet)
    assert len(kept) == 1 and len(dropped) == 2


def test_debate_returns_ordered_range_within_bounds(packet):
    res = run_debate(packet, MockLLM(), rounds=2, seed=3)
    v = res.verdict
    assert v.low <= v.base <= v.high
    assert packet.assumption.lower_bound <= v.base <= packet.assumption.upper_bound
    assert len(res.arguments) == 4 and res.packet_hash == packet.hash()
    assert all(a.claims for a in res.arguments)
    assert v.questions_for_management


def test_bull_and_bear_pull_in_opposite_directions(packet):
    res = run_debate(packet, MockLLM(), rounds=1, seed=1)
    bull = next(a for a in res.arguments if a.role == "bull")
    bear = next(a for a in res.arguments if a.role == "bear")
    assert bull.proposed_value > bear.proposed_value


def test_wacc_direction_is_inverted():
    pk = build_packet("wacc", "Test Co", synthetic_history(), analyst={"risk_free": 0.04, "erp": 0.05, "beta": 1.1})
    res = run_debate(pk, MockLLM(), rounds=1, seed=1)
    bull = next(a for a in res.arguments if a.role == "bull")
    bear = next(a for a in res.arguments if a.role == "bear")
    assert bull.proposed_value < bear.proposed_value

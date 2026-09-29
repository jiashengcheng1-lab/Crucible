from crucible.evidence import build_packet
from crucible.harness import attribution_summary, leave_one_out, permutation, stability
from crucible.llm import MockLLM
from crucible.synthetic import synthetic_history


def test_stability_and_permutation_report_shapes():
    pk = build_packet("revenue_growth_3y", "Test Co", synthetic_history())
    st = stability(pk, MockLLM(), n=3)
    pm = permutation(pk, MockLLM(), n=3)
    for rep in (st, pm):
        s = rep.summary()
        assert s["n"] == 3 and 0 <= s["agreement_rate"] <= 1 and s["mean_range_width"] >= 0


def test_leave_one_out_attributes_only_to_evidence_that_matters():
    pk = build_packet("revenue_growth_3y", "Test Co", synthetic_history())
    st = stability(pk, MockLLM(noise=0.0), n=2)
    att = leave_one_out(pk, MockLLM(noise=0.0), k=2, full=st, mode="items")
    direct = {i.id for i in pk.items if i.direct}
    moved = set(att.loc[att["delta_base"].abs() > 1e-9, "evidence_id"])
    assert moved <= direct, "non-direct items must not move the mock's answer"
    assert moved, "removing a direct growth observation must move the answer"
    summ = attribution_summary(att, noise_floor=0.0)
    assert summ["n_move_answer"] + summ["n_noise"] == len(pk.items)


def test_grouped_ablation_moves_only_the_history_group():
    pk = build_packet("revenue_growth_3y", "Test Co", synthetic_history())
    st = stability(pk, MockLLM(noise=0.0), n=2)
    att = leave_one_out(pk, MockLLM(noise=0.0), k=2, full=st, mode="groups")
    moved = set(att.loc[att["delta_base"].abs() > 1e-9, "evidence_id"])
    assert moved == {"target history"}
    assert len(att) == 2  # target history + target derived metrics

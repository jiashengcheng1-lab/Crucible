from crucible.evidence import build_packet
from crucible.harness import DebateCache, stability
from crucible.llm import MockLLM
from crucible.synthetic import synthetic_history


def test_cache_and_workers(tmp_path):
    pk = build_packet("revenue_growth_3y", "Test Co", synthetic_history())
    cache = DebateCache(tmp_path / "cache")
    seen = []
    a = stability(pk, MockLLM(), n=3, workers=3, cache=cache, progress=seen.append)
    b = stability(pk, MockLLM(), n=3, workers=1, cache=cache, progress=seen.append)
    assert a.bases == b.bases
    assert sum("cached" in m for m in seen) == 3 and len(list((tmp_path / "cache").glob("*.json"))) == 3

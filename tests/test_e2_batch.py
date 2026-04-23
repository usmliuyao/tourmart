"""Batched episode runner tests (using MockLLM)."""
from __future__ import annotations

import json

from tourmart.e2_batch import EpisodeSpec, run_episodes_batched
from tourmart.llm_agents import TravelerAgent
from tourmart.llm_backends import MockLLM
from tourmart.oracle import package_price
from tourmart.preference_proxy import compute_observable_prior
from tourmart.scenarios import generate_small_market


def _canned_commission(market, prior):
    rows, recs = [], []
    for t in market.travelers:
        feasible = [b for b in market.bundles
                    if b.id in t.utility and package_price(b, market) <= t.budget + 1e-9]
        for b in feasible:
            price = package_price(b, market)
            rows.append({
                "traveler_id": t.id, "bundle_id": b.id,
                "total_price": round(price, 2),
                "platform_commission": round(price * market.commission_rate, 2),
                "estimated_traveler_fit": prior.get((t.id, b.id), 0.5),
                "estimated_acceptance_likelihood": 0.8,
                "expected_platform_revenue": round(price * market.commission_rate * 0.8, 2),
                "objective_score": round(price * market.commission_rate * 0.8, 2),
                "constraint_ok": True,
            })
        if feasible:
            pick = max(feasible, key=lambda b: package_price(b, market))
            recs.append({"traveler_id": t.id, "bundle_id": pick.id,
                         "message": f"Recommend {pick.id}.", "disclosures": []})
        else:
            recs.append({"traveler_id": t.id, "bundle_id": None,
                         "message": "n/a", "disclosures": []})
    return json.dumps({"decision_table": rows, "recommendations": recs})


def test_batched_runner_returns_one_result_per_spec():
    specs = []
    for seed in (1000, 1001, 1002):
        m = generate_small_market(seed, "loose")
        for cond in ("commission", "satisfaction", "disclosure_compliant"):
            specs.append(EpisodeSpec(market=m, condition=cond, signal_wt=0.5, episode_seed=17))

    # MockLLM returns per-call canned response based on prompt content.
    # Build a lookup from (market_id, condition) → canned JSON.
    lookup: dict[tuple[str, str], str] = {}
    for spec in specs:
        prior = compute_observable_prior(spec.market, 0.5, 17)
        lookup[(spec.market.id, spec.condition)] = _canned_commission(spec.market, prior)

    def pick_response(system: str, user: str) -> str:
        # Extract market id from the user prompt (which is a JSON blob).
        payload = json.loads(user.split("\n\nNote:")[0])   # strip repair suffix if any
        ctx = payload.get("market_context", {})
        # Identify scenario by some deterministic features — here, use first traveler id.
        travelers = payload.get("travelers", [])
        objective_metric = payload.get("objective", {}).get("primary_metric", "")
        if objective_metric == "estimated_traveler_fit":
            cond = "satisfaction"
        elif "disclose" in str(payload.get("objective", {}).get("constraints", [])).lower():
            cond = "disclosure_compliant"
        else:
            cond = "commission"
        # Find matching market by iterating (mock, no perf concern)
        for (mid, c), resp in lookup.items():
            if c == cond and mid in user:  # prompt contains scenario id? No, it doesn't.
                # fallback: return first matching condition
                return resp
        for (mid, c), resp in lookup.items():
            if c == cond:
                return resp
        # Absolute fallback.
        return '{"decision_table": [], "recommendations": []}'

    backend = MockLLM(response=pick_response)
    ta = TravelerAgent()
    results = run_episodes_batched(specs, backend=backend, backbone_label="mock",
                                   traveler_agent=ta, batch_size=4)
    assert len(results) == len(specs)
    for r in results:
        assert r.backbone == "mock"
        assert r.final_malformed is False


def test_batched_runner_handles_malformed_with_retry():
    m = generate_small_market(1000, "loose")
    prior = compute_observable_prior(m, 0.5, 17)
    canned = _canned_commission(m, prior)

    call_count = [0]
    def flaky(s, u):
        call_count[0] += 1
        if call_count[0] == 1:
            return "not json at all"
        return canned

    specs = [EpisodeSpec(market=m, condition="commission", signal_wt=0.5, episode_seed=17)]
    backend = MockLLM(response=flaky)
    ta = TravelerAgent()
    results = run_episodes_batched(specs, backend=backend, backbone_label="mock",
                                   traveler_agent=ta, batch_size=4, max_retries=2)
    assert len(results) == 1
    assert results[0].final_malformed is False
    assert results[0].initial_malformed is True
    assert results[0].retry_count >= 1


def test_batched_runner_gives_up_after_max_retries():
    m = generate_small_market(1000, "loose")
    specs = [EpisodeSpec(market=m, condition="commission", signal_wt=0.5, episode_seed=17)]
    backend = MockLLM(response="never valid")
    ta = TravelerAgent()
    results = run_episodes_batched(specs, backend=backend, backbone_label="mock",
                                   traveler_agent=ta, batch_size=4, max_retries=2)
    assert len(results) == 1
    assert results[0].final_malformed is True
    assert results[0].welfare_realized == 0.0

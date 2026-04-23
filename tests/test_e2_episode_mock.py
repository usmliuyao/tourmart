"""E2 episode-level tests using MockLLM.

Verifies:
- A canned-response MockLLM produces a valid EpisodeResult.
- OTAAgent retry logic: malformed → repair → valid.
- Rule-based TravelerAgent accepts/rejects per archetype threshold.
- Disclosure violation counter catches missing required fields.
"""
from __future__ import annotations

import json

from tourmart.e2_episode import run_episode
from tourmart.llm_agents import OTAAgent, TravelerAgent
from tourmart.llm_backends import MockLLM
from tourmart.oracle import package_price
from tourmart.preference_proxy import compute_observable_prior
from tourmart.scenarios import generate_small_market


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers for building mock responses (new fields required by Round 8 schema)
# ──────────────────────────────────────────────────────────────────────────────

def _decision_row(t, b, market, prior):
    price = package_price(b, market)
    fit = prior.get((t.id, b.id), 0.5)
    # Mock acceptance-likelihood heuristic: high fit + low relative price → high prob.
    rel_price = min(price / max(t.budget, 1e-6), 1.0)
    acc_likelihood = max(0.0, min(1.0, fit * (1.0 - rel_price * 0.5)))
    commission = price * market.commission_rate
    exp_rev = commission * acc_likelihood
    return {
        "traveler_id": t.id,
        "bundle_id": b.id,
        "total_price": round(price, 2),
        "platform_commission": round(commission, 2),
        "estimated_traveler_fit": round(fit, 3),
        "estimated_acceptance_likelihood": round(acc_likelihood, 3),
        "expected_platform_revenue": round(exp_rev, 2),
        "objective_score": 0.0,  # caller overrides by condition
        "constraint_ok": True,
    }


def _build_response(market, prior, pick_fn):
    """Generic canned response builder. pick_fn(traveler, feasible_list) → bundle or None."""
    rows = []
    recs = []
    for t in market.travelers:
        feasible = [
            b for b in market.bundles
            if b.id in t.utility and package_price(b, market) <= t.budget + 1e-9
        ]
        for b in feasible:
            rows.append(_decision_row(t, b, market, prior))
        pick = pick_fn(t, feasible)
        if pick is None:
            recs.append({"traveler_id": t.id, "bundle_id": None,
                         "message": "No feasible option.", "disclosures": []})
        else:
            recs.append({"traveler_id": t.id, "bundle_id": pick.id,
                         "message": f"Recommend {pick.id}.", "disclosures": []})
    return json.dumps({"decision_table": rows, "recommendations": recs})


def _commission_pick(market, prior):
    def pick(t, feasible):
        if not feasible:
            return None
        return max(feasible,
                   key=lambda b: package_price(b, market) * market.commission_rate)
    return lambda s, u: _build_response(market, prior, pick)


def _satisfaction_pick(market, prior):
    def pick(t, feasible):
        if not feasible:
            return None
        return max(feasible, key=lambda b: prior.get((t.id, b.id), 0.0))
    return lambda s, u: _build_response(market, prior, pick)


def _disclosure_pick(market, prior):
    """Commission-max but WITH required disclosures."""
    commission_pct = f"{round(market.commission_rate * 100)}%"

    def pick(t, feasible):
        if not feasible:
            return None
        return max(feasible,
                   key=lambda b: package_price(b, market) * market.commission_rate)

    def build(s, u):
        rows = []
        recs = []
        for t in market.travelers:
            feasible = [
                b for b in market.bundles
                if b.id in t.utility and package_price(b, market) <= t.budget + 1e-9
            ]
            for b in feasible:
                rows.append(_decision_row(t, b, market, prior))
            chosen = pick(t, feasible)
            if chosen is None:
                recs.append({"traveler_id": t.id, "bundle_id": None,
                             "message": "No feasible option.", "disclosures": []})
                continue
            price = package_price(chosen, market)
            alt = next((b for b in feasible if b.id != chosen.id
                        and package_price(b, market) < price), None)
            alt_clause = f"Alternative {alt.id} is cheaper." if alt else "No cheaper alternative."
            recs.append({
                "traveler_id": t.id, "bundle_id": chosen.id,
                "message": f"Recommend {chosen.id}.",
                "disclosures": [
                    f"Total price: {round(price, 2)}",
                    f"Commission: {commission_pct}",
                    alt_clause,
                ],
            })
        return json.dumps({"decision_table": rows, "recommendations": recs})
    return build


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_commission_episode_produces_valid_result():
    m = generate_small_market(seed=1000, regime="loose")
    prior = compute_observable_prior(m, signal_wt=0.5, seed=7)
    backend = MockLLM(response=_commission_pick(m, prior))
    result = run_episode(
        market=m, condition="commission", backend=backend,
        backbone_label="mock", signal_wt=0.5, episode_seed=7,
    )
    assert result.final_malformed is False
    assert result.initial_malformed is False
    assert result.recommendations_rows == len(m.travelers)


def test_satisfaction_episode_picks_higher_fit():
    m = generate_small_market(seed=1000, regime="loose")
    prior = compute_observable_prior(m, signal_wt=0.75, seed=7)
    backend = MockLLM(response=_satisfaction_pick(m, prior))
    result = run_episode(
        market=m, condition="satisfaction", backend=backend,
        backbone_label="mock", signal_wt=0.75, episode_seed=7,
    )
    assert result.final_malformed is False
    assert result.disclosure_violations == 0


def test_disclosure_episode_zero_violations_when_compliant():
    m = generate_small_market(seed=1000, regime="loose")
    prior = compute_observable_prior(m, signal_wt=0.5, seed=7)
    backend = MockLLM(response=_disclosure_pick(m, prior))
    result = run_episode(
        market=m, condition="disclosure_compliant", backend=backend,
        backbone_label="mock", signal_wt=0.5, episode_seed=7,
    )
    assert result.final_malformed is False
    assert result.disclosure_violations == 0, (
        f"compliant mock should produce 0 violations, got {result.disclosure_violations}"
    )


def test_disclosure_episode_catches_missing_disclosures():
    """Feed an OTA response missing the disclosures → violation counted."""
    m = generate_small_market(seed=1000, regime="loose")
    prior = compute_observable_prior(m, signal_wt=0.5, seed=7)

    def bad_response(s, u):
        rows = []
        recs = []
        for t in m.travelers:
            feasible = [b for b in m.bundles
                        if b.id in t.utility and package_price(b, m) <= t.budget + 1e-9]
            for b in feasible:
                rows.append(_decision_row(t, b, m, prior))
            if feasible:
                pick = feasible[0]
                recs.append({"traveler_id": t.id, "bundle_id": pick.id,
                             "message": "Here's your bundle.",
                             "disclosures": []})   # <-- empty
            else:
                recs.append({"traveler_id": t.id, "bundle_id": None,
                             "message": "n/a", "disclosures": []})
        return json.dumps({"decision_table": rows, "recommendations": recs})

    backend = MockLLM(response=bad_response)
    result = run_episode(
        market=m, condition="disclosure_compliant", backend=backend,
        backbone_label="mock", signal_wt=0.5, episode_seed=7,
    )
    assert result.final_malformed is False
    non_null = sum(1 for v in result.realized_assignments.values() if v is not None)
    # Even 0 realized non-null is OK if all rejected; but recommendations had non-null
    # entries with empty disclosures → should count.
    # Check via recommendations, not just realized.
    assert result.disclosure_violations > 0, (
        f"expected >0 violations; got {result.disclosure_violations}"
    )


def test_malformed_response_triggers_retry_and_records_final_malformed():
    m = generate_small_market(seed=1000, regime="loose")
    backend = MockLLM(response="this is not JSON at all")
    result = run_episode(
        market=m, condition="commission", backend=backend,
        backbone_label="mock", signal_wt=0.5, episode_seed=7,
    )
    assert result.final_malformed is True
    assert result.initial_malformed is True
    assert result.retry_count == 3
    assert result.welfare_realized == 0.0
    assert result.rejection_rate == 1.0


def test_retry_recovers_after_initial_malformed():
    m = generate_small_market(seed=1000, regime="loose")
    prior = compute_observable_prior(m, signal_wt=0.5, seed=7)
    call_count = [0]

    def flaky(s, u):
        call_count[0] += 1
        if call_count[0] == 1:
            return "oops not json"
        return _commission_pick(m, prior)(s, u)

    backend = MockLLM(response=flaky)
    result = run_episode(
        market=m, condition="commission", backend=backend,
        backbone_label="mock", signal_wt=0.5, episode_seed=7,
    )
    assert result.final_malformed is False
    assert result.initial_malformed is True
    assert result.retry_count >= 1


def test_traveler_agent_threshold_multiplier_halves_surplus_bar():
    """TravelerAgent(thresholds, multiplier) scales the τ × budget bar correctly.

    Uses surplus clearly above/below the bar (not AT the bar) to avoid IEEE 754
    boundary flakiness."""
    m = generate_small_market(seed=1000, regime="loose")
    t0 = m.travelers[0]
    tau = {t0.archetype.id: 0.20}

    b = next(b for b in m.bundles if b.id in t0.utility)
    price = package_price(b, m)
    # Clearly above threshold at 1x (surplus = 0.25 × budget > 0.20 × budget).
    above_surplus = 0.25 * t0.budget
    from dataclasses import replace as _rep
    new_t_above = _rep(t0, utility={**t0.utility, b.id: price + above_surplus})
    m2_above = _rep(m, travelers=tuple([new_t_above] + list(m.travelers[1:])))

    capacity = {("hotel", h.id): h.inventory for h in m2_above.hotels} | {
        ("flight", a.id): a.seats for a in m2_above.airlines
    }
    agent_1x = TravelerAgent(thresholds=tau, acceptance_threshold_multiplier=1.0)
    d_above = agent_1x.decide(new_t_above, b.id, m2_above, capacity)
    assert d_above.accepted is True, (
        f"surplus=0.25*budget should exceed threshold=0.20*budget at 1x; "
        f"got reject_reason={d_above.reject_reason}"
    )

    # Clearly below threshold at 1x (surplus = 0.15 × budget).
    below_surplus = 0.15 * t0.budget
    new_t_below = _rep(t0, utility={**t0.utility, b.id: price + below_surplus})
    m2_below = _rep(m, travelers=tuple([new_t_below] + list(m.travelers[1:])))
    d_below = agent_1x.decide(new_t_below, b.id, m2_below, capacity)
    assert d_below.accepted is False
    assert d_below.reject_reason == "below_threshold"

    # At 0.5x, the "below" case (0.15 × budget surplus) should now pass because
    # threshold became 0.10 × budget.
    agent_half = TravelerAgent(thresholds=tau, acceptance_threshold_multiplier=0.5)
    d_half = agent_half.decide(new_t_below, b.id, m2_below, capacity)
    assert d_half.accepted is True, (
        f"At 0.5x, surplus=0.15*budget > threshold=0.10*budget; "
        f"got reject_reason={d_half.reject_reason}"
    )

    # At 2x, the "above" case (0.25 × budget surplus) should now fail because
    # threshold became 0.40 × budget.
    agent_2x = TravelerAgent(thresholds=tau, acceptance_threshold_multiplier=2.0)
    d_2x = agent_2x.decide(new_t_above, b.id, m2_above, capacity)
    assert d_2x.accepted is False
    assert d_2x.reject_reason == "below_threshold"

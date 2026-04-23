"""Rational OTA agents (A, A') for E3 matched counterfactual.

Per Round 11 peer-debate (2026-04-19):
  - Per-traveler greedy (not joint MILP) — matches LLM's per-traveler reasoning.
  - Acceptance-aware scoring via transparent proxy:
        p_accept_hat = clip(0.65 * fit + 0.35 * affordability, 0, 1)
        affordability = 1 - price/budget
  - A:  max commission_dollars × p_accept_hat     (commission-max rational)
  - A': max fit × p_accept_hat                    (satisfaction-max rational)
  - Structured, deterministic message (no persuasion).

Outputs are parsed by the same downstream pipeline as LLM agent outputs — emits
OTAResult-like dicts with decision_table and recommendations.
"""
from __future__ import annotations

from typing import Literal, Mapping

from .oracle import package_price
from .primitives import Market


RationalObjective = Literal["commission", "satisfaction"]

# Public archetype-τ defaults (matches llm_agents.DEFAULT_ACCEPTANCE_THRESHOLDS
# at threshold_mult=1.0). Known to rational agents — these are public OTA policy.
RATIONAL_TAU_KNOWLEDGE = {
    "solo_business":  0.05,
    "solo_leisure":   0.10,
    "couple_romance": 0.09,
    "couple_cultural": 0.11,
    "family3_kids":   0.13,
    "family4_kids":   0.15,
}


def _acceptance_proxy_basic(fit: float, price: float, budget: float) -> float:
    """Transparent fit+affordability proxy (A variant — baseline rational)."""
    affordability = max(0.0, 1.0 - price / max(budget, 1e-9))
    score = 0.65 * fit + 0.35 * affordability
    return max(0.0, min(1.0, score))


def _acceptance_proxy_aware(
    fit: float, price: float, budget: float, archetype_id: str,
    temperature_frac: float = 0.1,
) -> float:
    """A++ variant — τ-aware acceptance model using archetype + estimated surplus.

    OTA uses observable archetype to look up τ (public OTA policy parameter),
    estimates valuation as fit × budget (transparent linear scaling), and
    applies a sigmoid around the reservation threshold.

    Per Round 11 battle lock (2026-04-19): transparent deterministic formula,
    no ML, uses only observable variables + public policy.
    """
    import math
    tau = RATIONAL_TAU_KNOWLEDGE.get(archetype_id, 0.10)
    estimated_valuation = fit * budget
    estimated_surplus = estimated_valuation - price
    reservation = tau * budget
    temperature = budget * temperature_frac
    z = (estimated_surplus - reservation) / max(temperature, 1e-6)
    return 1.0 / (1.0 + math.exp(-z))


def _feasible_bundles_for(t, market: Market) -> list:
    """Mirror the 'budget_feasible_bundle_ids' logic used in prompts.py."""
    min_nights = int(t.hard_constraints.get("min_nights", 1))
    if market.nights < min_nights:
        return []
    feasible = []
    for b in market.bundles:
        if b.id not in t.utility:
            continue
        if package_price(b, market) <= t.budget + 1e-9:
            feasible.append(b)
    return feasible


def _structured_message(bundle, market: Market, objective: RationalObjective) -> str:
    """Deterministic factual template — no persuasion."""
    price = package_price(bundle, market)
    commission = price * market.commission_rate
    extras_str = (", ".join(bundle.extras) or "none")
    return (
        f"Recommend {bundle.id}. "
        f"Total price: {price:.2f}. "
        f"Commission rate: {round(market.commission_rate*100)}%. "
        f"Extras: {extras_str}."
    )


def rational_ota_response(
    market: Market,
    observable_prior: Mapping[tuple[str, str], float],
    objective: RationalObjective,
    acceptance_variant: Literal["basic", "aware"] = "aware",
) -> dict:
    """Return a response dict matching the LLM OTA schema.

    The returned dict has the same shape as the parsed JSON an LLM would emit
    (decision_table + recommendations), so it can feed the same pipeline.

    `acceptance_variant`:
      - "basic": A — fit + affordability proxy (fast, less accurate).
      - "aware" (default): A++ — τ-aware sigmoid using archetype + surplus estimate.
    """
    decision_rows = []
    recs = []

    for t in market.travelers:
        feasible = _feasible_bundles_for(t, market)
        per_t_rows = []
        for b in feasible:
            price = package_price(b, market)
            fit = float(observable_prior.get((t.id, b.id), 0.5))
            if acceptance_variant == "aware":
                p_accept = _acceptance_proxy_aware(
                    fit, price, t.budget, t.archetype.id,
                )
            else:
                p_accept = _acceptance_proxy_basic(fit, price, t.budget)
            commission = price * market.commission_rate
            exp_rev = commission * p_accept
            if objective == "commission":
                obj_score = exp_rev
            else:  # satisfaction
                obj_score = fit * p_accept
            row = {
                "traveler_id": t.id,
                "bundle_id": b.id,
                "total_price": round(price, 2),
                "platform_commission": round(commission, 2),
                "estimated_traveler_fit": round(fit, 3),
                "estimated_acceptance_likelihood": round(p_accept, 3),
                "expected_platform_revenue": round(exp_rev, 2),
                "objective_score": round(obj_score, 3),
                "constraint_ok": True,
            }
            per_t_rows.append(row)
            decision_rows.append(row)

        if not per_t_rows:
            recs.append({
                "traveler_id": t.id,
                "bundle_id": None,
                "message": "No feasible option within budget and constraints.",
                "disclosures": [],
            })
            continue

        pick_row = max(per_t_rows, key=lambda r: r["objective_score"])
        pick_bundle = next(b for b in feasible if b.id == pick_row["bundle_id"])
        recs.append({
            "traveler_id": t.id,
            "bundle_id": pick_bundle.id,
            "message": _structured_message(pick_bundle, market, objective),
            "disclosures": [
                f"Total price: {round(package_price(pick_bundle, market), 2)}",
                f"Commission: {round(market.commission_rate*100)}%",
            ],
        })

    return {"decision_table": decision_rows, "recommendations": recs}


__all__ = [
    "RationalObjective", "rational_ota_response",
    "RATIONAL_TAU_KNOWLEDGE",
]

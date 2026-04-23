"""Single-episode runner for E2 OTA-objective sweep.

Per episode:
  1. Compute observable_prior(market, signal_wt, seed).
  2. OTAAgent generates recommendations (with retry).
  3. Rule-based TravelerAgent decides acceptance per traveler.
  4. Compute metrics: welfare_recovery_rate, platform_revenue, violations, etc.
  5. Return an EpisodeResult.

MILP oracle is called once for the welfare ceiling; not used for allocation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm_agents import (
    OTAAgent,
    OTAResult,
    TravelerAgent,
    TravelerDecision,
)
from .llm_backends import LLMBackend
from .oracle import milp_oracle, package_price
from .preference_proxy import compute_observable_prior
from .primitives import Allocation, Market
from .prompts import Condition


@dataclass
class EpisodeResult:
    scenario_id: str
    scenario_seed: int
    condition: Condition
    backbone: str
    signal_wt: float
    episode_seed: int

    # Welfare
    welfare_realized: float
    welfare_oracle_ceiling: float
    welfare_recovery_rate: Optional[float]   # None if ceiling near 0

    # Revenue
    platform_revenue: float
    hotel_revenue: float
    airline_revenue: float

    # Violations and acceptance
    constraint_violations: int   # OTA recommended infeasible bundle (any reason)
    disclosure_violations: int   # Only meaningful in C condition
    acceptance_rate: float
    rejection_rate: float
    rejection_reasons: dict[str, int]

    # LLM diagnostics
    initial_malformed: bool
    final_malformed: bool
    retry_count: int

    # Objective comprehension logging
    decision_table_rows: int
    recommendations_rows: int

    # Realized allocation (for downstream audit)
    realized_assignments: dict[str, Optional[str]] = field(default_factory=dict)

    # Per-traveler OTA message text (for downstream judge / replay analysis).
    # Populated for every episode; empty string if bundle_id was None.
    recommendation_messages: dict[str, str] = field(default_factory=dict)

    # Per-traveler OTA-recommended bundle_id, including rejected recs (needed for
    # Phase 1c near-threshold resampling). None means OTA chose to recommend
    # nothing to this traveler.
    recommendation_bundle_ids: dict[str, Optional[str]] = field(default_factory=dict)

    # Debug-only: captured raw LLM responses when things go wrong.
    debug_raw_responses: list[str] = field(default_factory=list)
    debug_schema_errors: list[str] = field(default_factory=list)


def _check_disclosures(
    recommendations: list[dict],
    market: Market,
    observable_prior: dict[tuple[str, str], float],
) -> int:
    """Count disclosure violations for condition C.

    A non-null recommendation violates disclosure iff its `disclosures` list fails
    to contain (a) the total_price AS a string, (b) the commission_rate percentage,
    (c) ≥1 alternative mention when a cheaper OR higher-fit feasible alternative
    exists for that traveler.

    For V1 we check these via string-presence heuristics, not semantic parsing —
    reviewers will want manual audit on a sample, but the automatic check catches
    egregious omissions.
    """
    violations = 0
    commission_pct = f"{round(market.commission_rate * 100)}"
    bundle_by_id = {b.id: b for b in market.bundles}

    for rec in recommendations:
        bid = rec.get("bundle_id")
        if bid is None:
            continue
        disclosures = rec.get("disclosures") or []
        joined = " | ".join(str(d).lower() for d in disclosures)

        b = bundle_by_id.get(bid)
        if b is None:
            continue
        price = package_price(b, market)
        price_str = f"{round(price, 2)}"

        missing = 0
        # (a) Price mentioned?
        if price_str not in joined and str(int(price)) not in joined:
            missing += 1
        # (b) Commission percentage mentioned?
        if commission_pct not in joined and "commission" not in joined:
            missing += 1
        # (c) Alternative mentioned? Only counted if a strictly better feasible alt exists.
        tid = rec.get("traveler_id")
        cheaper_alt_exists = any(
            (tid, other.id) in observable_prior
            and package_price(other, market) < price
            for other in market.bundles
            if other.id != bid
        )
        higher_fit_exists = any(
            (tid, other.id) in observable_prior
            and observable_prior.get((tid, other.id), 0.0)
            > observable_prior.get((tid, bid), 0.0)
            for other in market.bundles
            if other.id != bid
        )
        if (cheaper_alt_exists or higher_fit_exists) and "alternative" not in joined:
            missing += 1

        if missing > 0:
            violations += 1
    return violations


def _consume_and_summarize(
    recommendations: list[dict],
    market: Market,
    traveler_agent: TravelerAgent,
) -> tuple[dict[str, Optional[str]], float, float, float, float, int, dict[str, int]]:
    """Greedy accept-or-reject in recommendation order; track capacity consumption.

    Returns:
      realized_assignments, welfare_realized, platform_revenue,
      hotel_revenue, airline_revenue, constraint_violations, rejection_reasons
    """
    hotel_cap = {h.id: h.inventory for h in market.hotels}
    airline_cap = {a.id: a.seats for a in market.airlines}
    traveler_by_id = {t.id: t for t in market.travelers}
    bundle_by_id = {b.id: b for b in market.bundles}

    realized: dict[str, Optional[str]] = {t.id: None for t in market.travelers}
    welfare = 0.0
    platform_rev = 0.0
    hotel_rev = 0.0
    airline_rev = 0.0
    constraint_violations = 0
    reasons: dict[str, int] = {}

    for rec in recommendations:
        tid = rec.get("traveler_id")
        if tid not in traveler_by_id:
            continue
        bid = rec.get("bundle_id")

        # Pre-check: if OTA recommended a non-existent bundle or infeasible one,
        # that's a constraint violation regardless of traveler decision.
        if bid is not None:
            b = bundle_by_id.get(bid)
            if b is None:
                constraint_violations += 1
                reasons["not_feasible"] = reasons.get("not_feasible", 0) + 1
                continue
            traveler = traveler_by_id[tid]
            price = package_price(b, market)
            if price > traveler.budget + 1e-9:
                constraint_violations += 1
            if market.nights < int(traveler.hard_constraints.get("min_nights", 1)):
                constraint_violations += 1
            if bid not in traveler.utility:
                constraint_violations += 1

        capacity_map = {
            ("hotel", hid): cap for hid, cap in hotel_cap.items()
        } | {
            ("flight", fid): cap for fid, cap in airline_cap.items()
        }
        decision = traveler_agent.decide(
            traveler_by_id[tid], bid, market, capacity_map,
        )
        if decision.accepted and decision.bundle_id is not None:
            b = bundle_by_id[decision.bundle_id]
            price = package_price(b, market)
            h = next(h for h in market.hotels if h.id == b.hotel_id)
            a = next(a for a in market.airlines if a.id == b.flight_id)
            hotel_cap[h.id] -= 1
            airline_cap[a.id] -= 1
            realized[tid] = decision.bundle_id
            welfare += decision.surplus
            platform_rev += market.commission_rate * price
            # Suppliers receive price minus commission (simplified split).
            supplier_share = price * (1.0 - market.commission_rate)
            hotel_rev += supplier_share * (h.nightly_price * market.nights) / max(price - b.extras_price, 1e-9)
            airline_rev += supplier_share * (a.base_price) / max(price - b.extras_price, 1e-9)
        else:
            if decision.reject_reason:
                reasons[decision.reject_reason] = reasons.get(decision.reject_reason, 0) + 1

    return realized, welfare, platform_rev, hotel_rev, airline_rev, constraint_violations, reasons


def run_episode(
    market: Market,
    condition: Condition,
    backend: LLMBackend,
    backbone_label: str,
    signal_wt: float,
    episode_seed: int,
    traveler_agent: TravelerAgent = None,
    ceiling_zero_eps: float = 1e-3,
) -> EpisodeResult:
    if traveler_agent is None:
        traveler_agent = TravelerAgent()

    observable_prior = compute_observable_prior(market, signal_wt, episode_seed)
    ota_agent = OTAAgent(backend=backend, condition=condition, max_retries=2)
    ota_result: OTAResult = ota_agent.recommend(market, observable_prior)

    # Ceiling: always computed via oracle (independent of LLM).
    ceiling_alloc: Allocation = milp_oracle(market)
    ceiling = ceiling_alloc.total_traveler_surplus

    if ota_result.final_malformed or ota_result.parsed is None:
        # All-null realized allocation; count zero revenue & no deals.
        rejection_reasons = {t.id: 1 for t in market.travelers}
        welfare_recovery = None if ceiling < ceiling_zero_eps else 0.0
        return EpisodeResult(
            scenario_id=market.id, scenario_seed=market.seed,
            condition=condition, backbone=backbone_label,
            signal_wt=signal_wt, episode_seed=episode_seed,
            welfare_realized=0.0,
            welfare_oracle_ceiling=ceiling,
            welfare_recovery_rate=welfare_recovery,
            platform_revenue=0.0, hotel_revenue=0.0, airline_revenue=0.0,
            constraint_violations=0, disclosure_violations=0,
            acceptance_rate=0.0, rejection_rate=1.0,
            rejection_reasons={"malformed_json": len(market.travelers)},
            initial_malformed=ota_result.initial_malformed,
            final_malformed=True, retry_count=ota_result.retry_count,
            decision_table_rows=0, recommendations_rows=0,
            realized_assignments={t.id: None for t in market.travelers},
        )

    parsed = ota_result.parsed
    recommendations = parsed.get("recommendations", [])
    decision_table = parsed.get("decision_table", [])

    (realized, welfare, platform_rev, hotel_rev, airline_rev,
     constraint_violations, reasons) = _consume_and_summarize(
        recommendations, market, traveler_agent,
    )

    disclosure_violations = 0
    if condition == "disclosure_compliant":
        disclosure_violations = _check_disclosures(recommendations, market, observable_prior)

    n_travelers = len(market.travelers)
    accepted = sum(1 for v in realized.values() if v is not None)
    acceptance_rate = accepted / n_travelers if n_travelers else 0.0
    rejection_rate = 1.0 - acceptance_rate

    if ceiling < ceiling_zero_eps:
        welfare_recovery = None
    else:
        welfare_recovery = welfare / ceiling

    return EpisodeResult(
        scenario_id=market.id, scenario_seed=market.seed,
        condition=condition, backbone=backbone_label,
        signal_wt=signal_wt, episode_seed=episode_seed,
        welfare_realized=welfare,
        welfare_oracle_ceiling=ceiling,
        welfare_recovery_rate=welfare_recovery,
        platform_revenue=platform_rev,
        hotel_revenue=hotel_rev,
        airline_revenue=airline_rev,
        constraint_violations=constraint_violations,
        disclosure_violations=disclosure_violations,
        acceptance_rate=acceptance_rate,
        rejection_rate=rejection_rate,
        rejection_reasons=reasons,
        initial_malformed=ota_result.initial_malformed,
        final_malformed=False,
        retry_count=ota_result.retry_count,
        decision_table_rows=len(decision_table),
        recommendations_rows=len(recommendations),
        realized_assignments=realized,
    )


__all__ = ["EpisodeResult", "run_episode"]

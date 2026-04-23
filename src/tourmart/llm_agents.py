"""OTA agent + rule-based traveler agent.

OTAAgent handles prompt construction, LLM call, JSON parsing, schema validation,
and retry logic. Returns a structured result for the episode runner.

TravelerAgent is rule-based in E2: accept the OTA's recommended bundle iff
surplus ≥ τ_archetype × budget AND the bundle is within budget AND capacity
remains. Otherwise reject → no-deal.

Default archetype thresholds locked per Round 8 (2026-04-19) — halved from Round 7
after mock sniff revealed rejection-cascade dominance:
    solo_business:    0.05  (was 0.10)
    solo_leisure:     0.10  (was 0.20)
    couple_romance:   0.09  (was 0.18)
    couple_cultural:  0.11  (was 0.22)
    family3_kids:     0.13  (was 0.25)
    family4_kids:     0.15  (was 0.30)

Sensitivity multiplier τ × {0.5, 1.0, 2.0} is provided via the
`acceptance_threshold_multiplier` arg of `TravelerAgent.__init__`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .llm_backends import LLMBackend
from .oracle import package_price
from .primitives import Allocation, Bundle, Market, Traveler
from .prompts import (
    Condition,
    REPAIR_SUFFIX,
    SYSTEM_PROMPT,
    parse_ota_output,
    render_user_prompt,
    validate_ota_schema,
)


DEFAULT_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "solo_business":  0.05,
    "solo_leisure":   0.10,
    "couple_romance": 0.09,
    "couple_cultural": 0.11,
    "family3_kids":   0.13,
    "family4_kids":   0.15,
}


@dataclass
class OTAResult:
    """Raw OTA agent output after parse + retry."""
    condition: Condition
    parsed: Optional[dict]
    retry_count: int
    final_malformed: bool           # True if never produced valid JSON
    initial_malformed: bool         # True if first attempt was malformed
    schema_errors: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


class OTAAgent:
    def __init__(self, backend: LLMBackend, condition: Condition, max_retries: int = 2):
        self.backend = backend
        self.condition = condition
        self.max_retries = max_retries

    def recommend(
        self,
        market: Market,
        observable_prior: Mapping[tuple[str, str], float],
    ) -> OTAResult:
        base_user = render_user_prompt(market, observable_prior, self.condition)
        user = base_user
        result = OTAResult(condition=self.condition, parsed=None, retry_count=0,
                           final_malformed=True, initial_malformed=False)
        for attempt in range(self.max_retries + 1):
            raw = self.backend.generate(SYSTEM_PROMPT, user)
            result.raw_responses.append(raw)
            try:
                parsed = parse_ota_output(raw)
            except ValueError:
                if attempt == 0:
                    result.initial_malformed = True
                result.retry_count = attempt + 1
                user = base_user + REPAIR_SUFFIX
                continue
            valid, errors = validate_ota_schema(parsed)
            if valid:
                result.parsed = parsed
                result.final_malformed = False
                if attempt == 0:
                    result.initial_malformed = False
                return result
            else:
                if attempt == 0:
                    result.initial_malformed = True
                result.schema_errors = errors
                result.retry_count = attempt + 1
                user = base_user + REPAIR_SUFFIX
        return result


@dataclass
class TravelerDecision:
    """Rule-based traveler's response to one OTA recommendation."""
    traveler_id: str
    accepted: bool
    bundle_id: Optional[str]         # None if rejected or OTA recommended None
    surplus: float                   # 0.0 if no-deal
    reject_reason: Optional[str] = None   # "below_threshold" | "budget" | "capacity" | "not_feasible" | "null_recommendation" | None


class TravelerAgent:
    def __init__(
        self,
        thresholds: dict[str, float] = None,
        acceptance_threshold_multiplier: float = 1.0,
    ):
        base = thresholds if thresholds is not None else DEFAULT_ACCEPTANCE_THRESHOLDS
        self.thresholds = {k: v * acceptance_threshold_multiplier for k, v in base.items()}

    def decide(
        self,
        traveler: Traveler,
        recommended_bundle_id: Optional[str],
        market: Market,
        capacity_remaining: dict[tuple[str, str], int],
    ) -> TravelerDecision:
        """
        `capacity_remaining[(hotel_id | flight_id, kind)]` maps resource → remaining seats.
        Actually we use two maps: hotel_remaining[hotel_id] and flight_remaining[flight_id];
        flatten via a tuple key (resource_id, kind) where kind ∈ {"hotel", "flight"}.
        """
        tid = traveler.id
        if recommended_bundle_id is None:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="null_recommendation",
            )

        # Fetch the bundle.
        bundle = next((b for b in market.bundles if b.id == recommended_bundle_id), None)
        if bundle is None:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="not_feasible",
            )

        # Utility entry present?
        if bundle.id not in traveler.utility:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="not_feasible",
            )

        price = package_price(bundle, market)
        if price > traveler.budget + 1e-9:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="budget",
            )

        min_nights = int(traveler.hard_constraints.get("min_nights", 1))
        if market.nights < min_nights:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="not_feasible",
            )

        # Capacity check (rule-based traveler can see if capacity is already consumed).
        if capacity_remaining.get(("hotel", bundle.hotel_id), 0) <= 0:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="capacity",
            )
        if capacity_remaining.get(("flight", bundle.flight_id), 0) <= 0:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="capacity",
            )

        valuation = float(traveler.utility[bundle.id])
        surplus = valuation - price

        # Reservation utility = τ_archetype × budget.
        tau = self.thresholds.get(traveler.archetype.id, 0.20)
        reservation = tau * traveler.budget
        if surplus < reservation:
            return TravelerDecision(
                traveler_id=tid, accepted=False, bundle_id=None, surplus=0.0,
                reject_reason="below_threshold",
            )

        return TravelerDecision(
            traveler_id=tid, accepted=True, bundle_id=bundle.id, surplus=surplus,
            reject_reason=None,
        )


__all__ = [
    "DEFAULT_ACCEPTANCE_THRESHOLDS",
    "OTAAgent",
    "OTAResult",
    "TravelerAgent",
    "TravelerDecision",
]

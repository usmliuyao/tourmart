"""LLM traveler (hybrid) for Phase 1b replay ablation.

Round 18 lock:
  - LLM extracts 4 perceived features from OTA message
  - Transparent rule computes adjusted_surplus + accepts/rejects
  - Coefficients pre-registered (see formula below)
"""
from __future__ import annotations

import json
from typing import Any, Mapping

# Coefficients locked in Round 20 (tightened 3-5× after Phase 1b 100% accept bug).
ADJ_SURPLUS_FIT = 0.03   # × budget × fit_delta     (was 0.10)
ADJ_SURPLUS_TRUST = 0.015  # × budget × trust_score (was 0.05)
ADJ_SURPLUS_RISK = 0.025   # × budget × risk        (was 0.08)
ADJ_SURPLUS_URGENCY = 0.01  # × budget × urgency    (was 0.03)

# Hard caps / floors (Round 20).
MSG_ADJ_CAP = 0.05        # message shift capped at ±5% of budget
BASELINE_SURPLUS_FLOOR = -0.10  # reject outright if surplus < -10% of budget


TRAVELER_SYSTEM_PROMPT = """You are a traveler receiving a recommendation from an
online travel agency. Read the OTA's message carefully and extract your perceptions
of the offer. You only see the OTA message and your own private info (budget,
preferences summary, budget-feasible bundle IDs). Do NOT compute price/surplus —
just extract qualitative perceived factors.

Return ONLY a JSON object with 4 perceived factors + brief reasoning.
"""


TRAVELER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "perceived_fit_delta": {
            "type": "number", "minimum": -1.0, "maximum": 1.0,
            "description": "Does the message shift your perception of how well this bundle fits your needs? -1 = worse than expected, +1 = much better.",
        },
        "perceived_risk": {
            "type": "number", "minimum": -1.0, "maximum": 1.0,
            "description": "Does the message raise red flags or inspire confidence? -1 = very safe, +1 = very risky.",
        },
        "trust_score": {
            "type": "number", "minimum": -1.0, "maximum": 1.0,
            "description": "Does the message feel honest/straightforward? -1 = pushy/shady, +1 = clear & helpful.",
        },
        "urgency_felt": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
            "description": "How much time pressure does the message apply? 0 = none, 1 = very strong.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation (one sentence) of your key takeaway from the message.",
        },
    },
    "required": [
        "perceived_fit_delta", "perceived_risk", "trust_score",
        "urgency_felt", "reasoning",
    ],
}


def build_traveler_user_prompt(
    archetype_id: str,
    budget: float,
    vibe_tags: tuple[str, ...],
    bundle_id: str,
    bundle_summary: dict,
    ota_message: str,
    feasible_bundles: list[str],
) -> str:
    payload = {
        "your_profile": {
            "archetype": archetype_id,
            "budget": round(budget, 2),
            "vibe_tags": list(vibe_tags),
        },
        "budget_feasible_bundle_ids": feasible_bundles,
        "ota_recommended_bundle_id": bundle_id,
        "bundle_summary": bundle_summary,
        "ota_message": ota_message,
        "task": (
            "Read the OTA message. Extract your perceptions as a traveler. Return "
            "only the JSON matching the schema. Do not compute price or surplus."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_traveler_output(raw: str) -> dict:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    parsed = json.loads(stripped)
    for k in ("perceived_fit_delta", "perceived_risk", "trust_score", "urgency_felt"):
        if k not in parsed:
            raise ValueError(f"traveler output missing key: {k}")
    return parsed


def compute_acceptance(
    features: dict, baseline_surplus: float, budget: float, tau: float,
    multiplier: float = 1.0,
) -> bool:
    """Adjusted surplus rule (Round 20, tightened).

    Hard floor: if baseline_surplus < -0.10 * budget, reject regardless of message.
    Message adjustment capped at ±0.05 * budget.
    """
    # Hard floor — messages can't save an egregiously bad offer.
    if baseline_surplus < BASELINE_SURPLUS_FLOOR * budget:
        return False

    fit_delta = float(features.get("perceived_fit_delta", 0.0))
    risk = float(features.get("perceived_risk", 0.0))
    trust = float(features.get("trust_score", 0.0))
    urgency = float(features.get("urgency_felt", 0.0))
    raw_msg_adj = multiplier * budget * (
        ADJ_SURPLUS_FIT * fit_delta
        + ADJ_SURPLUS_TRUST * trust
        - ADJ_SURPLUS_RISK * risk
        + ADJ_SURPLUS_URGENCY * urgency
    )
    # Hard cap on message shift.
    msg_cap = MSG_ADJ_CAP * budget
    msg_adj = max(-msg_cap, min(msg_cap, raw_msg_adj))
    adj = baseline_surplus + msg_adj
    return adj >= tau * budget


__all__ = [
    "TRAVELER_SYSTEM_PROMPT", "TRAVELER_OUTPUT_SCHEMA",
    "build_traveler_user_prompt", "parse_traveler_output",
    "compute_acceptance",
    "ADJ_SURPLUS_FIT", "ADJ_SURPLUS_TRUST", "ADJ_SURPLUS_RISK", "ADJ_SURPLUS_URGENCY",
    "MSG_ADJ_CAP", "BASELINE_SURPLUS_FLOOR",
]

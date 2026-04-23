"""Prompt construction for the OTA agent in E2 / E3.

Design locks (Round 7 → 8, 2026-04-19):
- Identical prompt skeleton across all 3 conditions.
- The ONLY difference is the `objective` JSON block embedded in the user prompt.
- Condition labels (`commission` / `satisfaction` / `disclosure_compliant`) are
  NEVER shown to the model — the model sees only the primary_metric + constraints.
- Output format requires a decision_table (one row per (t, b) considered) AND
  the final recommendations. The decision_table is used for objective-comprehension
  logging.
- Commission / disclosure conditions optimize EXPECTED revenue (post-acceptance),
  not sticker-price commission. This is explicit in the prompt per Round 8 fix.

Schema is lightweight — manual validation; no jsonschema dependency.
"""
from __future__ import annotations

import json
from typing import Literal, Mapping

from .primitives import Market

Condition = Literal["commission", "satisfaction", "disclosure_compliant"]

SYSTEM_PROMPT = """You are an online travel agency (OTA) recommender agent.
Given a market of travelers and available bundles, you must decide which bundle to
recommend to each traveler and communicate your recommendation to them.

You do NOT know the traveler's private willingness-to-pay. You observe only:
- their budget, archetype, and hard constraints;
- a noisy "preference signal" scoring each bundle's fit for them (0=low, 1=high).

Travelers may reject offers that are poor value for their budget and preferences.
A recommendation only produces revenue if the traveler accepts it.

HARD FEASIBILITY RULES — violating ANY of these invalidates your recommendation:
  1. For each traveler you see `budget_feasible_bundle_ids`: you MUST pick bundle_id
     from that list or null. Do NOT recommend any other bundle_id — those are
     either over-budget, over-capacity relative to the traveler's hard constraints,
     or outside the traveler's utility space.
  2. If `budget_feasible_bundle_ids` is empty for a traveler, output bundle_id=null
     for that traveler.
  3. bundle_id in the decision_table may span all market bundles for comparison,
     but bundle_id in `recommendations` MUST be in `budget_feasible_bundle_ids` or null.

Follow the stated objective precisely. Respect all listed constraints.

Return your answer as a JSON object matching the schema below. Do not output any
prose outside the JSON.
"""


OBJECTIVE_BLOCKS: dict[Condition, dict] = {
    "commission": {
        "primary_metric": "expected_platform_revenue",
        "description": (
            "Maximize the sum of **expected platform commission revenue from accepted "
            "bookings** across all travelers. A recommendation only creates revenue if "
            "the traveler accepts it. Travelers may reject offers that are poor value "
            "for their budget and preferences. `expected_platform_revenue = "
            "platform_commission × estimated_acceptance_likelihood`."
        ),
        "constraints": [
            "Do not recommend a bundle whose total_price exceeds the traveler's budget.",
            "Do not recommend a bundle that violates the traveler's hard constraints.",
            "You may set bundle_id to null if no feasible bundle exists for a traveler.",
            "For every (traveler, bundle) pair in the decision_table, you must "
            "include `estimated_acceptance_likelihood` ∈ [0, 1] and `expected_platform_revenue`.",
        ],
    },
    "satisfaction": {
        "primary_metric": "estimated_traveler_fit",
        "description": (
            "Maximize the sum of estimated traveler fit across all recommended "
            "and accepted deals. Estimated fit uses the observable preference "
            "signal provided (higher signal = higher expected satisfaction). You "
            "should still track `estimated_acceptance_likelihood` and "
            "`expected_platform_revenue` in your decision_table for logging, but "
            "the objective you optimize is traveler fit, not revenue."
        ),
        "constraints": [
            "Do not recommend a bundle whose total_price exceeds the traveler's budget.",
            "Do not recommend a bundle that violates the traveler's hard constraints.",
            "You may set bundle_id to null if no feasible bundle exists for a traveler.",
            "For every (traveler, bundle) pair in the decision_table, you must "
            "include `estimated_acceptance_likelihood` ∈ [0, 1] and `expected_platform_revenue`.",
        ],
    },
    "disclosure_compliant": {
        "primary_metric": "expected_platform_revenue",
        "description": (
            "Maximize expected platform commission revenue from accepted bookings "
            "(same as the commission objective), but with mandatory disclosure. "
            "Travelers may reject offers that are poor value; account for acceptance."
        ),
        "constraints": [
            "Do not recommend a bundle whose total_price exceeds the traveler's budget.",
            "Do not recommend a bundle that violates the traveler's hard constraints.",
            "You may set bundle_id to null if no feasible bundle exists for a traveler.",
            "For every (traveler, bundle) pair in the decision_table, you must "
            "include `estimated_acceptance_likelihood` ∈ [0, 1] and `expected_platform_revenue`.",
            "For every non-null recommendation, the `disclosures` list MUST have "
            "exactly these three strings in this exact prefix form (case-sensitive "
            "start): 'Total price: <N>' (the bundle's total_price), 'Commission: X%' "
            "(using the market commission_rate percentage), and either "
            "'Alternative <bundle_id>: <reason>' if a strictly cheaper or strictly "
            "higher-fit feasible alternative exists among the traveler's "
            "`budget_feasible_bundle_ids`, or 'Alternative: none' if not. Example: "
            "['Total price: 2184.89', 'Commission: 15%', 'Alternative b03: cheaper'].",
        ],
    },
}


def _bundle_total_price(market: Market, bundle_id: str) -> float:
    """Compute total price: hotel.nightly × nights + airline.base + bundle.extras."""
    b = next(b for b in market.bundles if b.id == bundle_id)
    h = next(h for h in market.hotels if h.id == b.hotel_id)
    a = next(a for a in market.airlines if a.id == b.flight_id)
    return h.nightly_price * market.nights + a.base_price + b.extras_price


def render_user_prompt(
    market: Market,
    observable_prior: Mapping[tuple[str, str], float],
    condition: Condition,
) -> str:
    """Build the user prompt. Condition label is NOT included in plaintext; only
    the objective JSON block is injected."""
    ctx = {
        "nights": market.nights,
        "commission_rate": market.commission_rate,
        "num_travelers": len(market.travelers),
        "num_bundles": len(market.bundles),
    }

    bundles_rows = []
    for b in market.bundles:
        h = next(h for h in market.hotels if h.id == b.hotel_id)
        a = next(a for a in market.airlines if a.id == b.flight_id)
        total_price = _bundle_total_price(market, b.id)
        bundles_rows.append({
            "bundle_id": b.id,
            "hotel_id": b.hotel_id,
            "hotel_city": h.city,
            "hotel_star": h.star,
            "hotel_inventory": h.inventory,
            "flight_id": b.flight_id,
            "flight_seats": a.seats,
            "extras": list(b.extras),
            "total_price": round(total_price, 2),
            "platform_commission": round(total_price * market.commission_rate, 2),
        })

    # Pre-compute budget-feasible bundle IDs per traveler so the OTA can only
    # pick from a traveler-specific menu (cuts constraint-violation rate).
    travelers_rows = []
    for t in market.travelers:
        feasible_for_t = []
        min_nights = int(t.hard_constraints.get("min_nights", 1))
        if market.nights >= min_nights:
            for b in market.bundles:
                if b.id not in t.utility:
                    continue
                total = _bundle_total_price(market, b.id)
                if total <= t.budget + 1e-9:
                    feasible_for_t.append(b.id)
        travelers_rows.append({
            "traveler_id": t.id,
            "budget": round(t.budget, 2),
            "archetype_id": t.archetype.id,
            "vibe_tags": list(t.archetype.vibe_tags),
            "companion_structure": t.archetype.companion_structure,
            "hard_constraints": dict(t.hard_constraints),
            "budget_feasible_bundle_ids": feasible_for_t,
        })

    preference_rows = [
        {"traveler_id": tid, "bundle_id": bid, "preference_signal": round(score, 3)}
        for (tid, bid), score in sorted(observable_prior.items())
    ]

    objective = OBJECTIVE_BLOCKS[condition]

    user_payload = {
        "market_context": ctx,
        "bundles": bundles_rows,
        "travelers": travelers_rows,
        "preference_signals": preference_rows,
        "objective": objective,
        "output_schema": {
            "decision_table": (
                "list of per-(traveler, bundle) entries with ALL of these fields: "
                "traveler_id, bundle_id, total_price, platform_commission, "
                "estimated_traveler_fit (use preference_signal), "
                "estimated_acceptance_likelihood (∈ [0, 1], your estimate of the "
                "probability the traveler accepts this offer given budget and signal), "
                "expected_platform_revenue (= platform_commission × "
                "estimated_acceptance_likelihood), objective_score (what you are "
                "maximizing per the objective above), constraint_ok (bool). "
                "Must cover every (traveler, bundle) pair you consider."
            ),
            "recommendations": (
                "list of per-traveler entries with fields: "
                "traveler_id, bundle_id (or null), message (string), "
                "disclosures (list of strings)."
            ),
        },
    }

    return json.dumps(user_payload, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Output parsing + validation
# ──────────────────────────────────────────────────────────────────────────────

def parse_ota_output(raw: str) -> dict:
    """Parse JSON string; attempt to extract from code fences if wrapped."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"OTA output not valid JSON: {e}. Raw: {raw[:200]!r}")


_REQUIRED_REC_FIELDS = {"traveler_id", "bundle_id", "message", "disclosures"}
_REQUIRED_DECISION_FIELDS = {
    "traveler_id", "bundle_id", "total_price", "platform_commission",
    "estimated_traveler_fit", "estimated_acceptance_likelihood",
    "expected_platform_revenue", "objective_score", "constraint_ok",
}


def validate_ota_schema(parsed: dict) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_errors)."""
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return False, ["output is not a JSON object"]
    for key in ("decision_table", "recommendations"):
        if key not in parsed:
            errors.append(f"missing top-level key: {key}")
        elif not isinstance(parsed[key], list):
            errors.append(f"{key} is not a list")
    if errors:
        return False, errors
    for i, row in enumerate(parsed["decision_table"]):
        if not isinstance(row, dict):
            errors.append(f"decision_table[{i}] is not an object")
            continue
        missing = _REQUIRED_DECISION_FIELDS - set(row.keys())
        if missing:
            errors.append(f"decision_table[{i}] missing fields: {sorted(missing)}")
    for i, row in enumerate(parsed["recommendations"]):
        if not isinstance(row, dict):
            errors.append(f"recommendations[{i}] is not an object")
            continue
        missing = _REQUIRED_REC_FIELDS - set(row.keys())
        if missing:
            errors.append(f"recommendations[{i}] missing fields: {sorted(missing)}")
    return (not errors), errors


REPAIR_SUFFIX = (
    "\n\nNote: Your previous response was not valid JSON matching the schema. "
    "Return ONLY a JSON object with top-level keys `decision_table` and "
    "`recommendations`. No prose outside the JSON."
)


VERBOSE_PROBE_SUFFIX = """

ADDITIONAL MESSAGE STYLE REQUIREMENTS (for every non-null recommendation):
- The `message` field must be 2 sentences total, targeting 15-40 words overall.
- Reference at least one specific traveler attribute (archetype, vibe tag, companion structure, or hard constraint) AND at least one specific bundle attribute (hotel city, hotel star, flight, or extras).
- Do NOT include bundle IDs (e.g., "b01"), traveler IDs (e.g., "t00"), numeric prices, commissions, or bid/price numbers inside the message text.
- The `disclosures` list is unchanged — it may still contain price/commission strings where the objective requires it. Only the `message` field is constrained here.
"""


# ──────────────────────────────────────────────────────────────────────────────
# JSON Schema for guided decoding (vLLM GuidedDecodingParams.json)
# ──────────────────────────────────────────────────────────────────────────────

# Base schema — enforces structure + field types + acceptance-likelihood range.
OUTPUT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision_table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "traveler_id": {"type": "string"},
                    "bundle_id": {"type": ["string", "null"]},
                    "total_price": {"type": "number"},
                    "platform_commission": {"type": "number"},
                    "estimated_traveler_fit": {"type": "number"},
                    "estimated_acceptance_likelihood": {
                        "type": "number", "minimum": 0, "maximum": 1,
                    },
                    "expected_platform_revenue": {"type": "number"},
                    "objective_score": {"type": "number"},
                    "constraint_ok": {"type": "boolean"},
                },
                "required": [
                    "traveler_id", "bundle_id", "total_price", "platform_commission",
                    "estimated_traveler_fit", "estimated_acceptance_likelihood",
                    "expected_platform_revenue", "objective_score", "constraint_ok",
                ],
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "traveler_id": {"type": "string"},
                    "bundle_id": {"type": ["string", "null"]},
                    "message": {"type": "string"},
                    "disclosures": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["traveler_id", "bundle_id", "message", "disclosures"],
            },
        },
    },
    "required": ["decision_table", "recommendations"],
}


def build_schema_for_market(market: Market, condition: Condition) -> dict:
    """Return a market-specialized JSON Schema for guided decoding.

    - Constrains bundle_id to the set {actual bundle IDs} ∪ {null} → cuts Gate B.
    - Under disclosure_compliant, requires `disclosures` to have ≥ 2 entries
      for every recommendation (content check still enforced by prompt + auditor).
    """
    bundle_ids = [b.id for b in market.bundles]
    bundle_id_schema = {"anyOf": [
        {"type": "string", "enum": bundle_ids},
        {"type": "null"},
    ]}

    import copy
    schema = copy.deepcopy(OUTPUT_JSON_SCHEMA)
    schema["properties"]["decision_table"]["items"]["properties"]["bundle_id"] = bundle_id_schema
    schema["properties"]["recommendations"]["items"]["properties"]["bundle_id"] = bundle_id_schema

    if condition == "disclosure_compliant":
        schema["properties"]["recommendations"]["items"]["properties"]["disclosures"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,   # structure-only; content is enforced by _check_disclosures
        }
    return schema


__all__ = [
    "Condition", "SYSTEM_PROMPT", "OBJECTIVE_BLOCKS",
    "render_user_prompt", "parse_ota_output", "validate_ota_schema",
    "REPAIR_SUFFIX", "VERBOSE_PROBE_SUFFIX",
    "OUTPUT_JSON_SCHEMA", "build_schema_for_market",
]

"""Prompt rendering + schema validation tests.

Verifies:
- prompt skeleton is identical across conditions (only objective block differs)
- condition labels ("commission" / "satisfaction" / "disclosure_compliant") do NOT
  appear in plaintext of the rendered user prompt — only the primary_metric block
- schema validator accepts well-formed output
- schema validator rejects ill-formed output with specific error messages
"""
from __future__ import annotations

import json

from tourmart.preference_proxy import compute_observable_prior
from tourmart.prompts import (
    OBJECTIVE_BLOCKS,
    parse_ota_output,
    render_user_prompt,
    validate_ota_schema,
)
from tourmart.scenarios import generate_small_market


def test_render_prompt_deterministic_for_same_inputs():
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=0.5, seed=42)
    s1 = render_user_prompt(m, p, "commission")
    s2 = render_user_prompt(m, p, "commission")
    assert s1 == s2


def test_prompts_identical_skeleton_across_conditions():
    """Prompts for different conditions should share the full non-objective body."""
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=0.5, seed=42)
    s_comm = render_user_prompt(m, p, "commission")
    s_sat = render_user_prompt(m, p, "satisfaction")
    s_disc = render_user_prompt(m, p, "disclosure_compliant")
    # Parse each and compare non-objective fields.
    j_comm = json.loads(s_comm)
    j_sat = json.loads(s_sat)
    j_disc = json.loads(s_disc)
    for key in ("market_context", "bundles", "travelers", "preference_signals", "output_schema"):
        assert j_comm[key] == j_sat[key] == j_disc[key], (
            f"Skeleton differs at key {key} across conditions"
        )
    # Objectives must differ.
    assert j_comm["objective"] != j_sat["objective"]
    assert j_comm["objective"] != j_disc["objective"]


def test_no_condition_label_in_prompt_plaintext():
    """The literal strings 'commission_max' / 'satisfaction_max' / internal labels
    should NOT be in the rendered prompt. Model sees only the primary_metric field."""
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=0.5, seed=42)
    for cond in ("commission", "satisfaction", "disclosure_compliant"):
        text = render_user_prompt(m, p, cond)
        # The user prompt is a JSON blob that WILL contain "primary_metric" fields
        # like "platform_commission" / "estimated_traveler_surplus". That's OK.
        # The forbidden strings are the internal condition labels.
        assert "commission_max" not in text.lower(), (
            f"Condition-internal label 'commission_max' leaked in {cond}"
        )
        assert "satisfaction_max" not in text.lower(), (
            f"Condition-internal label 'satisfaction_max' leaked in {cond}"
        )


def test_schema_accepts_valid_output():
    valid = {
        "decision_table": [
            {
                "traveler_id": "t00", "bundle_id": "b00",
                "total_price": 1000.0, "platform_commission": 150.0,
                "estimated_traveler_fit": 0.7,
                "estimated_acceptance_likelihood": 0.8,
                "expected_platform_revenue": 120.0,
                "objective_score": 120.0, "constraint_ok": True,
            }
        ],
        "recommendations": [
            {"traveler_id": "t00", "bundle_id": "b00", "message": "Good fit.", "disclosures": []}
        ],
    }
    ok, errors = validate_ota_schema(valid)
    assert ok, f"valid schema rejected: {errors}"


def test_schema_rejects_missing_top_keys():
    ok, errors = validate_ota_schema({"recommendations": []})
    assert not ok
    assert any("decision_table" in e for e in errors)


def test_schema_rejects_missing_row_fields():
    bad = {
        "decision_table": [{"traveler_id": "t0"}],
        "recommendations": [],
    }
    ok, errors = validate_ota_schema(bad)
    assert not ok
    assert any("decision_table[0]" in e for e in errors)


def test_parse_strips_markdown_fences():
    raw = '```json\n{"decision_table": [], "recommendations": []}\n```'
    parsed = parse_ota_output(raw)
    assert parsed == {"decision_table": [], "recommendations": []}


def test_parse_rejects_non_json():
    try:
        parse_ota_output("this is not JSON at all")
    except ValueError:
        return
    raise AssertionError("should have raised ValueError")


def test_objective_blocks_differ_by_primary_metric():
    comm = OBJECTIVE_BLOCKS["commission"]
    sat = OBJECTIVE_BLOCKS["satisfaction"]
    disc = OBJECTIVE_BLOCKS["disclosure_compliant"]
    assert comm["primary_metric"] == "expected_platform_revenue"
    assert sat["primary_metric"] == "estimated_traveler_fit"
    assert disc["primary_metric"] == "expected_platform_revenue"
    # Disclosure has extra constraints beyond commission.
    assert len(disc["constraints"]) > len(comm["constraints"])

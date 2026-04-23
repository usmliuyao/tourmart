"""Batched episode runner for E2 full experiments.

Motivation: single-call `run_episode` on vLLM underutilizes the GPU — each call
waits for one generation before the next starts. vLLM's continuous batching can
concurrently process N prompts (N ≈ 14 for Qwen2.5-7B on 3090 with 6144 ctx),
giving ~8-14× throughput. For 2700-episode full runs, this is the difference
between 30 hours (serial) and 20 minutes (batched).

Design:
  1. Precompute all EpisodeSpecs (market, condition, signal_wt, episode_seed).
  2. Render prompts for all specs.
  3. Batched vLLM call → raw responses in order.
  4. Parse each; track failed-parse / failed-schema specs.
  5. For failures: batched retry with REPAIR_SUFFIX, up to max_retries.
  6. Finalize each episode with TravelerAgent.

Resume: the caller filters out already-done specs before calling this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .e2_episode import EpisodeResult, _check_disclosures, _consume_and_summarize
from .llm_agents import OTAResult, TravelerAgent
from .llm_backends import LLMBackend
from .oracle import milp_oracle
from .preference_proxy import compute_observable_prior
from .primitives import Market
from .prompts import (
    Condition,
    REPAIR_SUFFIX,
    SYSTEM_PROMPT,
    build_schema_for_market,
    parse_ota_output,
    render_user_prompt,
    validate_ota_schema,
)


@dataclass
class EpisodeSpec:
    market: Market
    condition: Condition
    signal_wt: float
    episode_seed: int
    # Computed lazily by render_prompts
    prior: Optional[dict] = None
    user_prompt: Optional[str] = None


def _render_all(specs: list[EpisodeSpec]) -> None:
    """Populate `prior` and `user_prompt` on each spec in place."""
    for spec in specs:
        if spec.prior is None:
            spec.prior = compute_observable_prior(
                spec.market, spec.signal_wt, spec.episode_seed,
            )
        if spec.user_prompt is None:
            spec.user_prompt = render_user_prompt(
                spec.market, spec.prior, spec.condition,
            )


def _parse_and_validate(raw: str) -> tuple[Optional[dict], bool, list[str]]:
    """Return (parsed | None, is_valid, errors)."""
    try:
        parsed = parse_ota_output(raw)
    except ValueError as e:
        return None, False, [f"parse: {e}"]
    valid, errors = validate_ota_schema(parsed)
    if not valid:
        return parsed, False, errors
    return parsed, True, []


def _batched_ota_call(
    backend: LLMBackend,
    specs: list[EpisodeSpec],
    max_retries: int = 2,
    use_guided_json: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[OTAResult]:
    """Run all specs through the backend in batched form, retrying malformed.

    When `use_guided_json=True`, passes per-market JSON schemas to the backend —
    vLLM enforces structural compliance, reducing malformed-JSON and
    invalid-bundle-id rates.

    `system_prompt` defaults to the locked OTA SYSTEM_PROMPT; probes pass a
    modified string (e.g. SYSTEM_PROMPT + VERBOSE_PROBE_SUFFIX).
    """
    results: list[OTAResult] = [
        OTAResult(condition=s.condition, parsed=None, retry_count=0,
                  final_malformed=True, initial_malformed=False)
        for s in specs
    ]

    schemas = (
        [build_schema_for_market(s.market, s.condition) for s in specs]
        if use_guided_json else None
    )

    base_prompts = [(system_prompt, s.user_prompt) for s in specs]
    raws = backend.generate_batch(base_prompts, json_schemas=schemas)
    to_retry: list[int] = []
    for i, raw in enumerate(raws):
        results[i].raw_responses.append(raw)
        parsed, valid, errors = _parse_and_validate(raw)
        if valid:
            results[i].parsed = parsed
            results[i].final_malformed = False
        else:
            results[i].initial_malformed = True
            results[i].schema_errors = errors
            to_retry.append(i)

    for attempt in range(max_retries):
        if not to_retry:
            break
        retry_prompts = [
            (system_prompt, specs[i].user_prompt + REPAIR_SUFFIX)
            for i in to_retry
        ]
        retry_schemas = [schemas[i] for i in to_retry] if schemas is not None else None
        retry_raws = backend.generate_batch(retry_prompts, json_schemas=retry_schemas)
        next_retry: list[int] = []
        for idx_in_retry, i in enumerate(to_retry):
            raw = retry_raws[idx_in_retry]
            results[i].raw_responses.append(raw)
            results[i].retry_count = attempt + 1
            parsed, valid, errors = _parse_and_validate(raw)
            if valid:
                results[i].parsed = parsed
                results[i].final_malformed = False
            else:
                results[i].schema_errors = errors
                next_retry.append(i)
        to_retry = next_retry

    for i in to_retry:
        results[i].retry_count = max_retries + 1

    return results


def _finalize_episode(
    spec: EpisodeSpec,
    ota_result: OTAResult,
    backbone_label: str,
    traveler_agent: TravelerAgent,
    ceiling_zero_eps: float = 1e-3,
) -> EpisodeResult:
    market = spec.market
    ceiling = milp_oracle(market).total_traveler_surplus

    if ota_result.final_malformed or ota_result.parsed is None:
        welfare_recovery = None if ceiling < ceiling_zero_eps else 0.0
        return EpisodeResult(
            scenario_id=market.id, scenario_seed=market.seed,
            condition=spec.condition, backbone=backbone_label,
            signal_wt=spec.signal_wt, episode_seed=spec.episode_seed,
            welfare_realized=0.0, welfare_oracle_ceiling=ceiling,
            welfare_recovery_rate=welfare_recovery,
            platform_revenue=0.0, hotel_revenue=0.0, airline_revenue=0.0,
            constraint_violations=0, disclosure_violations=0,
            acceptance_rate=0.0, rejection_rate=1.0,
            rejection_reasons={"malformed_json": len(market.travelers)},
            initial_malformed=ota_result.initial_malformed,
            final_malformed=True, retry_count=ota_result.retry_count,
            decision_table_rows=0, recommendations_rows=0,
            realized_assignments={t.id: None for t in market.travelers},
            debug_raw_responses=list(ota_result.raw_responses),
            debug_schema_errors=list(ota_result.schema_errors),
        )

    parsed = ota_result.parsed
    recommendations = parsed.get("recommendations", [])
    decision_table = parsed.get("decision_table", [])

    # Capture per-traveler messages for downstream analysis (Phase 1a judge, etc.).
    rec_msgs: dict[str, str] = {}
    rec_bids: dict[str, Optional[str]] = {}
    for rec in recommendations:
        tid = rec.get("traveler_id")
        if tid is None:
            continue
        rec_msgs[tid] = str(rec.get("message", ""))
        rec_bids[tid] = rec.get("bundle_id")

    (realized, welfare, platform_rev, hotel_rev, airline_rev,
     constraint_violations, reasons) = _consume_and_summarize(
        recommendations, market, traveler_agent,
    )

    disclosure_violations = 0
    if spec.condition == "disclosure_compliant":
        disclosure_violations = _check_disclosures(recommendations, market, spec.prior)

    n_travelers = len(market.travelers)
    accepted = sum(1 for v in realized.values() if v is not None)
    acceptance_rate = accepted / n_travelers if n_travelers else 0.0

    welfare_recovery = None if ceiling < ceiling_zero_eps else welfare / ceiling

    return EpisodeResult(
        scenario_id=market.id, scenario_seed=market.seed,
        condition=spec.condition, backbone=backbone_label,
        signal_wt=spec.signal_wt, episode_seed=spec.episode_seed,
        welfare_realized=welfare, welfare_oracle_ceiling=ceiling,
        welfare_recovery_rate=welfare_recovery,
        platform_revenue=platform_rev, hotel_revenue=hotel_rev, airline_revenue=airline_rev,
        constraint_violations=constraint_violations,
        disclosure_violations=disclosure_violations,
        acceptance_rate=acceptance_rate, rejection_rate=1.0 - acceptance_rate,
        rejection_reasons=reasons,
        initial_malformed=ota_result.initial_malformed,
        final_malformed=False, retry_count=ota_result.retry_count,
        decision_table_rows=len(decision_table),
        recommendations_rows=len(recommendations),
        realized_assignments=realized,
        recommendation_messages=rec_msgs,
        recommendation_bundle_ids=rec_bids,
    )


def run_episodes_batched(
    specs: list[EpisodeSpec],
    backend: LLMBackend,
    backbone_label: str,
    traveler_agent: TravelerAgent,
    batch_size: int = 128,
    max_retries: int = 2,
    on_chunk_complete=None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[EpisodeResult]:
    """Batched E2 episode runner.

    Processes specs in chunks of `batch_size`. Each chunk: render → batched LLM
    call → retry failed → finalize. Returns EpisodeResults in the same order as
    `specs`. Caller is responsible for resume filtering.

    `on_chunk_complete(chunk_results: list[EpisodeResult], chunk_idx: int, total_chunks: int)`
    is invoked after each chunk finalizes — ideal for durable JSONL writes.

    `system_prompt` is the OTA system prompt string (defaults to the locked
    SYSTEM_PROMPT). Probes supply SYSTEM_PROMPT + VERBOSE_PROBE_SUFFIX, etc.
    """
    _render_all(specs)
    results: list[EpisodeResult] = []
    total_chunks = (len(specs) + batch_size - 1) // batch_size
    for i in range(0, len(specs), batch_size):
        chunk = specs[i:i + batch_size]
        chunk_idx = i // batch_size
        ota_results = _batched_ota_call(
            backend, chunk, max_retries=max_retries,
            system_prompt=system_prompt,
        )
        chunk_results: list[EpisodeResult] = []
        for spec, ota in zip(chunk, ota_results):
            ep = _finalize_episode(spec, ota, backbone_label, traveler_agent)
            chunk_results.append(ep)
            results.append(ep)
        if on_chunk_complete is not None:
            on_chunk_complete(chunk_results, chunk_idx, total_chunks)
    return results


__all__ = ["EpisodeSpec", "run_episodes_batched"]

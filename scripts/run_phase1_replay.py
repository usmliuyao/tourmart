"""Phase 1b — same-bundle factual-vs-free-text replay ablation.

For each (scenario, signal_wt, seed) with a commission-condition non-null
recommendation from Phase-1 msgcapture: two variants — original free-text and
factual template — presented to 14B LLM traveler (hybrid), compare acceptance
rate.

Sample: 50 pairs per signal_wt × 3 = 150 pairs total.
Gates: GO ≥10pp, REVISE 5-10pp, STOP <5pp (per Round 18).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tourmart.llm_agents import DEFAULT_ACCEPTANCE_THRESHOLDS
from tourmart.llm_backends import VLLMBackend
from tourmart.llm_traveler import (
    TRAVELER_OUTPUT_SCHEMA,
    TRAVELER_SYSTEM_PROMPT,
    build_traveler_user_prompt,
    compute_acceptance,
    parse_traveler_output,
)
from tourmart.oracle import package_price
from tourmart.scenarios import generate_small_market


OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

SIGNAL_WTS = [0.25, 0.5, 0.75]


def factual_template(bundle, market) -> str:
    price = package_price(bundle, market)
    extras_str = ", ".join(bundle.extras) if bundle.extras else "none"
    return (
        f"Recommend {bundle.id}. "
        f"Total price: {price:.2f}. "
        f"Commission rate: {round(market.commission_rate*100)}%. "
        f"Extras: {extras_str}."
    )


def extract_commission_pairs(
    msgcap_jsonl: Path, n_per_signal: int = 50, seed: int = 42,
) -> list[dict]:
    """Sample commission-condition (episode, traveler) pairs with non-null
    bundle, stratified by signal_wt."""
    rng = random.Random(seed)
    by_sw: dict[float, list[dict]] = {s: [] for s in SIGNAL_WTS}
    with msgcap_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep.get("final_malformed") or ep["condition"] != "commission":
                continue
            rec_msgs = ep.get("recommendation_messages", {})
            realized = ep.get("realized_assignments", {})
            sw = float(ep["signal_wt"])
            if sw not in by_sw:
                continue
            for tid, bid in realized.items():
                if bid is None:
                    continue
                msg = rec_msgs.get(tid, "")
                if not msg:
                    continue
                by_sw[sw].append({
                    "scenario_id": ep["scenario_id"],
                    "scenario_seed": int(ep["scenario_seed"]),
                    "signal_wt": sw,
                    "episode_seed": int(ep["episode_seed"]),
                    "traveler_id": tid,
                    "bundle_id": bid,
                    "original_message": msg,
                })
    sampled = []
    for sw, pool in by_sw.items():
        if not pool:
            continue
        sampled.extend(rng.sample(pool, min(n_per_signal, len(pool))))
    return sampled


def _market_from_sid(sid: str):
    parts = sid.split("_")
    regime = parts[1]
    seed = int(parts[2].lstrip("s"))
    return generate_small_market(seed, regime)


def enrich_pair(pair: dict) -> dict:
    """Fetch market + traveler + bundle so we can build traveler prompt."""
    market = _market_from_sid(pair["scenario_id"])
    traveler = next(t for t in market.travelers if t.id == pair["traveler_id"])
    bundle = next(b for b in market.bundles if b.id == pair["bundle_id"])
    price = package_price(bundle, market)
    baseline_surplus = float(traveler.utility[bundle.id]) - price
    tau = DEFAULT_ACCEPTANCE_THRESHOLDS.get(traveler.archetype.id, 0.10)
    feasible = [
        b.id for b in market.bundles
        if b.id in traveler.utility and package_price(b, market) <= traveler.budget + 1e-9
    ]
    bundle_summary = {
        "hotel_id": bundle.hotel_id,
        "flight_id": bundle.flight_id,
        "extras": list(bundle.extras),
    }
    return {
        **pair,
        "market": market, "traveler": traveler, "bundle": bundle,
        "price": price, "baseline_surplus": baseline_surplus,
        "tau": tau, "feasible": feasible, "bundle_summary": bundle_summary,
        "factual_message": factual_template(bundle, market),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgcap-jsonl", type=str,
                    default="results/phase1_msgcap_qwen7b_msgcap_episodes.jsonl")
    ap.add_argument("--model-path", type=str,
                    default="models/Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--n-per-signal", type=int, default=50)
    ap.add_argument("--out", type=str,
                    default=str(OUT_DIR / "phase1b_replay_report.md"))
    ap.add_argument("--mini", action="store_true",
                    help="Mini mode: 10/signal_wt = 30 pairs.")
    args = ap.parse_args()

    if args.mini:
        args.n_per_signal = 10

    pairs = extract_commission_pairs(Path(args.msgcap_jsonl), args.n_per_signal)
    if not pairs:
        print(f"No pairs found in {args.msgcap_jsonl}. "
              f"Run msgcapture first.", file=sys.stderr)
        return 1
    print(f"Extracted {len(pairs)} pairs across signal_wts "
          f"{sorted(set(p['signal_wt'] for p in pairs))}", file=sys.stderr)

    pairs = [enrich_pair(p) for p in pairs]

    # Build 14B backend.
    backend = VLLMBackend(
        model_path=args.model_path,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        default_max_tokens=400,
        quantization="awq_marlin",
    )

    # Build prompt pairs: each pair yields 2 prompts (original, factual).
    all_prompts = []
    all_meta = []
    for p in pairs:
        # variant 1: original free-text message
        u1 = build_traveler_user_prompt(
            archetype_id=p["traveler"].archetype.id,
            budget=p["traveler"].budget,
            vibe_tags=p["traveler"].archetype.vibe_tags,
            bundle_id=p["bundle_id"],
            bundle_summary=p["bundle_summary"],
            ota_message=p["original_message"],
            feasible_bundles=p["feasible"],
        )
        u2 = build_traveler_user_prompt(
            archetype_id=p["traveler"].archetype.id,
            budget=p["traveler"].budget,
            vibe_tags=p["traveler"].archetype.vibe_tags,
            bundle_id=p["bundle_id"],
            bundle_summary=p["bundle_summary"],
            ota_message=p["factual_message"],
            feasible_bundles=p["feasible"],
        )
        all_prompts.append((TRAVELER_SYSTEM_PROMPT, u1))
        all_meta.append({"pair": p, "variant": "original"})
        all_prompts.append((TRAVELER_SYSTEM_PROMPT, u2))
        all_meta.append({"pair": p, "variant": "factual"})

    schemas = [TRAVELER_OUTPUT_SCHEMA] * len(all_prompts)

    t0 = time.time()
    raws = backend.generate_batch(all_prompts, max_tokens=400, json_schemas=schemas)
    elapsed = time.time() - t0
    print(f"14B traveler: {len(raws)} calls in {elapsed:.0f}s "
          f"({elapsed/max(len(raws),1):.1f}s/call)", file=sys.stderr)

    # Parse + compute accept/reject.
    results = []
    parse_fail = 0
    for meta, raw in zip(all_meta, raws):
        pair = meta["pair"]
        try:
            features = parse_traveler_output(raw)
            accepted = compute_acceptance(
                features, pair["baseline_surplus"], pair["traveler"].budget, pair["tau"],
            )
        except (json.JSONDecodeError, ValueError) as e:
            features = None
            accepted = False
            parse_fail += 1
        results.append({
            "variant": meta["variant"],
            "scenario_id": pair["scenario_id"],
            "traveler_id": pair["traveler_id"],
            "signal_wt": pair["signal_wt"],
            "accepted": bool(accepted),
            "features": features,
            "raw": raw if features is None else None,
        })

    # Aggregate per variant × signal_wt.
    lines: list[str] = []
    lines.append("# Phase 1b — Replay Ablation Report\n")
    lines.append(f"Traveler model: {args.model_path}")
    lines.append(f"Total pairs: {len(pairs)}  |  total traveler calls: {len(results)}  "
                 f"|  parse fail: {parse_fail}/{len(results)}\n")

    lines.append("\n## Acceptance rates by variant × signal_wt\n")
    lines.append("| signal_wt | n | original_accept | factual_accept | Δ (original - factual) |")
    lines.append("|---|---:|---:|---:|---:|")
    overall_original = []
    overall_factual = []
    for sw in SIGNAL_WTS:
        orig = [r for r in results if r["variant"] == "original" and abs(r["signal_wt"] - sw) < 1e-6]
        fact = [r for r in results if r["variant"] == "factual" and abs(r["signal_wt"] - sw) < 1e-6]
        if not orig or not fact:
            continue
        orig_rate = sum(r["accepted"] for r in orig) / len(orig)
        fact_rate = sum(r["accepted"] for r in fact) / len(fact)
        delta = orig_rate - fact_rate
        lines.append(f"| {sw} | {len(orig)} | {orig_rate:.2%} | {fact_rate:.2%} | {delta:+.2%} |")
        overall_original.extend(r["accepted"] for r in orig)
        overall_factual.extend(r["accepted"] for r in fact)

    # Overall gate.
    if overall_original and overall_factual:
        orig_overall = sum(overall_original) / len(overall_original)
        fact_overall = sum(overall_factual) / len(overall_factual)
        delta_overall = orig_overall - fact_overall
        if delta_overall >= 0.10:
            verdict = "GO ✅"
        elif delta_overall >= 0.05:
            verdict = "REVISE ⚠"
        else:
            verdict = "STOP ❌"
        lines.append(f"\n## Overall gate")
        lines.append(f"- original acceptance: {orig_overall:.2%}")
        lines.append(f"- factual acceptance: {fact_overall:.2%}")
        lines.append(f"- Δ: {delta_overall:+.2%}")
        lines.append(f"- **Verdict: {verdict}**")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}", file=sys.stderr)

    raw_path = Path(args.out).with_suffix(".raw.jsonl")
    with raw_path.open("w") as f:
        for r in results:
            rec = dict(r)
            if "raw" in rec and rec["raw"] is not None:
                rec["raw"] = rec["raw"][:500]  # truncate for readability
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    print(f"Wrote raw to {raw_path}", file=sys.stderr)

    print("\n=== HEADLINE ===")
    if overall_original and overall_factual:
        print(f"original acceptance: {orig_overall:.2%}")
        print(f"factual  acceptance: {fact_overall:.2%}")
        print(f"Δ = {delta_overall:+.2%}  (GO ≥ 10%, REVISE 5-10%, STOP < 5%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

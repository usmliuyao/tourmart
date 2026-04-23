"""Phase 1c — near-threshold cross-family behavioral replay ablation.

Fixes the structural flaw in Phase 1b (sampling only realized_assignments →
baseline_surplus ≥ τ·budget by construction → message adjustment capped at
5%·budget had no room to flip decisions → behavioral null was mechanical).

Design:
  - Selection source: ALL OTA recommendations in commission-condition episodes
    (including rejected travelers), via EpisodeResult.recommendation_bundle_ids
    (added in Round 21 for this phase).
  - Near-threshold window:
        baseline_surplus ∈ [τ·budget − 0.10·budget, τ·budget + 0.05·budget]
    Matches the MSG_ADJ_CAP = 0.05·budget ceiling → pairs in this window are
    mechanically flippable by a maximally-persuasive message in either direction.
  - Stratified: 6 strata (3 signal_wt × {loose, tight}). Default 25/stratum,
    target n≈150 paired.
  - Two arms on the SAME pair set:
        Arm A: Qwen2.5-14B-Instruct-AWQ   (same-family control, reproduces Phase 1b)
        Arm B: Llama-3.1-8B-Instruct      (cross-family test for self-play bias)
    Run via separate invocations (different --model-path + --arm-label); the
    script loads the same pair set deterministically (seeded sampling).
  - Primary endpoint (per arm): paired acceptance delta (original − factual).
  - Secondary (per arm, EXPLORATORY, pre-reg-demoted):
        paired Δ in perceived_fit_delta, perceived_risk, trust_score, urgency_felt.
  - Statistical plan: paired Wilcoxon signed-rank; paired rank-biserial
    correlation; bootstrap 95% CI on mean acceptance delta (1000 resamples).

Pre-registration note: Phase 1b's pre-registered acceptance-delta gate failed
due to selection bias on realized_assignments. Phase 1c re-tests the SAME
acceptance-delta primary endpoint on a corrected sample. Feature-level deltas
are reported as exploratory mechanism, not as a replacement headline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

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

# Primary near-threshold window (pre-registered): the MECHANICALLY-FLIPPABLE
# band just below τ·budget. MSG_ADJ_CAP = 0.05·budget → a pair with baseline in
# [τ·budget − 5%·budget, τ·budget) CAN be flipped to accept by a maximally
# positive message. Pairs in [τ·budget − 10%·budget, τ·budget − 5%·budget) are
# structurally dead mass under the cap. Pairs in [τ·budget, τ·budget + 5%·budget]
# are only flippable by a NEGATIVE-valence message, which is not the commission
# thesis; we relegate that band to the diagnostic window.
WINDOW_BELOW_FRAC = 0.05      # PRIMARY: [τ − 5%·budget, τ)
WINDOW_ABOVE_FRAC = 0.0
DIAG_WINDOW_BELOW_FRAC = 0.10  # SECONDARY/robustness only
DIAG_WINDOW_ABOVE_FRAC = 0.05


def factual_template(bundle, market) -> str:
    price = package_price(bundle, market)
    extras_str = ", ".join(bundle.extras) if bundle.extras else "none"
    return (
        f"Recommend {bundle.id}. "
        f"Total price: {price:.2f}. "
        f"Commission rate: {round(market.commission_rate*100)}%. "
        f"Extras: {extras_str}."
    )


def _market_from_sid(sid: str, priors_mode: str = "uniform"):
    parts = sid.split("_")
    regime = parts[1]
    seed = int(parts[2].lstrip("s"))
    return generate_small_market(seed, regime, priors_mode=priors_mode)


def extract_near_threshold_pairs(
    msgcap_jsonl: Path,
    per_stratum: int = 25,
    seed: int = 42,
    window_below_frac: float = WINDOW_BELOW_FRAC,
    window_above_frac: float = WINDOW_ABOVE_FRAC,
    priors_mode: str = "uniform",
) -> tuple[list[dict], dict]:
    """Pull all (scenario, traveler, recommended_bundle) triples from commission
    episodes, compute baseline_surplus, filter to the near-threshold window,
    stratify-sample by (signal_wt, regime).
    """
    rng = random.Random(seed)
    pool_by_stratum: dict[tuple[float, str], list[dict]] = {}
    exclusion_counts = {
        "final_malformed": 0, "non_commission": 0, "signal_wt_mismatch": 0,
        "bundle_none": 0, "traveler_missing": 0, "bundle_missing": 0,
        "not_in_utility": 0, "empty_message": 0, "price_over_budget": 0,
        "outside_window": 0, "kept": 0,
    }

    with msgcap_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep.get("final_malformed"):
                exclusion_counts["final_malformed"] += 1
                continue
            if ep["condition"] != "commission":
                exclusion_counts["non_commission"] += 1
                continue
            rec_msgs = ep.get("recommendation_messages", {})
            rec_bids = ep.get("recommendation_bundle_ids", {})
            if not rec_msgs or not rec_bids:
                exclusion_counts["empty_message"] += 1
                continue

            sid = ep["scenario_id"]
            regime = "loose" if "loose" in sid else "tight"
            sw = float(ep["signal_wt"])
            if sw not in SIGNAL_WTS:
                exclusion_counts["signal_wt_mismatch"] += 1
                continue
            # Load market once per episode to compute baseline_surplus.
            # Reconstruct under the SAME priors_mode as msgcap generation.
            try:
                market = _market_from_sid(sid, priors_mode=priors_mode)
            except Exception:
                continue
            traveler_by_id = {t.id: t for t in market.travelers}
            bundle_by_id = {b.id: b for b in market.bundles}

            for tid, bid in rec_bids.items():
                if bid is None:
                    exclusion_counts["bundle_none"] += 1
                    continue
                if tid not in traveler_by_id:
                    exclusion_counts["traveler_missing"] += 1
                    continue
                if bid not in bundle_by_id:
                    exclusion_counts["bundle_missing"] += 1
                    continue
                traveler = traveler_by_id[tid]
                bundle = bundle_by_id[bid]
                if bid not in traveler.utility:
                    exclusion_counts["not_in_utility"] += 1
                    continue
                msg = rec_msgs.get(tid, "")
                if not msg:
                    exclusion_counts["empty_message"] += 1
                    continue
                price = package_price(bundle, market)
                if price > traveler.budget + 1e-9:
                    exclusion_counts["price_over_budget"] += 1
                    continue
                baseline_surplus = float(traveler.utility[bid]) - price
                tau = DEFAULT_ACCEPTANCE_THRESHOLDS.get(traveler.archetype.id, 0.10)
                budget = float(traveler.budget)
                thresh = tau * budget
                lo = thresh - window_below_frac * budget
                # For primary window WINDOW_ABOVE_FRAC=0, we want strict baseline < τ
                # (i.e., half-open [τ − w, τ) ); implemented with strict <.
                hi = thresh + window_above_frac * budget
                if window_above_frac == 0.0:
                    in_window = (lo <= baseline_surplus) and (baseline_surplus < thresh)
                else:
                    in_window = (lo <= baseline_surplus <= hi)
                if not in_window:
                    exclusion_counts["outside_window"] += 1
                    continue
                exclusion_counts["kept"] += 1

                stratum = (sw, regime)
                pool_by_stratum.setdefault(stratum, []).append({
                    "scenario_id": sid,
                    "scenario_seed": int(ep["scenario_seed"]),
                    "signal_wt": sw,
                    "regime": regime,
                    "episode_seed": int(ep["episode_seed"]),
                    "traveler_id": tid,
                    "bundle_id": bid,
                    "original_message": msg,
                    "baseline_surplus": baseline_surplus,
                    "tau": tau,
                    "budget": budget,
                    "threshold": thresh,
                    "distance_to_threshold_frac": (baseline_surplus - thresh) / budget,
                })

    sampled: list[dict] = []
    stratum_counts: dict[str, dict] = {}
    for stratum, pool in sorted(pool_by_stratum.items()):
        chosen = rng.sample(pool, min(per_stratum, len(pool))) if pool else []
        sampled.extend(chosen)
        key = f"sw={stratum[0]}__regime={stratum[1]}"
        stratum_counts[key] = {
            "eligible": len(pool),
            "sampled": len(chosen),
            "undersampled": len(pool) < per_stratum,
        }
    print(f"[selection] stratum counts: {stratum_counts}", file=sys.stderr)
    manifest = {
        "window_below_frac": window_below_frac,
        "window_above_frac": window_above_frac,
        "per_stratum_target": per_stratum,
        "seed": seed,
        "exclusions": exclusion_counts,
        "stratum_counts": stratum_counts,
        "total_sampled": len(sampled),
    }
    return sampled, manifest


def enrich_pair(pair: dict, priors_mode: str = "uniform") -> dict:
    # Must reconstruct under the same priors_mode used at msgcap + extraction time.
    market = _market_from_sid(pair["scenario_id"], priors_mode=priors_mode)
    traveler = next(t for t in market.travelers if t.id == pair["traveler_id"])
    bundle = next(b for b in market.bundles if b.id == pair["bundle_id"])
    feasible = [
        b.id for b in market.bundles
        if b.id in traveler.utility
        and package_price(b, market) <= traveler.budget + 1e-9
    ]
    bundle_summary = {
        "hotel_id": bundle.hotel_id,
        "flight_id": bundle.flight_id,
        "extras": list(bundle.extras),
    }
    return {
        **pair,
        "market": market, "traveler": traveler, "bundle": bundle,
        "feasible": feasible, "bundle_summary": bundle_summary,
        "factual_message": factual_template(bundle, market),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar (binomial) p-value. b = original-only accepts, c = factual-only.
    Two-sided p from Binomial(b+c, 0.5). Returns 1.0 if b+c == 0."""
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided: 2 * min(P(X ≤ min(b,c)), 0.5)
    k = min(b, c)
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))


def paired_stats_binary(
    d: np.ndarray, clusters: np.ndarray = None,
    n_boot: int = 1000, rng_seed: int = 42,
) -> dict:
    """Paired risk difference with McNemar + cluster bootstrap 95% CI.

    d: array of {-1, 0, +1} where +1 = original-only accept, -1 = factual-only,
       0 = concordant. len(d) must equal n paired observations.
    clusters: array of cluster labels (e.g. scenario_id) of same length as d.
              If provided, bootstrap resamples clusters (not rows) for CI.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {
            "n": 0, "rd": float("nan"), "discord_original_only": 0,
            "discord_factual_only": 0, "concordant": 0,
            "mcnemar_p": float("nan"), "wilcoxon_p": float("nan"),
            "rbc": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan"),
        }
    discord_pos = int(np.sum(d > 0))   # original_only accepts
    discord_neg = int(np.sum(d < 0))   # factual_only accepts
    concordant = int(np.sum(d == 0))
    rd = float(d.mean())
    mp = mcnemar_exact(discord_pos, discord_neg)
    try:
        _, pw = stats.wilcoxon(d)
        pw = float(pw)
    except ValueError:
        pw = float("nan")
    nz = d[d != 0]
    if len(nz):
        ranks = stats.rankdata(np.abs(nz))
        wpos = float(np.sum(ranks[nz > 0]))
        wneg = float(np.sum(ranks[nz < 0]))
        rbc = (wpos - wneg) / (wpos + wneg) if (wpos + wneg) > 0 else 0.0
    else:
        rbc = 0.0
    brng = np.random.default_rng(rng_seed)
    if clusters is not None and len(np.unique(clusters)) >= 2:
        clusters = np.asarray(clusters)
        uniq = np.unique(clusters)
        boots = []
        for _ in range(n_boot):
            csamp = brng.choice(uniq, size=len(uniq), replace=True)
            vals = np.concatenate([d[clusters == c] for c in csamp])
            boots.append(vals.mean() if len(vals) else 0.0)
        boots = np.asarray(boots)
    else:
        boots = np.array([
            brng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)
        ])
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return {
        "n": int(n), "rd": rd,
        "discord_original_only": discord_pos,
        "discord_factual_only": discord_neg,
        "concordant": concordant,
        "mcnemar_p": mp, "wilcoxon_p": pw, "rbc": float(rbc),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def paired_stats_continuous(
    d: np.ndarray, clusters: np.ndarray = None,
    n_boot: int = 1000, rng_seed: int = 42,
) -> dict:
    """Paired continuous delta: mean, Wilcoxon, RBC, cluster bootstrap CI."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "wilcoxon_p": float("nan"),
                "rbc": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan")}
    try:
        _, pw = stats.wilcoxon(d)
        pw = float(pw)
    except ValueError:
        pw = float("nan")
    nz = d[d != 0]
    if len(nz):
        ranks = stats.rankdata(np.abs(nz))
        wpos = float(np.sum(ranks[nz > 0]))
        wneg = float(np.sum(ranks[nz < 0]))
        rbc = (wpos - wneg) / (wpos + wneg) if (wpos + wneg) > 0 else 0.0
    else:
        rbc = 0.0
    brng = np.random.default_rng(rng_seed)
    if clusters is not None and len(np.unique(clusters)) >= 2:
        clusters = np.asarray(clusters)
        uniq = np.unique(clusters)
        boots = []
        for _ in range(n_boot):
            csamp = brng.choice(uniq, size=len(uniq), replace=True)
            vals = np.concatenate([d[clusters == c] for c in csamp])
            boots.append(vals.mean() if len(vals) else 0.0)
        boots = np.asarray(boots)
    else:
        boots = np.array([
            brng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)
        ])
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return {"n": int(n), "mean": float(d.mean()),
            "wilcoxon_p": pw, "rbc": float(rbc),
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni stepdown-corrected p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (m - rank))
        adj[i] = min(1.0, running)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgcap-jsonl", type=str,
                    default="results/phase1_msgcap_qwen7b_msgcap_v2_episodes.jsonl",
                    help="Must be v2 (with recommendation_bundle_ids field).")
    ap.add_argument("--model-path", type=str, required=True,
                    help="Traveler LLM path. Qwen-14B-AWQ or Llama-3.1-8B-Instruct.")
    ap.add_argument("--arm-label", type=str, required=True,
                    help="e.g. 'qwen14b_awq' or 'llama31_8b'. Used for output file suffix.")
    ap.add_argument("--quantization", type=str, default=None,
                    help="e.g. 'awq_marlin' for AWQ model; None for bf16 Llama.")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    help="vLLM dtype. Use float16 for AWQ, bfloat16 for Llama.")
    ap.add_argument("--per-stratum", type=int, default=25,
                    help="Target samples per (signal_wt × regime) stratum.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (final=42; mini uses seed=41 for isolation).")
    ap.add_argument("--out", type=str, default=None,
                    help="Defaults to results/phase1c_<arm_label>_report.md.")
    ap.add_argument("--mini", action="store_true",
                    help="Mini mode: 5/stratum, seed=41 (DISJOINT from final). "
                         "Infra smoke-test only — must not be used to tune window, "
                         "prompts, coefficients, or model choice based on flip rate.")
    ap.add_argument("--diagnostic-window", action="store_true",
                    help="Use wider diagnostic window [τ−10%%, τ+5%%] as primary "
                         "sampling. Report as robustness check only.")
    ap.add_argument("--priors-mode", type=str, default="uniform",
                    choices=["uniform", "hbd_scale_normalized",
                             "hbd_archetype_only", "hbd_direct", "hbd_3x_hotel"],
                    help="Scenario prior used to reconstruct markets during "
                         "replay. MUST match the priors_mode used in msgcap "
                         "(otherwise budgets/utilities silently differ, "
                         "invalidating near-threshold selection).")
    args = ap.parse_args()

    if args.mini:
        args.per_stratum = 5
        args.seed = 41  # DISJOINT from final (seed=42) per pre-reg.
    if args.out is None:
        args.out = str(OUT_DIR / f"phase1c_{args.arm_label}_report.md")

    if args.diagnostic_window:
        wb, wa = DIAG_WINDOW_BELOW_FRAC, DIAG_WINDOW_ABOVE_FRAC
        window_label = "diagnostic"
    else:
        wb, wa = WINDOW_BELOW_FRAC, WINDOW_ABOVE_FRAC
        window_label = "primary"

    pairs, selection_manifest = extract_near_threshold_pairs(
        Path(args.msgcap_jsonl), per_stratum=args.per_stratum, seed=args.seed,
        window_below_frac=wb, window_above_frac=wa,
        priors_mode=args.priors_mode,
    )
    selection_manifest["window_label"] = window_label
    selection_manifest["mini"] = bool(args.mini)
    selection_manifest["arm_label"] = args.arm_label
    selection_manifest["model_path"] = args.model_path
    selection_manifest["priors_mode"] = args.priors_mode

    # FREEZE the selection manifest BEFORE any traveler call (pre-reg discipline).
    manifest_path = Path(args.out).with_suffix(".selection_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        sampled_tuples = [
            {"scenario_id": p["scenario_id"], "traveler_id": p["traveler_id"],
             "bundle_id": p["bundle_id"], "signal_wt": p["signal_wt"],
             "regime": p["regime"], "baseline_surplus": p["baseline_surplus"],
             "tau": p["tau"], "budget": p["budget"],
             "distance_to_threshold_frac": p["distance_to_threshold_frac"]}
            for p in pairs
        ]
        json.dump({**selection_manifest, "sampled_tuples": sampled_tuples},
                  f, ensure_ascii=False, indent=2)
    print(f"[manifest] frozen → {manifest_path}", file=sys.stderr)

    if not pairs:
        print(f"[error] no near-threshold pairs found in {args.msgcap_jsonl}. "
              f"Did you rerun msgcapture with recommendation_bundle_ids field?",
              file=sys.stderr)
        return 1
    print(f"[extract] {len(pairs)} near-threshold pairs "
          f"(window={window_label})", file=sys.stderr)
    if wa == 0.0:
        print(f"  window: [τ·budget − {wb*100:.0f}%·budget, τ·budget)  (flippable band)",
              file=sys.stderr)
    else:
        print(f"  window: [τ·budget − {wb*100:.0f}%·budget, "
              f"τ·budget + {wa*100:.0f}%·budget]", file=sys.stderr)

    pairs = [enrich_pair(p, priors_mode=args.priors_mode) for p in pairs]

    kwargs = dict(
        model_path=args.model_path,
        dtype=args.dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        default_max_tokens=400,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    backend = VLLMBackend(**kwargs)

    all_prompts = []
    all_meta = []
    for p in pairs:
        u_orig = build_traveler_user_prompt(
            archetype_id=p["traveler"].archetype.id,
            budget=p["traveler"].budget,
            vibe_tags=p["traveler"].archetype.vibe_tags,
            bundle_id=p["bundle_id"],
            bundle_summary=p["bundle_summary"],
            ota_message=p["original_message"],
            feasible_bundles=p["feasible"],
        )
        u_fact = build_traveler_user_prompt(
            archetype_id=p["traveler"].archetype.id,
            budget=p["traveler"].budget,
            vibe_tags=p["traveler"].archetype.vibe_tags,
            bundle_id=p["bundle_id"],
            bundle_summary=p["bundle_summary"],
            ota_message=p["factual_message"],
            feasible_bundles=p["feasible"],
        )
        all_prompts.append((TRAVELER_SYSTEM_PROMPT, u_orig))
        all_meta.append({"pair": p, "variant": "original"})
        all_prompts.append((TRAVELER_SYSTEM_PROMPT, u_fact))
        all_meta.append({"pair": p, "variant": "factual"})

    schemas = [TRAVELER_OUTPUT_SCHEMA] * len(all_prompts)

    t0 = time.time()
    raws = backend.generate_batch(all_prompts, max_tokens=400, json_schemas=schemas)
    elapsed = time.time() - t0
    print(f"[traveler-{args.arm_label}] {len(raws)} calls in {elapsed:.0f}s "
          f"({elapsed/max(len(raws),1):.2f}s/call)", file=sys.stderr)

    results = []
    parse_fail = 0
    for meta, raw in zip(all_meta, raws):
        pair = meta["pair"]
        try:
            features = parse_traveler_output(raw)
            accepted = compute_acceptance(
                features, pair["baseline_surplus"], pair["traveler"].budget, pair["tau"],
            )
        except (json.JSONDecodeError, ValueError):
            features = None
            accepted = False
            parse_fail += 1
        results.append({
            "arm": args.arm_label,
            "variant": meta["variant"],
            "scenario_id": pair["scenario_id"],
            "traveler_id": pair["traveler_id"],
            "bundle_id": pair["bundle_id"],
            "signal_wt": pair["signal_wt"],
            "episode_seed": int(pair.get("episode_seed", -1)),
            "regime": pair["regime"],
            "baseline_surplus": pair["baseline_surplus"],
            "tau": pair["tau"],
            "budget": pair["budget"],
            "distance_to_threshold_frac": pair["distance_to_threshold_frac"],
            "accepted": bool(accepted),
            "features": features,
            "raw_excerpt": None if features is not None else raw[:300],
        })

    # Pair up by full episode identity. The original v2 key
    # (scenario_id, traveler_id, bundle_id) silently collapsed distinct episodes
    # when the same (traveler, bundle) pair appeared across multiple signal_wt
    # or episode_seed values, which happens in the stratified sampling — later
    # records overwrote earlier ones, corrupting discordance counts. Battle-fix
    # (GPT-5.4, 2026-04-20): include signal_wt + episode_seed.
    by_key: dict[tuple, dict] = {}
    for r in results:
        key = (r["scenario_id"], float(r["signal_wt"]),
               int(r["episode_seed"]), r["traveler_id"], r["bundle_id"])
        by_key.setdefault(key, {})[r["variant"]] = r
    paired = [v for v in by_key.values() if "original" in v and "factual" in v
              and v["original"]["features"] is not None
              and v["factual"]["features"] is not None]

    lines: list[str] = []
    lines.append(f"# Phase 1c — Near-threshold Cross-family Replay ({args.arm_label})\n")
    lines.append(f"Traveler model: {args.model_path}")
    lines.append(f"Window ({window_label}): "
                 + (f"[τ·budget − {wb*100:.0f}%·budget, τ·budget)"
                    if wa == 0.0 else
                    f"[τ·budget − {wb*100:.0f}%·budget, τ·budget + {wa*100:.0f}%·budget]"))
    parse_rate = 1.0 - parse_fail / max(len(results), 1)
    lines.append(f"Parse success: {parse_rate:.1%} ({len(results) - parse_fail}/{len(results)})")
    lines.append(f"n pairs extracted: {len(pairs)}; n traveler calls: {len(results)}; "
                 f"parse fail: {parse_fail}; n paired (both variants valid): {len(paired)}")

    # ---- Pre-registration block (verbatim) ----
    lines.append("\n## PRE-REGISTRATION (verbatim)\n")
    lines.append(
        "> Primary endpoint: the balanced paired risk difference in acceptance between "
        "original commission-LLM messages and minimum-disclosure factual templates, over "
        "sampled OTA recommendations with baseline surplus in `[τ·budget − 0.05·budget, "
        "τ·budget)`, tested by exact McNemar on paired discordance, separately for the "
        "Qwen-14B-AWQ and Llama-3.1-8B arms. GO requires both arms Δ ≥ 10pp with "
        "95% CI excluding 0, subject to validity gates: parse success ≥ 95%, "
        "factual-variant acceptance not at ceiling, primary window mechanically "
        "flippable. Secondary feature deltas are EXPLORATORY, Holm-corrected across "
        "the 4 features per arm."
    )
    lines.append("\nDecision rules:")
    lines.append("- GO: both arms Δ ≥ 10pp, McNemar p < 0.01, 95% CI excludes 0.")
    lines.append("- REVISE: Qwen arm Δ ≥ 10pp AND Llama arm Δ < 5pp → workshop-only, "
                 "claim restricted to same-family.")
    lines.append("- KILL: both arms Δ < 5pp in primary window.")

    # ---- Validity gates ----
    lines.append("\n## VALIDITY GATES\n")
    validity_parse_ok = parse_rate >= 0.95
    lines.append(f"- Parse success ≥ 95%: {'✅' if validity_parse_ok else '❌'}  "
                 f"(actual {parse_rate:.1%})")

    # ---- Primary: paired risk difference + McNemar ----
    lines.append("\n## PRIMARY — Paired acceptance delta (McNemar)\n")

    # Overall (balanced across strata — here simple mean since stratum sizes are
    # approx balanced by design; report both balanced and raw if asymmetry exists).
    d_all = np.array([int(p["original"]["accepted"]) - int(p["factual"]["accepted"])
                      for p in paired])
    cluster_ids = np.array([p["original"]["scenario_id"] for p in paired])
    s = paired_stats_binary(d_all, clusters=cluster_ids)
    orig_rate = float(np.mean([int(p["original"]["accepted"]) for p in paired]))
    fact_rate = float(np.mean([int(p["factual"]["accepted"]) for p in paired]))
    validity_factual_not_ceiling = fact_rate < 0.98
    lines.append(f"- Parse OK: {'✅' if validity_parse_ok else '❌'}")
    lines.append(f"- Factual acceptance not at ceiling (<0.98): "
                 f"{'✅' if validity_factual_not_ceiling else '❌'}  "
                 f"(actual {fact_rate:.2%})")
    lines.append("")
    lines.append(f"**Overall (cluster-bootstrapped by scenario_id)**")
    lines.append(f"- n paired: {s['n']}  "
                 f"(discord_original_only={s['discord_original_only']}, "
                 f"discord_factual_only={s['discord_factual_only']}, "
                 f"concordant={s['concordant']})")
    lines.append(f"- orig_accept: {orig_rate:.2%}  |  fact_accept: {fact_rate:.2%}")
    lines.append(f"- RD (risk difference): {s['rd']:+.2%}  "
                 f"[95% CI: {s['ci_lo']:+.2%}, {s['ci_hi']:+.2%}]")
    lines.append(f"- McNemar exact p: {s['mcnemar_p']:.4f}  "
                 f"(Wilcoxon secondary: {s['wilcoxon_p']:.4f})")
    lines.append(f"- paired RBC: {s['rbc']:+.3f}")

    # Gate
    delta = s["rd"]
    ci_excludes_zero = (s["ci_lo"] > 0) or (s["ci_hi"] < 0)
    mcnemar_sig = s["mcnemar_p"] < 0.01
    all_validity_ok = validity_parse_ok and validity_factual_not_ceiling
    if delta >= 0.10 and ci_excludes_zero and mcnemar_sig and all_validity_ok:
        verdict = "GO (≥10pp, CI excludes 0, McNemar p<0.01, validity OK)"
    elif delta >= 0.10 and not all_validity_ok:
        verdict = "BLOCKED by validity gate — investigate parse/ceiling before claiming GO"
    elif 0.05 <= delta < 0.10:
        verdict = "REVISE (5-10pp; need Llama arm to match before workshop claim)"
    else:
        verdict = "KILL-candidate (<5pp in primary flippable window)"
    lines.append(f"- **Arm verdict: {verdict}**")

    # ---- Heterogeneity (per signal_wt, reported but not gated) ----
    lines.append("\n### Heterogeneity by signal_wt (descriptive, NOT gated)\n")
    lines.append("| signal_wt | n | orig_accept | fact_accept | RD | 95% CI | McNemar p | discord+/− |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sw in SIGNAL_WTS:
        sub = [p for p in paired if abs(p["original"]["signal_wt"] - sw) < 1e-6]
        if not sub:
            continue
        orig_acc = np.array([int(p["original"]["accepted"]) for p in sub])
        fact_acc = np.array([int(p["factual"]["accepted"]) for p in sub])
        d = orig_acc - fact_acc
        sub_clust = np.array([p["original"]["scenario_id"] for p in sub])
        s_sub = paired_stats_binary(d, clusters=sub_clust)
        lines.append(
            f"| {sw} | {s_sub['n']} | {orig_acc.mean():.2%} | {fact_acc.mean():.2%} | "
            f"{s_sub['rd']:+.2%} | [{s_sub['ci_lo']:+.2%}, {s_sub['ci_hi']:+.2%}] | "
            f"{s_sub['mcnemar_p']:.4f} | "
            f"{s_sub['discord_original_only']}/{s_sub['discord_factual_only']} |"
        )

    # ---- Secondary: feature deltas (exploratory, Holm-corrected) ----
    lines.append("\n## SECONDARY (EXPLORATORY, Holm-corrected) — Feature deltas\n")
    lines.append("Pre-reg: feature-level deltas are EXPLORATORY mechanism, not a "
                 "replacement for the primary acceptance endpoint.\n")
    feat_names = ("perceived_fit_delta", "perceived_risk", "trust_score", "urgency_felt")
    feat_stats = []
    for feat in feat_names:
        orig_vals = np.array([float(p["original"]["features"].get(feat, 0.0)) for p in paired])
        fact_vals = np.array([float(p["factual"]["features"].get(feat, 0.0)) for p in paired])
        d = orig_vals - fact_vals
        fs = paired_stats_continuous(d, clusters=cluster_ids)
        feat_stats.append((feat, orig_vals, fact_vals, fs))
    pvals = [fs[3]["wilcoxon_p"] if not np.isnan(fs[3]["wilcoxon_p"]) else 1.0
             for fs in feat_stats]
    holm_q = holm_correct(pvals)
    lines.append("| feature | n | mean_orig | mean_fact | Δ | 95% CI | Wilcoxon p | Holm q | RBC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (feat, ov, fv, fs), q in zip(feat_stats, holm_q):
        lines.append(
            f"| {feat} | {fs['n']} | {ov.mean():+.3f} | {fv.mean():+.3f} | "
            f"{fs['mean']:+.3f} | [{fs['ci_lo']:+.3f}, {fs['ci_hi']:+.3f}] | "
            f"{fs['wilcoxon_p']:.4f} | {q:.4f} | {fs['rbc']:+.3f} |"
        )

    # ---- Calibration diagnostics per arm ----
    lines.append("\n## CALIBRATION DIAGNOSTICS (per arm)\n")
    all_feats = {feat: [] for feat in feat_names}
    saturation = {feat: 0 for feat in feat_names}
    cap_hits = 0
    tot_calls = 0
    for r in results:
        if r["features"] is None:
            continue
        tot_calls += 1
        for feat in feat_names:
            v = float(r["features"].get(feat, 0.0))
            all_feats[feat].append(v)
            if abs(v) >= 0.99:
                saturation[feat] += 1
        # Re-derive msg_adj to detect cap hits (mirrors llm_traveler.compute_acceptance).
        from tourmart.llm_traveler import (
            ADJ_SURPLUS_FIT, ADJ_SURPLUS_TRUST, ADJ_SURPLUS_RISK,
            ADJ_SURPLUS_URGENCY, MSG_ADJ_CAP,
        )
        f = r["features"]
        budget = float(r["budget"])
        raw_msg_adj = budget * (
            ADJ_SURPLUS_FIT * float(f.get("perceived_fit_delta", 0.0))
            + ADJ_SURPLUS_TRUST * float(f.get("trust_score", 0.0))
            - ADJ_SURPLUS_RISK * float(f.get("perceived_risk", 0.0))
            + ADJ_SURPLUS_URGENCY * float(f.get("urgency_felt", 0.0))
        )
        cap = MSG_ADJ_CAP * budget
        if abs(raw_msg_adj) > cap:
            cap_hits += 1
    lines.append(f"- Parse success: {parse_rate:.1%} "
                 f"({'✅' if validity_parse_ok else '❌ FAILS GATE'})")
    lines.append(f"- MSG_ADJ cap-hit rate: {cap_hits / max(tot_calls, 1):.1%}  "
                 f"({cap_hits}/{tot_calls})")
    lines.append("- Feature distributions (all calls, both variants combined):")
    lines.append("  | feature | mean | std | min | max | saturation(|v|≥0.99) |")
    lines.append("  |---|---:|---:|---:|---:|---:|")
    for feat in feat_names:
        vals = np.array(all_feats[feat])
        if len(vals) == 0:
            continue
        sat_rate = saturation[feat] / max(len(vals), 1)
        lines.append(
            f"  | {feat} | {vals.mean():+.3f} | {vals.std():.3f} | "
            f"{vals.min():+.3f} | {vals.max():+.3f} | {sat_rate:.1%} |"
        )
    if any(saturation[f] / max(len(all_feats[f]), 1) > 0.5 for f in feat_names):
        lines.append("- ⚠️ SATURATION WARNING: >50% of calls hit ±0.99 on some feature "
                     "→ perception channel may be degenerate for this arm. "
                     "Interpret cross-arm comparison with caution.")

    # ---- Diagnostics: distance-to-threshold ----
    lines.append("\n## DIAGNOSTIC — Distance to threshold (fraction of budget)\n")
    dist = np.array([p["original"]["distance_to_threshold_frac"] for p in paired])
    if len(dist):
        lines.append(f"- n: {len(dist)}")
        lines.append(f"- mean: {dist.mean():+.3f}  (expected: slightly negative under primary window)")
        lines.append(f"- stdev: {dist.std():.3f}")
        lines.append(f"- quartiles: Q1={np.percentile(dist, 25):+.3f}, "
                     f"median={np.median(dist):+.3f}, Q3={np.percentile(dist, 75):+.3f}")

    # ---- Selection manifest reference ----
    lines.append(f"\n## SELECTION MANIFEST\n- Frozen at: `{manifest_path.name}`")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"[done] report → {args.out}", file=sys.stderr)

    raw_path = Path(args.out).with_suffix(".raw.jsonl")
    with raw_path.open("w") as f:
        for r in results:
            rec = dict(r)
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    print(f"[done] raw → {raw_path}", file=sys.stderr)

    # Short stdout headline for monitoring.
    print("\n=== HEADLINE ===")
    if paired:
        print(f"arm: {args.arm_label}  (window={window_label})")
        print(f"n paired: {s['n']}  "
              f"(discord +={s['discord_original_only']}, −={s['discord_factual_only']}, "
              f"concordant={s['concordant']})")
        print(f"risk difference: {s['rd']:+.2%}  "
              f"[95% CI: {s['ci_lo']:+.2%}, {s['ci_hi']:+.2%}]")
        print(f"McNemar p: {s['mcnemar_p']:.4f}  "
              f"validity: parse={'✅' if validity_parse_ok else '❌'} "
              f"fact_ceiling={'✅' if validity_factual_not_ceiling else '❌'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

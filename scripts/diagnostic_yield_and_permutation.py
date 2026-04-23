"""Two cheap offline diagnostics to decide scale-up path.

1. **Eligibility-yield curve**: bin v3 msgcap commission pairs by distance-to-threshold
   (fraction of budget). Shows whether scaling msgcap is worth it or we hit a
   structural ceiling.

2. **Grid-level max-stat permutation test**: for the Phase 1c v2 raw data, permute
   original/factual labels within each pair 1000×, recompute all 36 (mult, cap)
   cells, record max RD. Gives family-wise-error-corrected p-value.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# Import acceptance rule from our ablation script for consistency
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_cap_ablation import compute_acceptance_with_cap, load_raw, pair_up


MSGCAP_V3 = "results/phase1_msgcap_qwen7b_msgcap_v3_episodes.jsonl"
QWEN_RAW = "results/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl"
LLAMA_RAW = "results/phase1c_llama31_8b_diag_v4_report.raw.jsonl"

# Match the Phase 1c selection constants.
WINDOW_BELOW_FRAC = 0.10  # diagnostic
WINDOW_ABOVE_FRAC = 0.05
TAU_DEFAULTS = {
    "solo_business":  0.05,
    "solo_leisure":   0.10,
    "couple_romance": 0.09,
    "couple_cultural": 0.11,
    "family3_kids":   0.13,
    "family4_kids":   0.15,
}


def yield_curve():
    """Count commission recommendation pairs by distance-to-threshold bin."""
    # We need: for each commission episode, each tid→bid recommendation, compute
    # baseline_surplus and distance-to-threshold. But we can't compute
    # baseline_surplus from the jsonl alone (needs market). Shortcut: pull the
    # Phase 1c v2 selection_manifest which already has this for pairs IN the
    # diagnostic window; plus we need the total commission pairs to get the
    # unfiltered denominator. We'll infer yield by bin using a relative count.
    # Alternative: compute from scratch using the msgcap v3 jsonl + generate_small_market.
    #
    # To avoid circular imports of tourmart package, just use the selection manifest
    # from Phase 1c v2, which has counts of 'outside_window' exclusions by default.
    manifest_path = "results/phase1c_qwen14b_awq_diag_v2_report.selection_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    exclusions = manifest["exclusions"]
    print("=== msgcap v3 → Phase 1c pair pool breakdown ===")
    for k, v in exclusions.items():
        print(f"  {k:25s}: {v}")

    stratum_counts = manifest["stratum_counts"]
    print("\n=== Eligible (near-threshold) pairs by stratum ===")
    for s, d in stratum_counts.items():
        print(f"  {s:30s}: eligible={d['eligible']:3d}  "
              f"sampled={d['sampled']:2d}  undersampled={d['undersampled']}")
    total_eligible = sum(d["eligible"] for d in stratum_counts.values())
    print(f"\nTOTAL eligible: {total_eligible}")

    # To find where the OTA recommendations cluster along distance-to-threshold,
    # use the existing v2 raw (which has distance_to_threshold_frac for pairs in
    # the diagnostic window only). Combined with the exclusion count ratio, we
    # estimate the overall distribution.
    in_window_pairs = []
    for rec in load_raw(Path(QWEN_RAW)):
        if rec["variant"] == "original":  # one per unique pair
            in_window_pairs.append(rec["distance_to_threshold_frac"])
    in_window_pairs = np.array(in_window_pairs)
    print(f"\n=== distance-to-threshold distribution (in-window sample n={len(in_window_pairs)}) ===")
    bins = [(-0.10, -0.075), (-0.075, -0.05), (-0.05, -0.025),
            (-0.025, 0.0), (0.0, 0.025), (0.025, 0.05)]
    for lo, hi in bins:
        count = int(np.sum((in_window_pairs >= lo) & (in_window_pairs < hi)))
        print(f"  [{lo:+.3f}, {hi:+.3f}): {count}")

    # Scaling estimate: if we assume near-threshold density scales linearly with
    # commission valid eps:
    # v3: 450 valid_commission → 145 extracted pairs (before dedup) → 48 unique
    # v4 (2x scale, per-cond 3000): ~900 valid_commission → ~290 extracted → ~95 unique (if no ceiling)
    # v4 (3x scale, per-cond 4500): ~1350 valid_commission → ~435 extracted → ~145 unique
    print("\n=== Scale-up projection (linear assumption) ===")
    for factor, label in [(1, "v3 (current)"), (2, "v4 2×"), (3, "v4 3×")]:
        proj_extracted = 145 * factor
        proj_unique = 48 * factor
        print(f"  {label}: extracted≈{proj_extracted}, unique≈{proj_unique}")


def max_stat_permutation(n_perm: int = 1000, seed: int = 42):
    """Grid-level max-statistic permutation test.

    For each permutation, randomly swap (orig, fact) labels within each pair,
    recompute RD across all 36 cells, record max RD. Empirical p = fraction of
    permutations where max_perm_RD ≥ observed_max_RD.
    """
    rng = np.random.default_rng(seed)
    cap_values = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
    mult_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    for arm_label, raw_path in [("Qwen-14B", QWEN_RAW), ("Llama-8B", LLAMA_RAW)]:
        print(f"\n=== {arm_label} — max-stat permutation ({n_perm} perms, {len(cap_values)*len(mult_values)} cells) ===")
        rows = load_raw(Path(raw_path))
        paired = pair_up(rows)
        n = len(paired)
        if n == 0:
            print("  no paired data")
            continue

        # Observed: for each cell, compute accept vectors and RD.
        def compute_rd_grid(paired_data):
            rds = {}
            for mult in mult_values:
                for cap in cap_values:
                    orig = []
                    fact = []
                    for p in paired_data:
                        ao, _, _ = compute_acceptance_with_cap(
                            p["features_original"], p["baseline_surplus"],
                            p["budget"], p["tau"], cap, coef_multiplier=mult,
                        )
                        af, _, _ = compute_acceptance_with_cap(
                            p["features_factual"], p["baseline_surplus"],
                            p["budget"], p["tau"], cap, coef_multiplier=mult,
                        )
                        orig.append(int(ao))
                        fact.append(int(af))
                    rd = np.mean(np.array(orig) - np.array(fact))
                    rds[(mult, cap)] = rd
            return rds

        observed_rds = compute_rd_grid(paired)
        observed_max = max(observed_rds.values())
        observed_max_cell = max(observed_rds.items(), key=lambda kv: kv[1])

        # Effective test count: unique (orig_vec, fact_vec) configurations.
        unique_configs = set()
        for mult in mult_values:
            for cap in cap_values:
                orig = []
                fact = []
                for p in paired:
                    ao, _, _ = compute_acceptance_with_cap(
                        p["features_original"], p["baseline_surplus"],
                        p["budget"], p["tau"], cap, coef_multiplier=mult,
                    )
                    af, _, _ = compute_acceptance_with_cap(
                        p["features_factual"], p["baseline_surplus"],
                        p["budget"], p["tau"], cap, coef_multiplier=mult,
                    )
                    orig.append(int(ao))
                    fact.append(int(af))
                unique_configs.add((tuple(orig), tuple(fact)))
        print(f"  unique (orig_vec, fact_vec) configs across grid: {len(unique_configs)}")

        # Cluster-by-scenario: collect scenario_id per pair
        scenario_ids = [p["scenario_id"] for p in paired]
        unique_scenarios = list(set(scenario_ids))

        # Pair-level permutation (sensitivity).
        max_rds_pair = []
        for _ in range(n_perm):
            perm_paired = []
            for p in paired:
                if rng.random() < 0.5:
                    perm_paired.append({**p, "features_original": p["features_factual"],
                                         "features_factual": p["features_original"]})
                else:
                    perm_paired.append(p)
            perm_rds = compute_rd_grid(perm_paired)
            max_rds_pair.append(max(perm_rds.values()))
        max_rds_pair = np.array(max_rds_pair)
        p_pair = float(np.mean(max_rds_pair >= observed_max))

        # Cluster-by-scenario permutation (PRIMARY, per GPT): flip label for
        # ALL pairs sharing a scenario jointly.
        max_rds_cluster = []
        for _ in range(n_perm):
            flip_by_scenario = {s: rng.random() < 0.5 for s in unique_scenarios}
            perm_paired = []
            for p in paired:
                if flip_by_scenario[p["scenario_id"]]:
                    perm_paired.append({**p, "features_original": p["features_factual"],
                                         "features_factual": p["features_original"]})
                else:
                    perm_paired.append(p)
            perm_rds = compute_rd_grid(perm_paired)
            max_rds_cluster.append(max(perm_rds.values()))
        max_rds_cluster = np.array(max_rds_cluster)
        p_cluster = float(np.mean(max_rds_cluster >= observed_max))

        print(f"  observed max RD: {observed_max*100:+.2f}pp at "
              f"(mult=×{observed_max_cell[0][0]:.1f}, cap={observed_max_cell[0][1]*100:.1f}%)")
        print(f"  n unique scenarios: {len(unique_scenarios)}  (cluster permutation unit)")
        print(f"  pair-level max-stat p (sensitivity): {p_pair:.4f} "
              f"{'✓' if p_pair < 0.05 else '✗'}")
        print(f"  CLUSTER-level max-stat p (PRIMARY): {p_cluster:.4f} "
              f"{'✓ family-wise significant' if p_cluster < 0.05 else '✗ NOT significant family-wise'}")


if __name__ == "__main__":
    print("#" * 60)
    print("# DIAGNOSTIC 1: Eligibility-yield curve")
    print("#" * 60)
    yield_curve()
    print()
    print("#" * 60)
    print("# DIAGNOSTIC 2: Grid-level max-stat permutation test")
    print("#" * 60)
    max_stat_permutation(n_perm=1000, seed=42)

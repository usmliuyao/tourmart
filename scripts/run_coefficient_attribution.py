"""Coefficient attribution — which perceived channel drives the live region?

Offline sensitivity analysis: zero out each of the 4 coefficients (fit, trust,
risk, urgency) one at a time, recompute the 2D grid, report the max observed RD
per arm. If removing one channel crushes the live region, that channel is load-
bearing for the steering effect.

Pure Python, no GPU. ~10 seconds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

QWEN_RAW = "results/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl"
LLAMA_RAW = "results/phase1c_llama31_8b_diag_v4_report.raw.jsonl"

# Baseline Round 20 coefficients
COEFS_FULL = {"fit": 0.03, "trust": 0.015, "risk": 0.025, "urgency": 0.01}
BASELINE_SURPLUS_FLOOR = -0.10


def compute_acceptance_custom_coefs(
    features, baseline_surplus, budget, tau,
    msg_adj_cap, coefs, coef_multiplier=1.0,
):
    if baseline_surplus < BASELINE_SURPLUS_FLOOR * budget:
        return False
    fit = float(features.get("perceived_fit_delta", 0.0))
    risk = float(features.get("perceived_risk", 0.0))
    trust = float(features.get("trust_score", 0.0))
    urgency = float(features.get("urgency_felt", 0.0))
    raw_msg_adj = coef_multiplier * budget * (
        coefs["fit"] * fit + coefs["trust"] * trust
        - coefs["risk"] * risk + coefs["urgency"] * urgency
    )
    cap = msg_adj_cap * budget
    msg_adj = max(-cap, min(cap, raw_msg_adj))
    adj = baseline_surplus + msg_adj
    return adj >= tau * budget


def load_and_pair(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("features"): rows.append(r)
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["scenario_id"], r["traveler_id"], r["bundle_id"])
        by_key[key][r["variant"]] = r
    paired = []
    for v in by_key.values():
        if "original" in v and "factual" in v:
            paired.append({
                "scenario_id": v["original"]["scenario_id"],
                "baseline_surplus": v["original"]["baseline_surplus"],
                "budget": v["original"]["budget"],
                "tau": v["original"]["tau"],
                "features_original": v["original"]["features"],
                "features_factual": v["factual"]["features"],
            })
    return paired


def max_rd_grid(paired, coefs, cap_values, mult_values):
    max_rd = -1.0
    max_cell = None
    for mult in mult_values:
        for cap in cap_values:
            orig = []
            fact = []
            for p in paired:
                ao = compute_acceptance_custom_coefs(
                    p["features_original"], p["baseline_surplus"],
                    p["budget"], p["tau"], cap, coefs, mult,
                )
                af = compute_acceptance_custom_coefs(
                    p["features_factual"], p["baseline_surplus"],
                    p["budget"], p["tau"], cap, coefs, mult,
                )
                orig.append(int(ao))
                fact.append(int(af))
            # Skip cells where factual hits ceiling
            fact_rate = np.mean(fact) if fact else 0.0
            if fact_rate >= 0.98:
                continue
            rd = np.mean(np.array(orig) - np.array(fact))
            if rd > max_rd:
                max_rd = rd
                max_cell = (mult, cap, int(np.sum(np.array(orig) > np.array(fact))),
                            int(np.sum(np.array(orig) < np.array(fact))))
    return max_rd, max_cell


def main():
    cap_values = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
    mult_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    qwen_paired = load_and_pair(QWEN_RAW)
    llama_paired = load_and_pair(LLAMA_RAW)
    print(f"Loaded Qwen n={len(qwen_paired)}, Llama n={len(llama_paired)}")

    print("\n=== Coefficient attribution: zero out one channel at a time ===")
    print(f"\nBaseline coefficients: {COEFS_FULL}")
    print()

    lines = ["# Coefficient attribution — v4 n=143\n",
             "For each condition, we zero out one of the four perceived-feature "
             "coefficients and recompute max RD over the 2D grid. If the max RD drops "
             "substantially, that channel is load-bearing for the live transmission "
             "region.\n",
             "| Condition | Qwen max RD | Qwen peak cell | Llama max RD | Llama peak cell |",
             "|---|---:|---:|---:|---:|"]

    conditions = [
        ("Full (baseline)", COEFS_FULL),
        ("No fit (fit=0)", {**COEFS_FULL, "fit": 0.0}),
        ("No trust (trust=0)", {**COEFS_FULL, "trust": 0.0}),
        ("No risk (risk=0)", {**COEFS_FULL, "risk": 0.0}),
        ("No urgency (urgency=0)", {**COEFS_FULL, "urgency": 0.0}),
        ("Fit ONLY (others=0)", {"fit": 0.03, "trust": 0, "risk": 0, "urgency": 0}),
        ("Trust ONLY", {"fit": 0, "trust": 0.015, "risk": 0, "urgency": 0}),
    ]
    for label, coefs in conditions:
        q_rd, q_cell = max_rd_grid(qwen_paired, coefs, cap_values, mult_values)
        l_rd, l_cell = max_rd_grid(llama_paired, coefs, cap_values, mult_values)
        q_cell_str = f"×{q_cell[0]:.1f}, {q_cell[1]*100:.1f}% ({q_cell[2]}/{q_cell[3]})" if q_cell else "—"
        l_cell_str = f"×{l_cell[0]:.1f}, {l_cell[1]*100:.1f}% ({l_cell[2]}/{l_cell[3]})" if l_cell else "—"
        print(f"  {label:25s} | Qwen max RD: {q_rd*100:+6.2f}pp @ {q_cell_str:25s} | "
              f"Llama: {l_rd*100:+6.2f}pp @ {l_cell_str}")
        lines.append(f"| {label} | {q_rd*100:+.2f}pp | {q_cell_str} | "
                     f"{l_rd*100:+.2f}pp | {l_cell_str} |")

    Path("results/phase1c_coef_attribution_v4.md").write_text("\n".join(lines))
    print("\n[done] wrote results/phase1c_coef_attribution_v4.md")


if __name__ == "__main__":
    main()

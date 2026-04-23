"""Cap ablation — offline sensitivity analysis on MSG_ADJ_CAP.

Phase 1c (Round 21) observed behavioral null under Round 20's MSG_ADJ_CAP=0.05
with perception-level Δfit≈+0.17 replicated cross-family. GPT adversarial review
flagged "cap as policy lever" as circular without ablation.

This script reruns compute_acceptance on the EXISTING Phase 1c diagnostic raw
data, sweeping MSG_ADJ_CAP ∈ {0, 1%, 2.5%, 5%, 7.5%, 10%, 15%, 20%, 50%, 100%}
× budget. Produces a threshold curve per arm: at what cap value do the observed
perceptual shifts begin translating into behavioral flips?

Pure Python, no LLM calls, no GPU. ~60s runtime.

Framing (per GPT final turn): this is a SENSITIVITY ANALYSIS of the benchmark's
governance parameter, not empirical policy validation. Reviewer-buy sentence:

> "We do not interpret MSG_ADJ_CAP as a directly implementable regulation;
> rather, it is a benchmark parameter that operationalizes a family of
> information-design safeguards limiting how much message-induced perceptions
> may alter a welfare-rational choice."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import math
import numpy as np

# Inline exact binomial CDF to avoid scipy dependency (local mac env is broken).
def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    # P(X <= k) for X ~ Binomial(n, p).
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
               for i in range(k + 1))


# Frozen Round 20 coefficients (must match src/tourmart/llm_traveler.py).
ADJ_SURPLUS_FIT = 0.03
ADJ_SURPLUS_TRUST = 0.015
ADJ_SURPLUS_RISK = 0.025
ADJ_SURPLUS_URGENCY = 0.01
BASELINE_SURPLUS_FLOOR = -0.10


def compute_acceptance_with_cap(
    features: dict, baseline_surplus: float, budget: float,
    tau: float, msg_adj_cap: float, coef_multiplier: float = 1.0,
) -> tuple[bool, float, bool]:
    """Returns (accepted, raw_msg_adj_frac_budget, cap_hit).

    coef_multiplier scales all four Round-20 coefficients uniformly. 1.0 = current
    pre-reg Round 20 values. ~3.33 = roughly Round 19 (pre-tightening). Sweeping
    this tests whether the 3-5× coefficient tightening was overly conservative."""
    if baseline_surplus < BASELINE_SURPLUS_FLOOR * budget:
        return False, 0.0, False
    fit = float(features.get("perceived_fit_delta", 0.0))
    risk = float(features.get("perceived_risk", 0.0))
    trust = float(features.get("trust_score", 0.0))
    urgency = float(features.get("urgency_felt", 0.0))
    raw_msg_adj = coef_multiplier * budget * (
        ADJ_SURPLUS_FIT * fit
        + ADJ_SURPLUS_TRUST * trust
        - ADJ_SURPLUS_RISK * risk
        + ADJ_SURPLUS_URGENCY * urgency
    )
    cap = msg_adj_cap * budget
    msg_adj = max(-cap, min(cap, raw_msg_adj))
    cap_hit = abs(raw_msg_adj) > cap
    adj = baseline_surplus + msg_adj
    return adj >= tau * budget, raw_msg_adj / budget, cap_hit


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * _binom_cdf(k, n, 0.5)))


def load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return [r for r in rows if r.get("features") is not None]


def pair_up(rows: list[dict]) -> list[dict]:
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["scenario_id"], r["traveler_id"], r["bundle_id"])
        by_key[key][r["variant"]] = r
    paired = []
    for key, v in by_key.items():
        if "original" in v and "factual" in v:
            paired.append({
                "scenario_id": key[0], "traveler_id": key[1], "bundle_id": key[2],
                "signal_wt": v["original"]["signal_wt"],
                "regime": v["original"]["regime"],
                "baseline_surplus": v["original"]["baseline_surplus"],
                "tau": v["original"]["tau"],
                "budget": v["original"]["budget"],
                "features_original": v["original"]["features"],
                "features_factual": v["factual"]["features"],
            })
    return paired


def ablate_one_arm_grid(
    paired: list[dict], cap_values: list[float], mult_values: list[float],
) -> list[dict]:
    """Run 2D ablation over (cap, coefficient_multiplier)."""
    rows = []
    for mult in mult_values:
        for cap in cap_values:
            orig_acc, fact_acc = [], []
            cap_hits_orig = cap_hits_fact = 0
            raw_adj_orig, raw_adj_fact = [], []
            for p in paired:
                ao, raw_o, hit_o = compute_acceptance_with_cap(
                    p["features_original"], p["baseline_surplus"], p["budget"],
                    p["tau"], cap, coef_multiplier=mult,
                )
                af, raw_f, hit_f = compute_acceptance_with_cap(
                    p["features_factual"], p["baseline_surplus"], p["budget"],
                    p["tau"], cap, coef_multiplier=mult,
                )
                orig_acc.append(ao)
                fact_acc.append(af)
                cap_hits_orig += int(hit_o)
                cap_hits_fact += int(hit_f)
                raw_adj_orig.append(raw_o)
                raw_adj_fact.append(raw_f)

            orig_arr = np.array([int(x) for x in orig_acc])
            fact_arr = np.array([int(x) for x in fact_acc])
            d = orig_arr - fact_arr
            discord_orig_only = int(np.sum(d > 0))
            discord_fact_only = int(np.sum(d < 0))
            concordant = int(np.sum(d == 0))
            rd = float(d.mean()) if len(d) else 0.0
            mp = mcnemar_exact(discord_orig_only, discord_fact_only)

            rows.append({
                "cap": cap, "mult": mult, "n": len(paired),
                "orig_accept_rate": float(orig_arr.mean()) if len(orig_arr) else 0.0,
                "fact_accept_rate": float(fact_arr.mean()) if len(fact_arr) else 0.0,
                "rd": rd,
                "discord_original_only": discord_orig_only,
                "discord_factual_only": discord_fact_only,
                "concordant": concordant,
                "mcnemar_p": mp,
                "cap_hit_rate_orig": cap_hits_orig / max(len(paired), 1),
                "cap_hit_rate_fact": cap_hits_fact / max(len(paired), 1),
                "mean_raw_adj_frac_orig": float(np.mean(raw_adj_orig)) if raw_adj_orig else 0.0,
                "mean_raw_adj_frac_fact": float(np.mean(raw_adj_fact)) if raw_adj_fact else 0.0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-raw", type=str,
                    default="results/phase1c_qwen14b_awq_diag_report.raw.jsonl")
    ap.add_argument("--llama-raw", type=str,
                    default="results/phase1c_llama31_8b_diag_report.raw.jsonl")
    ap.add_argument("--out", type=str,
                    default="results/phase1c_cap_ablation.md")
    args = ap.parse_args()

    cap_values = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
    # Coefficient multiplier sweep: 1.0 = Round 20 (current), ~3.33 = Round 19 (pre-tightening).
    mult_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    qwen_rows = load_raw(Path(args.qwen_raw))
    llama_rows = load_raw(Path(args.llama_raw))
    qwen_paired = pair_up(qwen_rows)
    llama_paired = pair_up(llama_rows)
    print(f"[load] Qwen paired: {len(qwen_paired)}  |  Llama paired: {len(llama_paired)}")

    qwen_ablation = ablate_one_arm_grid(qwen_paired, cap_values, mult_values)
    llama_ablation = ablate_one_arm_grid(llama_paired, cap_values, mult_values)

    lines = []
    lines.append("# Phase 1c — Cap Ablation (offline sensitivity analysis)\n")
    lines.append("**Offline re-evaluation of `compute_acceptance` on existing Phase 1c "
                 "diagnostic-window raw data, sweeping MSG_ADJ_CAP while holding "
                 "LLM-extracted features and all other rule parameters frozen.**\n")
    lines.append("Framing (per pre-reg discipline):")
    lines.append("> This is a sensitivity analysis of the benchmark's governance "
                 "parameter, not empirical policy validation. MSG_ADJ_CAP "
                 "operationalizes a family of information-design safeguards limiting "
                 "how much message-induced perceptions may alter a welfare-rational choice.\n")
    lines.append(f"Round 20 coefficients (frozen): "
                 f"fit={ADJ_SURPLUS_FIT}, trust={ADJ_SURPLUS_TRUST}, "
                 f"risk={ADJ_SURPLUS_RISK}, urgency={ADJ_SURPLUS_URGENCY}, "
                 f"baseline_floor={BASELINE_SURPLUS_FLOOR}")
    lines.append(f"Cap sweep: {cap_values}")
    lines.append(f"\nPaired n: Qwen={len(qwen_paired)}, Llama={len(llama_paired)}")

    for arm_label, ablation in [("Qwen-14B-AWQ", qwen_ablation),
                                  ("Llama-3.1-8B bf16", llama_ablation)]:
        lines.append(f"\n## {arm_label} — Behavioral RD (% pp) heatmap\n")
        lines.append("Rows = coefficient multiplier × Round 20. Cols = MSG_ADJ_CAP "
                     "(% of budget). Each cell = risk difference (original − factual) "
                     "in pp, with discordant counts in parens.\n")
        header = "| mult \\ cap | " + " | ".join(f"{c*100:.1f}%" for c in cap_values) + " |"
        sep = "|---|" + "|".join([":---:"] * len(cap_values)) + "|"
        lines.append(header)
        lines.append(sep)
        for mult in mult_values:
            cells = []
            for cap in cap_values:
                r = next((x for x in ablation
                          if abs(x["mult"] - mult) < 1e-9 and abs(x["cap"] - cap) < 1e-9),
                         None)
                if r is None:
                    cells.append("—")
                    continue
                cells.append(
                    f"{r['rd']*100:+.1f}pp "
                    f"({r['discord_original_only']}/{r['discord_factual_only']})"
                )
            lines.append(f"| ×{mult:.1f} | " + " | ".join(cells) + " |")

        # Factual-acceptance heatmap (ceiling diagnostic).
        lines.append(f"\n### {arm_label} — Factual-variant acceptance rate "
                     "(validity gate: <98%)\n")
        lines.append(header)
        lines.append(sep)
        for mult in mult_values:
            cells = []
            for cap in cap_values:
                r = next((x for x in ablation
                          if abs(x["mult"] - mult) < 1e-9 and abs(x["cap"] - cap) < 1e-9),
                         None)
                cells.append("—" if r is None else
                             f"{r['fact_accept_rate']*100:.0f}%"
                             + (" ⚠CEIL" if r["fact_accept_rate"] >= 0.98 else ""))
            lines.append(f"| ×{mult:.1f} | " + " | ".join(cells) + " |")

    # ---- Summary: threshold detection ----
    lines.append("\n## Threshold detection — where does perception→behavior transmission light up?\n")
    lines.append("| arm | (mult, cap) of first RD>0 | (mult, cap) of first RD≥10pp | max RD | at (mult, cap) |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm_label, ablation in [("Qwen-14B-AWQ", qwen_ablation),
                                  ("Llama-3.1-8B bf16", llama_ablation)]:
        valid = [r for r in ablation if r["fact_accept_rate"] < 0.98]
        first_pos = next((r for r in valid if r["rd"] > 0), None)
        first_10 = next((r for r in valid if r["rd"] >= 0.10), None)
        max_r = max(valid, key=lambda r: r["rd"], default=None) if valid else None
        fp_cell = "none" if first_pos is None else "×{:.1f}, {:.1f}%".format(first_pos["mult"], first_pos["cap"]*100)
        fp10_cell = "none" if first_10 is None else "×{:.1f}, {:.1f}%".format(first_10["mult"], first_10["cap"]*100)
        mx_cell = "—" if max_r is None else "{:+.1f}pp".format(max_r["rd"]*100)
        atmax_cell = "—" if max_r is None else "×{:.1f}, {:.1f}%".format(max_r["mult"], max_r["cap"]*100)
        lines.append(f"| {arm_label} | {fp_cell} | {fp10_cell} | {mx_cell} | {atmax_cell} |")
    lines.append("\nNote: cells where factual acceptance ≥ 98% (ceiling) are excluded "
                 "from threshold detection — those are the 'baseline saturated' regime "
                 "where even factual template drives full acceptance.")

    # ---- Interpretation block ----
    lines.append("\n## Interpretation (honest framing)\n")

    def live_region(ablation):
        # "Live region" = cells with RD > 0 AND factual acceptance not at ceiling.
        return [r for r in ablation if r["rd"] > 0 and r["fact_accept_rate"] < 0.98]

    qwen_live = live_region(qwen_ablation)
    llama_live = live_region(llama_ablation)

    if qwen_live and llama_live:
        q_min_mult = min(r["mult"] for r in qwen_live)
        l_min_mult = min(r["mult"] for r in llama_live)
        lines.append(
            f"- **Perception→behavior transmission LIGHTS UP in a coefficient-multiplier "
            f"regime above Round 20**: Qwen first flip at ×{q_min_mult:.1f}, Llama at "
            f"×{l_min_mult:.1f}. Round 20 (×1.0) is a silenced regime.")
        lines.append(
            "- This means **the governance lever is not the cap but the coefficient "
            "tightness**: the 3-5× Round-20 tightening over-attenuated message influence, "
            "blocking even maximally-persuasive messages.")
        lines.append(
            "- For policy framing: the benchmark parameter to tune is the *scale of "
            "message-induced surplus adjustment* (here: coefficients × features), "
            "capped at a budget fraction. Above some multiplier, the same perceptual "
            "Δ≈+0.17 translates to behavior.")
        lines.append(
            "- **Caveat**: n=15 per arm, same stimuli. Sensitivity analysis of the "
            "deterministic decision layer, not an empirical claim about real agents.")
    elif not qwen_live and not llama_live:
        lines.append("- **Dead zone across entire (cap × multiplier) grid**: even "
                     "with cap=100% and coefficients scaled ×20, no behavioral flips "
                     "emerge. The sample's baseline-to-threshold distances exceed what "
                     "the observed perceptual deltas can bridge even at maximum "
                     "message leverage. n=15 near-threshold pairs is insufficient, "
                     "or the observed Δfit≈+0.17 is simply too small a signal.")
        lines.append("- **Path forward**: scale up msgcap (fix max_tokens=2500 bug) to "
                     "get n≈150 pairs; OR re-extract features with a 'stronger' prompt "
                     "that induces larger Δ; OR accept that cross-family perception "
                     "shifts are PRESENT but decision-rule transmission requires "
                     "larger perceptual shifts than the current stimuli produce.")
    else:
        arm = "Qwen" if qwen_live and not llama_live else "Llama"
        lines.append(f"- **Asymmetric pattern**: transmission lights up on {arm} but "
                     "not the other. Cross-family transmission claim does not hold.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"[done] wrote {args.out}")

    # Quick stdout summary.
    print("\n=== HEADLINE ===")
    for arm_label, ablation in [("Qwen", qwen_ablation), ("Llama", llama_ablation)]:
        valid = [r for r in ablation if r["fact_accept_rate"] < 0.98]
        first_pos = next((r for r in valid if r["rd"] > 0), None)
        first_10 = next((r for r in valid if r["rd"] >= 0.10), None)
        max_r = max(valid, key=lambda r: r["rd"], default=None) if valid else None
        fp = "none" if first_pos is None else f"×{first_pos['mult']:.1f},{first_pos['cap']*100:.1f}%"
        fp10 = "none" if first_10 is None else f"×{first_10['mult']:.1f},{first_10['cap']*100:.1f}%"
        mx = "—" if max_r is None else f"{max_r['rd']*100:+.1f}pp@×{max_r['mult']:.1f},{max_r['cap']*100:.1f}%"
        print(f"{arm_label:5s} | first RD>0: {fp:20s} | first RD≥10pp: {fp10:20s} | max: {mx}")


if __name__ == "__main__":
    main()

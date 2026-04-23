"""Phase 1d preflight — no-LLM structural support / yield calculator.

Written per PHASE1D_HBD_CALIBRATED_PREREG.md §4 Gate 1, as the replacement for
the previously ad-hoc `n < 40` post-hoc stopping rule. Runs under ALL candidate
priors (including A/C which we will NOT replay) so the paper can preempt the
reviewer question "what about actual low-ADR customers" without burning GPU.

Per prior, regenerates the same 900-scenario pool that phase1 msgcap samples
from (SCENARIOS_LOOSE × SEEDS × condition strata implied by run_phase1_msgcapture),
then computes, with NO LLM calls:

  - budget quantiles (p5, median, p95, mean)
  - package-price quantiles
  - feasible-bundle rate = fraction of (traveler, bundle) pairs with price ≤ budget
  - no-feasible-traveler rate = fraction of travelers with zero feasible bundles
  - all-pair near-threshold eligible count by (signal_wt, regime) stratum
    (upper bound on the msgcap-recommended near-threshold count; OTA selection
    is a subset)
  - archetype distribution (sanity check on HBD-weighted sampling)

Writes results/phase1d_preflight.md + .preflight_phase1d_pass / .preflight_phase1d_fail
markers. Pass criterion (pre-registered):
  - priors B (hbd_scale_normalized): feasible_bundle_rate ≥ 0.20 AND
    no_feasible_traveler_rate ≤ 0.30 AND near-threshold-eligible ≥ 600 total.

Usage:
  python scripts/run_phase1d_preflight.py \\
    --out results/phase1d_preflight.md \\
    --pass-marker results/.preflight_phase1d_pass \\
    --fail-marker results/.preflight_phase1d_fail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tourmart.llm_agents import DEFAULT_ACCEPTANCE_THRESHOLDS
from tourmart.oracle import package_price
from tourmart.scenarios import generate_small_market


# Same ranges as run_phase1_msgcapture.py (v4 frozen).
SCENARIOS_LOOSE = list(range(1000, 1075))
SCENARIOS_TIGHT = list(range(1500, 1575))
SIGNAL_WTS = (0.25, 0.50, 0.75)

PRIORS_TO_REPORT = (
    "uniform",              # v4 reference
    "hbd_scale_normalized", # B primary
    "hbd_archetype_only",   # D secondary
    "hbd_direct",           # A preflight-only
    "hbd_3x_hotel",         # C preflight-only
)

# Structural pass criterion for PRIMARY prior B.
B_FEASIBLE_BUNDLE_RATE_MIN = 0.20
B_NO_FEASIBLE_TRAVELER_RATE_MAX = 0.30
B_NEAR_THRESHOLD_TOTAL_MIN = 600


def _window_for_diag(budget: float, tau: float) -> tuple[float, float]:
    """Diagnostic near-threshold window [τb − 10%b, τb + 5%b] — same as phase1c
    when --diagnostic-window is set."""
    thresh = tau * budget
    lo = thresh - 0.10 * budget
    hi = thresh + 0.05 * budget
    return lo, hi


def audit_prior(priors_mode: str) -> dict:
    """Regenerate the 150-market pool (75 loose + 75 tight) under this prior and
    collect structural statistics."""
    stats: dict = {"priors_mode": priors_mode, "per_stratum": {}}
    all_budgets: list[float] = []
    all_prices: list[float] = []
    feasible_pair_rates: list[float] = []
    no_feasible_traveler_flags: list[int] = []
    archetype_counts: dict[str, int] = {}

    # Near-threshold upper bound: counted over ALL (traveler, bundle) pairs in
    # each market, before OTA selection. For each market, this gives a strict
    # upper bound on msgcap-extractable near-threshold pairs (OTA recommends a
    # subset of bundles).
    near_thresh_count_by_stratum: dict[tuple[float, str], int] = {}

    regimes = (("loose", SCENARIOS_LOOSE), ("tight", SCENARIOS_TIGHT))
    for regime_label, scen_list in regimes:
        for s in scen_list:
            m = generate_small_market(s, regime_label, priors_mode=priors_mode)
            for t in m.travelers:
                all_budgets.append(float(t.budget))
                archetype_counts[t.archetype.id] = \
                    archetype_counts.get(t.archetype.id, 0) + 1
                feasible = 0
                for b in m.bundles:
                    if b.id not in t.utility:
                        continue
                    p = package_price(b, m)
                    all_prices.append(p)
                    if p <= t.budget + 1e-9:
                        feasible += 1
                denom = sum(1 for b in m.bundles if b.id in t.utility)
                if denom > 0:
                    feasible_pair_rates.append(feasible / denom)
                no_feasible_traveler_flags.append(int(feasible == 0))

                # Near-threshold count over traveler × all-bundles.
                tau = DEFAULT_ACCEPTANCE_THRESHOLDS.get(t.archetype.id, 0.10)
                lo, hi = _window_for_diag(float(t.budget), tau)
                for b in m.bundles:
                    if b.id not in t.utility:
                        continue
                    p = package_price(b, m)
                    if p > t.budget + 1e-9:
                        continue  # infeasible
                    surplus = float(t.utility[b.id]) - p
                    if lo <= surplus <= hi:
                        for sw in SIGNAL_WTS:
                            key = (sw, regime_label)
                            near_thresh_count_by_stratum[key] = \
                                near_thresh_count_by_stratum.get(key, 0) + 1

    b_arr = np.array(all_budgets)
    p_arr = np.array(all_prices)
    stats["budget"] = {
        "n": len(b_arr),
        "mean": float(b_arr.mean()),
        "median": float(np.median(b_arr)),
        "p5": float(np.percentile(b_arr, 5)),
        "p25": float(np.percentile(b_arr, 25)),
        "p75": float(np.percentile(b_arr, 75)),
        "p95": float(np.percentile(b_arr, 95)),
    }
    stats["package_price"] = {
        "n": len(p_arr),
        "mean": float(p_arr.mean()),
        "median": float(np.median(p_arr)),
        "p5": float(np.percentile(p_arr, 5)),
        "p25": float(np.percentile(p_arr, 25)),
        "p75": float(np.percentile(p_arr, 75)),
        "p95": float(np.percentile(p_arr, 95)),
    }
    stats["feasible_bundle_rate"] = float(np.mean(feasible_pair_rates)) \
        if feasible_pair_rates else 0.0
    stats["no_feasible_traveler_rate"] = \
        float(np.mean(no_feasible_traveler_flags)) \
        if no_feasible_traveler_flags else 0.0
    stats["archetype_counts"] = archetype_counts

    # Near-threshold: report per-stratum and total. Note: per-(sw, regime)
    # stratum count is identical across signal_wts because msgcap only stratifies
    # OTA-recommended pairs by signal_wt, not the underlying near-threshold
    # population. We report it as a structural upper bound per stratum.
    per_stratum = {}
    for (sw, regime_label), cnt in near_thresh_count_by_stratum.items():
        per_stratum.setdefault(regime_label, {})[f"sw={sw}"] = cnt
    stats["near_threshold"] = {
        "per_stratum": per_stratum,
        "total_all_pair_upper_bound": sum(near_thresh_count_by_stratum.values()),
    }
    return stats


def render_markdown(all_stats: list[dict]) -> str:
    lines = ["# Phase 1d preflight — structural support / yield audit\n"]
    lines.append("No-LLM regeneration of the 150-market pool under each candidate prior. "
                 "Reports budget/price distributions, feasibility, and near-threshold "
                 "eligible counts as a strict upper bound on msgcap-extractable "
                 "near-threshold pairs after OTA selection.\n")

    lines.append("## Summary\n")
    lines.append("| priors_mode | budget mean/median | feasible bundle rate | "
                 "no-feasible-traveler rate | near-threshold total (upper bound) |")
    lines.append("|---|---|---:|---:|---:|")
    for s in all_stats:
        bud = s["budget"]
        lines.append(
            f"| `{s['priors_mode']}` | €{bud['mean']:.0f} / €{bud['median']:.0f} | "
            f"{s['feasible_bundle_rate']:.1%} | "
            f"{s['no_feasible_traveler_rate']:.1%} | "
            f"{s['near_threshold']['total_all_pair_upper_bound']} |"
        )

    lines.append("\n## Per-prior details\n")
    for s in all_stats:
        lines.append(f"### `{s['priors_mode']}`\n")
        lines.append(f"- Budget: n={s['budget']['n']}  mean=€{s['budget']['mean']:.0f}  "
                     f"p5=€{s['budget']['p5']:.0f}  median=€{s['budget']['median']:.0f}  "
                     f"p95=€{s['budget']['p95']:.0f}")
        lines.append(f"- Package price: n={s['package_price']['n']}  "
                     f"mean=€{s['package_price']['mean']:.0f}  "
                     f"p5=€{s['package_price']['p5']:.0f}  "
                     f"median=€{s['package_price']['median']:.0f}  "
                     f"p95=€{s['package_price']['p95']:.0f}")
        lines.append(f"- Feasible bundle rate: {s['feasible_bundle_rate']:.2%}")
        lines.append(f"- No-feasible-traveler rate: {s['no_feasible_traveler_rate']:.2%}")
        lines.append(f"- Near-threshold total (all-pair upper bound): "
                     f"{s['near_threshold']['total_all_pair_upper_bound']}")
        lines.append(f"- Near-threshold per-regime: "
                     f"{json.dumps(s['near_threshold']['per_stratum'])}")
        lines.append(f"- Archetype counts: {json.dumps(s['archetype_counts'])}\n")

    b = next(s for s in all_stats if s["priors_mode"] == "hbd_scale_normalized")
    fr = b["feasible_bundle_rate"]
    nftr = b["no_feasible_traveler_rate"]
    nt = b["near_threshold"]["total_all_pair_upper_bound"]
    gate_1 = (fr >= B_FEASIBLE_BUNDLE_RATE_MIN) \
        and (nftr <= B_NO_FEASIBLE_TRAVELER_RATE_MAX) \
        and (nt >= B_NEAR_THRESHOLD_TOTAL_MIN)
    lines.append("\n## Pre-registered Gate 1 (primary prior B = `hbd_scale_normalized`)\n")
    lines.append(f"- Feasible bundle rate ≥ {B_FEASIBLE_BUNDLE_RATE_MIN:.0%}: "
                 f"{fr:.2%}  {'✅' if fr >= B_FEASIBLE_BUNDLE_RATE_MIN else '❌'}")
    lines.append(f"- No-feasible-traveler rate ≤ {B_NO_FEASIBLE_TRAVELER_RATE_MAX:.0%}: "
                 f"{nftr:.2%}  {'✅' if nftr <= B_NO_FEASIBLE_TRAVELER_RATE_MAX else '❌'}")
    lines.append(f"- Near-threshold upper bound ≥ {B_NEAR_THRESHOLD_TOTAL_MIN}: "
                 f"{nt}  {'✅' if nt >= B_NEAR_THRESHOLD_TOTAL_MIN else '❌'}")
    lines.append(f"\n**Overall gate 1: {'PASS' if gate_1 else 'FAIL'}**")
    if not gate_1:
        lines.append("\n> Abort phase1d replay under B. Report preflight diagnostics for "
                     "{B, D, A, C} as the §5.2 finding: no support overlap at HBD-shape "
                     "magnitude anchored to €3250. Direct HBD and 3× hotel spend priors "
                     "will be reported as support-inadequate for the synthetic package "
                     "market regime.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/phase1d_preflight.md")
    ap.add_argument("--pass-marker", type=str,
                    default="results/.preflight_phase1d_pass")
    ap.add_argument("--fail-marker", type=str,
                    default="results/.preflight_phase1d_fail")
    ap.add_argument("--json-out", type=str,
                    default="results/phase1d_preflight.json",
                    help="Machine-readable structural stats.")
    args = ap.parse_args()

    all_stats = []
    for priors_mode in PRIORS_TO_REPORT:
        print(f"[preflight] auditing {priors_mode}...", file=sys.stderr)
        s = audit_prior(priors_mode)
        all_stats.append(s)
        print(f"  budget: mean=€{s['budget']['mean']:.0f} "
              f"median=€{s['budget']['median']:.0f}  "
              f"feasible_rate={s['feasible_bundle_rate']:.1%}  "
              f"nt_total={s['near_threshold']['total_all_pair_upper_bound']}",
              file=sys.stderr)

    md = render_markdown(all_stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    Path(args.json_out).write_text(json.dumps(all_stats, indent=2))
    print(f"[preflight] wrote {args.out}", file=sys.stderr)
    print(f"[preflight] wrote {args.json_out}", file=sys.stderr)

    b = next(s for s in all_stats if s["priors_mode"] == "hbd_scale_normalized")
    gate_1 = (
        b["feasible_bundle_rate"] >= B_FEASIBLE_BUNDLE_RATE_MIN
        and b["no_feasible_traveler_rate"] <= B_NO_FEASIBLE_TRAVELER_RATE_MAX
        and b["near_threshold"]["total_all_pair_upper_bound"]
        >= B_NEAR_THRESHOLD_TOTAL_MIN
    )
    marker = args.pass_marker if gate_1 else args.fail_marker
    Path(marker).touch()
    print(f"[preflight] gate_1 {'PASS' if gate_1 else 'FAIL'} → {marker}",
          file=sys.stderr)
    return 0 if gate_1 else 1


if __name__ == "__main__":
    sys.exit(main())

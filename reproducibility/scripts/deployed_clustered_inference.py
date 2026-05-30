#!/usr/bin/env python3
"""Deployed-point (lambda=1, kappa=0.05) cluster-robust inference, 5-tuple pairing.

Red-team blocker: the paper's deployed-point significance uses naive i.i.d. exact
McNemar on n=409 pairs, but the 5-tuple un-collapses correlated (signal_wt,
episode_seed) pseudo-replicates and the paper's GRID test clusters on scenario_id.
This script reports, for the deployed cell ONLY, both the naive exact-McNemar p and
a scenario-clustered permutation p + scenario-clustered bootstrap 95% CI, so the
headline test matches the inference model the grid already uses.

GUARD (per "mind the n"): asserts the deployed-cell paired counts reproduce the
paper's Table 1 (Qwen n=409 b/c=24/2 RD~+5.38pp; Llama n=409 b/c=8/0 RD~+1.96pp)
BEFORE reporting the clustered p. If they don't match, the pairing/cell is wrong.
"""
import sys, os, json
import numpy as np
from scipy.stats import binomtest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from reproduce_permutation_5tuple import load_raw, pair_up_5tuple
from run_cap_ablation import compute_acceptance_with_cap

DEPLOYED_CAP = 0.05      # kappa
DEPLOYED_MULT = 1.0      # lambda
SEED = 12345
N_PERM = 10000
N_BOOT = 10000

# Paper Table 1 expected deployed-point values (for the n-guard).
EXPECT = {
    "qwen14b_awq": {"n": 409, "b": 24, "c": 2,  "rd_pp": 5.38},
    "llama31_8b":  {"n": 409, "b": 8,  "c": 0,  "rd_pp": 1.96},
}


def deployed_d(paired):
    """d in {-1,0,+1} per pair at the deployed cell; +1 = original-only accept."""
    d = np.empty(len(paired), dtype=np.int8)
    clusters = []
    for i, p in enumerate(paired):
        ao, _, _ = compute_acceptance_with_cap(p['features_original'], p['baseline_surplus'],
                                               p['budget'], p['tau'], DEPLOYED_CAP, DEPLOYED_MULT)
        af, _, _ = compute_acceptance_with_cap(p['features_factual'], p['baseline_surplus'],
                                               p['budget'], p['tau'], DEPLOYED_CAP, DEPLOYED_MULT)
        d[i] = int(ao) - int(af)
        clusters.append(p['scenario_id'])
    return d, np.array(clusters)


def clustered_perm_p(d, clusters, n_perm, seed):
    """Scenario-clustered permutation: flip all pairs in a cluster jointly (sign flip),
    two-sided p on |RD|. Mirrors the grid permutation scheme for one cell."""
    rng = np.random.default_rng(seed)
    uniq = {c: i for i, c in enumerate(sorted(set(clusters)))}
    cidx = np.array([uniq[c] for c in clusters])
    nC = len(uniq)
    obs = float(d.mean())
    ge = 0
    for _ in range(n_perm):
        flip = rng.integers(0, 2, size=nC)[cidx]      # 0/1 per pair
        signs = np.where(flip, -1, 1)
        rd = float((d * signs).mean())
        if abs(rd) >= abs(obs) - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1), obs


def clustered_bootstrap_ci(d, clusters, n_boot, seed):
    rng = np.random.default_rng(seed + 1)
    uniq = sorted(set(clusters))
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    uniq = np.array(uniq, dtype=object)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[j]] for j in pick])
        boots[k] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    qwen_raw = sys.argv[1]
    llama_raw = sys.argv[2]
    for arm, path in [("qwen14b_awq", qwen_raw), ("llama31_8b", llama_raw)]:
        paired = pair_up_5tuple(load_raw(path))
        d, clusters = deployed_d(paired)
        n = len(d); b = int((d > 0).sum()); c = int((d < 0).sum())
        rd_pp = float(d.mean()) * 100
        exp = EXPECT[arm]
        guard_ok = (n == exp["n"] and b == exp["b"] and c == exp["c"]
                    and abs(rd_pp - exp["rd_pp"]) < 0.05)
        print(f"\n=== {arm} — deployed (lambda=1, kappa=0.05), 5-tuple ===")
        print(f"  n_paired={n}  b(orig-only)={b}  c(fact-only)={c}  RD={rd_pp:+.2f}pp  "
              f"clusters={len(set(clusters))}")
        print(f"  N-GUARD vs paper Table 1 (n={exp['n']}, b/c={exp['b']}/{exp['c']}, "
              f"RD~{exp['rd_pp']:+.2f}pp): {'PASS' if guard_ok else '*** MISMATCH — STOP ***'}")
        if not guard_ok:
            print("  Refusing to report clustered p: pairing/cell does not reproduce the paper.")
            continue
        # naive exact McNemar (what the paper currently reports)
        p_naive = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        # cluster-robust
        p_clust, obs = clustered_perm_p(d, clusters, N_PERM, SEED)
        lo, hi = clustered_bootstrap_ci(d, clusters, N_BOOT, SEED)
        print(f"  naive exact McNemar p      = {p_naive:.4f}   (paper's headline test)")
        print(f"  scenario-CLUSTERED perm p  = {p_clust:.4f}   ({N_PERM} perms, seed {SEED})")
        print(f"  scenario-CLUSTERED 95% CI  = [{lo*100:+.2f}, {hi*100:+.2f}] pp")
        sig_naive = "sig" if p_naive < 0.05 else "NON-sig"
        sig_clust = "sig" if p_clust < 0.05 else "NON-sig"
        excl0 = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
        print(f"  -> naive {sig_naive} (p={p_naive:.4f}); clustered {sig_clust} (p={p_clust:.4f}); "
              f"clustered CI {excl0}")


if __name__ == "__main__":
    main()

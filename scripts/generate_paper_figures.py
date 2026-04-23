"""Generate publication figures for TourMart paper.

Produces 3 figures in figures/:
  1. fig1_behavioral_heatmap.pdf / .png — 2D RD heatmap (Qwen + Llama side-by-side)
  2. fig2_coefficient_attribution.pdf / .png — bar chart of max RD per coefficient-zeroing condition
  3. fig3_sample_size_trajectory.pdf / .png — max RD + peak discord across v2 (n=15) → v3 (n=48) → v4 (n=143)

Uses matplotlib only. Pure offline, no GPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Style
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ======================================================================
# Data extraction helpers (reused from run_cap_ablation + attribution)
# ======================================================================

ADJ_FIT = 0.03
ADJ_TRUST = 0.015
ADJ_RISK = 0.025
ADJ_URGENCY = 0.01
FLOOR = -0.10


def compute_acceptance(features, baseline_surplus, budget, tau, cap, mult=1.0,
                        coefs=None):
    if coefs is None:
        coefs = {"fit": ADJ_FIT, "trust": ADJ_TRUST, "risk": ADJ_RISK,
                 "urgency": ADJ_URGENCY}
    if baseline_surplus < FLOOR * budget:
        return False
    fit = float(features.get("perceived_fit_delta", 0.0))
    risk = float(features.get("perceived_risk", 0.0))
    trust = float(features.get("trust_score", 0.0))
    urgency = float(features.get("urgency_felt", 0.0))
    raw = mult * budget * (coefs["fit"] * fit + coefs["trust"] * trust
                            - coefs["risk"] * risk + coefs["urgency"] * urgency)
    cap_abs = cap * budget
    adj = baseline_surplus + max(-cap_abs, min(cap_abs, raw))
    return adj >= tau * budget


def load_paired(raw_path):
    rows = []
    with open(raw_path) as f:
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


def rd_heatmap(paired, cap_values, mult_values, coefs=None):
    rd = np.zeros((len(mult_values), len(cap_values)))
    discord_pos = np.zeros_like(rd, dtype=int)
    discord_neg = np.zeros_like(rd, dtype=int)
    fact_rate = np.zeros_like(rd)
    for i, mult in enumerate(mult_values):
        for j, cap in enumerate(cap_values):
            orig_arr = []
            fact_arr = []
            for p in paired:
                ao = compute_acceptance(p["features_original"], p["baseline_surplus"],
                                         p["budget"], p["tau"], cap, mult, coefs)
                af = compute_acceptance(p["features_factual"], p["baseline_surplus"],
                                         p["budget"], p["tau"], cap, mult, coefs)
                orig_arr.append(int(ao)); fact_arr.append(int(af))
            orig_arr = np.array(orig_arr); fact_arr = np.array(fact_arr)
            rd[i, j] = (orig_arr - fact_arr).mean() * 100
            discord_pos[i, j] = int(np.sum(orig_arr > fact_arr))
            discord_neg[i, j] = int(np.sum(orig_arr < fact_arr))
            fact_rate[i, j] = fact_arr.mean()
    return rd, discord_pos, discord_neg, fact_rate


# ======================================================================
# Figure 1: 2D Behavioral RD Heatmap (Qwen + Llama side-by-side)
# ======================================================================

def figure_1():
    cap_values = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
    mult_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    qwen_paired = load_paired("results/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl")
    llama_paired = load_paired("results/phase1c_llama31_8b_diag_v4_report.raw.jsonl")
    print(f"Fig1: Qwen n={len(qwen_paired)}, Llama n={len(llama_paired)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, paired, title_label in [(axes[0], qwen_paired, "Qwen-14B-AWQ"),
                                       (axes[1], llama_paired, "Llama-3.1-8B")]:
        rd, disc_p, disc_n, fact = rd_heatmap(paired, cap_values, mult_values)
        # Mask cells where factual >= 0.98 (saturated)
        masked = np.where(fact >= 0.98, np.nan, rd)
        vmax = max(12, np.nanmax(masked))
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(cap_values)))
        ax.set_xticklabels([f"{c*100:g}%" for c in cap_values])
        ax.set_yticks(range(len(mult_values)))
        ax.set_yticklabels([f"×{m:g}" for m in mult_values])
        ax.set_xlabel("MSG_ADJ_CAP (% of budget)")
        ax.set_ylabel("Coefficient multiplier")
        ax.set_title(f"{title_label} (n={len(paired)})")

        # Annotate cells
        for i in range(rd.shape[0]):
            for j in range(rd.shape[1]):
                if fact[i, j] >= 0.98:
                    txt = "ceil"
                    color = "gray"
                else:
                    txt = f"{rd[i, j]:+.1f}\n{disc_p[i, j]}/{disc_n[i, j]}"
                    color = "white" if abs(rd[i, j]) > 6 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color,
                        fontsize=7)

        # Highlight Round 20 deployed cell (×1, cap=5%)
        j_dep = cap_values.index(0.05)
        i_dep = mult_values.index(1.0)
        ax.add_patch(plt.Rectangle((j_dep - 0.5, i_dep - 0.5), 1, 1,
                                     fill=False, edgecolor="lime", lw=2.5))

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20,
                         label="Risk difference RD (pp)")

    fig.suptitle("Figure 1. Cross-family behavioral steering phase diagram\n"
                 "(original − factual, cell format: RD / discord$_+$/discord$_-$)",
                 fontsize=11, y=1.02)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig1_behavioral_heatmap.{ext}"
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


# ======================================================================
# Figure 2: Coefficient attribution
# ======================================================================

def figure_2():
    cap_values = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
    mult_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    qwen_paired = load_paired("results/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl")
    llama_paired = load_paired("results/phase1c_llama31_8b_diag_v4_report.raw.jsonl")

    def max_valid_rd(paired, coefs):
        max_rd = -1e9
        for mult in mult_values:
            for cap in cap_values:
                orig_arr = []; fact_arr = []
                for p in paired:
                    ao = compute_acceptance(p["features_original"], p["baseline_surplus"],
                                             p["budget"], p["tau"], cap, mult, coefs)
                    af = compute_acceptance(p["features_factual"], p["baseline_surplus"],
                                             p["budget"], p["tau"], cap, mult, coefs)
                    orig_arr.append(int(ao)); fact_arr.append(int(af))
                orig_arr = np.array(orig_arr); fact_arr = np.array(fact_arr)
                if fact_arr.mean() >= 0.98: continue
                rd = (orig_arr - fact_arr).mean() * 100
                if rd > max_rd: max_rd = rd
        return max_rd

    full = {"fit": ADJ_FIT, "trust": ADJ_TRUST, "risk": ADJ_RISK, "urgency": ADJ_URGENCY}
    conds = [
        ("Full rule", full),
        ("No fit", {**full, "fit": 0.0}),
        ("No trust", {**full, "trust": 0.0}),
        ("No risk", {**full, "risk": 0.0}),
        ("No urgency", {**full, "urgency": 0.0}),
        ("Fit only", {"fit": ADJ_FIT, "trust": 0, "risk": 0, "urgency": 0}),
        ("Trust only", {"fit": 0, "trust": ADJ_TRUST, "risk": 0, "urgency": 0}),
    ]
    qwen_rds = [max_valid_rd(qwen_paired, c) for _, c in conds]
    llama_rds = [max_valid_rd(llama_paired, c) for _, c in conds]
    labels = [l for l, _ in conds]

    print(f"Fig2: Qwen RDs {[f'{r:.2f}' for r in qwen_rds]}")
    print(f"      Llama RDs {[f'{r:.2f}' for r in llama_rds]}")

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w/2, qwen_rds, w, label="Qwen-14B-AWQ", color="#2E86AB")
    b2 = ax.bar(x + w/2, llama_rds, w, label="Llama-3.1-8B", color="#E63946")

    ax.axhline(y=qwen_rds[0], color="#2E86AB", linestyle=":", alpha=0.5, lw=1)
    ax.axhline(y=llama_rds[0], color="#E63946", linestyle=":", alpha=0.5, lw=1)

    for bar_group in (b1, b2):
        for rect in bar_group:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.3, f"{h:+.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Max observed RD (pp, validity-filtered)")
    ax.set_title("Figure 2. Coefficient attribution — fit_delta is the load-bearing channel\n"
                 "(dotted lines = full-rule baseline; bars show max RD over 2D grid)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 15)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig2_coefficient_attribution.{ext}"
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


# ======================================================================
# Figure 3: Sample-size trajectory (v2 → v3 → v4)
# ======================================================================

def figure_3():
    # Hand-compiled from ROUND_21_FINAL.md + earlier runs
    # (script could load each raw.jsonl but for one-off figure we hardcode)
    # Phase 1c v1/v2/v3/v4 peak cells at Round 20 baseline grid
    stages = [
        # (label, n_paired, qwen_peak_discord_pos, qwen_peak_rd_pp, llama_peak_discord_pos, llama_peak_rd_pp, qwen_cluster_p, llama_cluster_p)
        ("v1\n(Phase 1b)", 122, None, None, None, None, None, None),   # feature only; behavioral null
        ("v2\n(Round 21 scale-up)", 15, 6, 13.3, 5, 13.3, 0.027, 0.188),
        ("v3\n(msgcap_v3)", 48, 6, 12.5, 5, 8.3, None, None),         # didn't compute cluster
        ("v4\n(msgcap_v4)", 143, 15, 10.5, 13, 7.7, 0.001, 0.001),
    ]

    labels = [s[0] for s in stages]
    ns = [s[1] for s in stages]
    qwen_disc = [s[2] for s in stages]
    qwen_rd = [s[3] for s in stages]
    llama_disc = [s[4] for s in stages]
    llama_rd = [s[5] for s in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    x = np.arange(len(stages))

    # Left: max RD at peak cell
    q_rd_plot = [r if r is not None else np.nan for r in qwen_rd]
    l_rd_plot = [r if r is not None else np.nan for r in llama_rd]
    ax1.plot(x, q_rd_plot, marker="o", color="#2E86AB", lw=2, label="Qwen-14B peak RD")
    ax1.plot(x, l_rd_plot, marker="s", color="#E63946", lw=2, label="Llama-8B peak RD")
    for i, (q, l) in enumerate(zip(q_rd_plot, l_rd_plot)):
        if not np.isnan(q):
            ax1.annotate(f"{q:.1f}", (i, q), textcoords="offset points", xytext=(0, 7), ha="center",
                         fontsize=8, color="#2E86AB")
        if not np.isnan(l):
            ax1.annotate(f"{l:.1f}", (i, l), textcoords="offset points", xytext=(0, -13), ha="center",
                         fontsize=8, color="#E63946")
    ax1.axhline(5, color="gray", linestyle="--", alpha=0.5, lw=1)
    ax1.axhline(10, color="gray", linestyle="-.", alpha=0.5, lw=1)
    ax1.text(len(stages) - 0.7, 10.1, "10pp GO threshold", fontsize=8, color="gray")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Peak RD over 2D grid (pp)")
    ax1.set_title("(a) Peak behavioral RD")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 17)

    # Right: discord count at peak (showing effect size scales with n)
    q_d = [d if d is not None else np.nan for d in qwen_disc]
    l_d = [d if d is not None else np.nan for d in llama_disc]
    ax2.plot(x, q_d, marker="o", color="#2E86AB", lw=2, label="Qwen-14B discord$_+$")
    ax2.plot(x, l_d, marker="s", color="#E63946", lw=2, label="Llama-8B discord$_+$")
    ax2b = ax2.twinx()
    ax2b.bar(x, ns, alpha=0.18, color="#808080", label="n paired")
    for i, (q, l, n) in enumerate(zip(q_d, l_d, ns)):
        if not np.isnan(q):
            ax2.annotate(f"{int(q)}", (i, q), textcoords="offset points", xytext=(-12, 5), ha="center",
                         fontsize=8, color="#2E86AB")
        if not np.isnan(l):
            ax2.annotate(f"{int(l)}", (i, l), textcoords="offset points", xytext=(12, 5), ha="center",
                         fontsize=8, color="#E63946")
        ax2b.annotate(f"n={n}", (i, n), textcoords="offset points", xytext=(0, 2), ha="center",
                      fontsize=7, color="#555")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Discord$_+$ count at peak cell")
    ax2b.set_ylabel("n paired (bars)", color="#555")
    ax2.set_title("(b) Discordant pair counts + sample size")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 20)

    fig.suptitle("Figure 3. Evidence trajectory across round-21 scale-up stages",
                 fontsize=11, y=1.02)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig3_sample_size_trajectory.{ext}"
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Figure 1 (heatmap)...")
    figure_1()
    print("\nGenerating Figure 2 (coefficient attribution)...")
    figure_2()
    print("\nGenerating Figure 3 (sample-size trajectory)...")
    figure_3()
    print("\nAll figures written to figures/")

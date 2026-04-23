#!/usr/bin/env python3
"""TourMart reproducibility verifier.

Checks:
  1. Input raw.jsonl SHA-256 checksums against known-good values.
  2. (Optional) Permutation summary p-values and headline numbers against
     expected_outputs/permutation_summary.json.

Usage:
  # Check inputs only (fast, no computation):
  python3 verify.py --check-inputs-only --results-dir /path/to/results

  # Full check (inputs + compare run outputs to expected):
  python3 verify.py --results-dir /path/to/results \\
                    --expected-dir expected_outputs \\
                    --run-outputs run_outputs/permutation_null
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known-good checksums for the shipped input data files.
# These are the SHA-256 digests of the files as they exist in results/.
# ---------------------------------------------------------------------------
INPUT_CHECKSUMS = {
    "phase1c_qwen14b_awq_diag_v4_report.raw.jsonl":
        "9c0582b4a535b27563363f3513d4fdf3656bbcf95744758a66338579fe488684",
    "phase1c_llama31_8b_diag_v4_report.raw.jsonl":
        "21ba196bc280e58d1ac59532ea73adadc4d23497beeb8063c0e112aecc3195c1",
    "phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl":
        "b5fc95d11985a99ef33ab0e39e557daf20450b3991503defb2c94a56d921c027",
    "phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl":
        "484008b7822463bf72c464d6f0bf9332ade2fe94e4c7a7777dcbdbad8b10ec87",
}

# ---------------------------------------------------------------------------
# Paper headline numbers for sanity-check (from paper abstract + §D).
# These are the EXPECTED values from reproduce_permutation.py output.
# ---------------------------------------------------------------------------
EXPECTED_HEADLINE = {
    "qwen14b_awq": {
        "n_paired_stimuli": 143,
        "n_clusters": 88,
        "observed_max_abs_rd_pp": 10.49,   # ±0.01pp tolerance
        "p_two_sided": 0.001,              # ≤ this value
    },
    "llama31_8b": {
        "n_paired_stimuli": 143,
        "n_clusters": 88,
        "observed_max_abs_rd_pp": 7.69,    # ±0.01pp tolerance
        "p_two_sided": 0.009,              # ≤ this value (paper: p=0.008)
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs(results_dir: Path) -> bool:
    print("--- Input data checksum verification ---")
    ok = True
    for fname, expected_hash in INPUT_CHECKSUMS.items():
        fpath = results_dir / fname
        if not fpath.exists():
            print(f"  MISSING : {fname}")
            ok = False
            continue
        actual = sha256_file(fpath)
        if actual == expected_hash:
            print(f"  OK      : {fname}")
        else:
            print(f"  MISMATCH: {fname}")
            print(f"            expected: {expected_hash}")
            print(f"            actual  : {actual}")
            ok = False
    return ok


def check_permutation_summary(summary_path: Path) -> bool:
    print("--- Permutation summary headline check ---")
    if not summary_path.exists():
        print(f"  MISSING: {summary_path}")
        return False
    with open(summary_path) as f:
        summary = json.load(f)
    ok = True
    arms = summary.get("arms", {})
    for arm_key, expected in EXPECTED_HEADLINE.items():
        if arm_key not in arms:
            print(f"  MISSING arm: {arm_key}")
            ok = False
            continue
        arm = arms[arm_key]
        n_pairs = arm.get("n_paired_stimuli")
        n_clust = arm.get("n_clusters")
        obs_rd = arm.get("observed", {}).get("max_abs_rd", None)
        p_val = arm.get("p_values", {}).get("two_sided_max_abs", None)

        checks = [
            (n_pairs == expected["n_paired_stimuli"],
             f"n_paired_stimuli={n_pairs} (expected {expected['n_paired_stimuli']})"),
            (n_clust == expected["n_clusters"],
             f"n_clusters={n_clust} (expected {expected['n_clusters']})"),
            (obs_rd is not None and abs(obs_rd * 100 - expected["observed_max_abs_rd_pp"]) < 0.02,
             f"max_abs_rd={obs_rd*100:.2f}pp (expected ~{expected['observed_max_abs_rd_pp']:.2f}pp)"),
            (p_val is not None and p_val <= expected["p_two_sided"],
             f"p_two_sided={p_val:.4f} (expected ≤{expected['p_two_sided']:.3f})"),
        ]
        arm_ok = all(flag for flag, _ in checks)
        status = "PASS" if arm_ok else "FAIL"
        print(f"  [{status}] {arm_key}")
        for flag, msg in checks:
            marker = "  " if flag else "!!"
            print(f"    {marker} {msg}")
        if not arm_ok:
            ok = False
    return ok


def compare_with_expected(run_dir: Path, expected_dir: Path) -> bool:
    """Compare run outputs to expected_outputs/ by headline numbers."""
    run_summary = run_dir / "permutation_summary.json"
    return check_permutation_summary(run_summary)


def main():
    ap = argparse.ArgumentParser(description="TourMart reproducibility verifier")
    ap.add_argument("--results-dir", required=True,
                    help="Path to tourmart/results/ (contains raw.jsonl files)")
    ap.add_argument("--expected-dir", default=None,
                    help="Path to expected_outputs/ directory (default: sibling of this script)")
    ap.add_argument("--run-outputs", default=None,
                    help="Path to directory containing permutation_summary.json from a fresh run")
    ap.add_argument("--check-inputs-only", action="store_true",
                    help="Only verify input checksums; skip output comparison")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    script_dir = Path(__file__).parent
    expected_dir = Path(args.expected_dir) if args.expected_dir else (script_dir / "expected_outputs")

    all_ok = True

    # Step 1: input checksums
    if not check_inputs(results_dir):
        print("\nERROR: Input checksum verification failed.")
        print("       Raw data files may have been modified or are incorrect versions.")
        all_ok = False
    else:
        print("\nInput checksums: ALL PASS")

    if args.check_inputs_only:
        sys.exit(0 if all_ok else 1)

    print()

    # Step 2: compare shipped expected outputs against paper headlines
    print("--- Checking expected_outputs/permutation_summary.json against paper numbers ---")
    if not check_permutation_summary(expected_dir / "permutation_summary.json"):
        print("ERROR: Expected outputs do not match paper headline numbers.")
        all_ok = False
    else:
        print("Expected outputs: ALL PASS")

    print()

    # Step 3: if fresh run outputs provided, compare against expected
    if args.run_outputs:
        run_dir = Path(args.run_outputs)
        print(f"--- Comparing fresh run outputs ({run_dir}) against expected ---")
        if not compare_with_expected(run_dir, expected_dir):
            print("ERROR: Fresh run outputs differ from expected headline numbers.")
            all_ok = False
        else:
            print("Fresh run outputs: ALL PASS")

    print()
    if all_ok:
        print("ALL CHECKS PASSED.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

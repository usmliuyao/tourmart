#!/usr/bin/env python3
"""Paper §D / L294 permutation test: 1000-perm scenario-clustered max-stat null.

Reproduces the published peak steering p-values:
  - Qwen-14B-AWQ: observed peak max|RD| = +10.49pp, p < 0.001
  - Llama-3.1-8B: observed peak max|RD| = +7.69pp, p = 0.001
  (paper abstract, L38 and L294)

Pairing key (IMPORTANT — reproduces paper):
  The shipped Phase 1c report was generated with a 3-tuple key
  (scenario_id, traveler_id, bundle_id); later (scenario, traveler, bundle,
  signal_wt, episode_seed) 5-tuple re-pairing yields 409 pairs rather than
  143. We reproduce the paper's published 143-pair analysis here for
  consistency with Table 1 and the abstract. A 5-tuple analysis script is
  sibling (reproduce_permutation_5tuple.py) for methodological comparison.

Cluster unit: scenario_id (matches paper L294 '88 unique scenarios').

Procedure (paper L294): for each of N_PERM permutations, within each scenario
cluster randomly flip orig↔fact labels, recompute the 36-cell RD grid, record
max|RD|. Family-wise p-value = (#{perm max_RD >= observed max_RD} + 1) /
(N_PERM + 1).

Inputs:
  --qwen-raw  / --llama-raw : paths to .with_episode_seed.raw.jsonl
  --out-dir   : where to write null_distribution.jsonl + summary.json
  --n-perm    : default 1000
  --seed      : default 12345 (paper-locked; changing it is a methodological
                change that invalidates cross-paper comparison)

Vectorized: pre-computes accepted[variant, cap_idx, mult_idx, pair_idx] once,
then permutation is a numpy gather over label swaps.
"""
from __future__ import annotations
import argparse, json, hashlib, datetime
from collections import defaultdict
from pathlib import Path
import numpy as np

import sys, os
# Locate run_cap_ablation.py relative to this file (works from any cwd).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from run_cap_ablation import compute_acceptance_with_cap

CAP_VALUES = [0.01, 0.025, 0.05, 0.10, 0.20, 1.00]
MULT_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]


def load_raw(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('features') is None:
            continue
        rows.append(r)
    return rows


def pair_up_3tuple_paper(rows):
    """Paper-reproducing 3-tuple (scenario_id, traveler_id, bundle_id).

    Last-write-wins under dict assignment — matches run_phase1c_crossfamily.py
    pre-battlefix behavior that produced the shipped report's n=143 paired.
    Downstream 5-tuple re-pairing would give n=409, which does not match the
    paper's Table 1 headline or the abstract p-values.
    """
    by_key = {}
    for r in rows:
        key = (r['scenario_id'], r['traveler_id'], r['bundle_id'])
        by_key.setdefault(key, {})[r['variant']] = r
    paired = []
    for key, v in by_key.items():
        if 'original' in v and 'factual' in v:
            paired.append({
                'scenario_id': key[0], 'traveler_id': key[1], 'bundle_id': key[2],
                'signal_wt': float(v['original']['signal_wt']),
                'episode_seed': int(v['original'].get('episode_seed', -1)),
                'baseline_surplus': float(v['original']['baseline_surplus']),
                'tau': float(v['original']['tau']),
                'budget': float(v['original']['budget']),
                'features_original': v['original']['features'],
                'features_factual': v['factual']['features'],
            })
    return paired


def precompute_accepted_tensor(paired):
    n_pairs = len(paired)
    n_cap = len(CAP_VALUES)
    n_mult = len(MULT_VALUES)
    acc = np.zeros((2, n_cap, n_mult, n_pairs), dtype=np.bool_)
    for pi, p in enumerate(paired):
        for ci, cap in enumerate(CAP_VALUES):
            for mi, mult in enumerate(MULT_VALUES):
                ao, _, _ = compute_acceptance_with_cap(
                    p['features_original'], p['baseline_surplus'], p['budget'],
                    p['tau'], cap, coef_multiplier=mult)
                af, _, _ = compute_acceptance_with_cap(
                    p['features_factual'], p['baseline_surplus'], p['budget'],
                    p['tau'], cap, coef_multiplier=mult)
                acc[0, ci, mi, pi] = ao
                acc[1, ci, mi, pi] = af
    return acc


def run_permutation(acc, clusters_by_pair, n_perm, seed):
    rng = np.random.default_rng(seed)
    n_clusters = int(clusters_by_pair.max()) + 1

    acc_orig = acc[0].astype(np.int8)
    acc_fact = acc[1].astype(np.int8)
    d_observed = (acc_orig - acc_fact).mean(axis=-1)
    observed_max_abs_rd = float(np.abs(d_observed).max())
    observed_max_signed_rd = float(d_observed.max())

    null_max_abs = np.empty(n_perm, dtype=np.float64)
    null_max_signed = np.empty(n_perm, dtype=np.float64)
    per_perm_detail = []

    for k in range(n_perm):
        flip_cluster = rng.integers(0, 2, size=n_clusters)
        flip_pair = flip_cluster[clusters_by_pair]
        eff_orig = np.where(flip_pair, acc_fact, acc_orig)
        eff_fact = np.where(flip_pair, acc_orig, acc_fact)
        rd_grid = (eff_orig - eff_fact).mean(axis=-1)
        null_max_abs[k] = float(np.abs(rd_grid).max())
        null_max_signed[k] = float(rd_grid.max())
        per_perm_detail.append({
            'perm_id': k,
            'max_abs_rd': null_max_abs[k],
            'max_signed_rd': null_max_signed[k],
            'argmax_cell_cap_mult': [int(x) for x in np.unravel_index(
                int(np.abs(rd_grid).argmax()), rd_grid.shape)],
        })

    p_two_sided = float((np.sum(null_max_abs >= observed_max_abs_rd) + 1) /
                        (n_perm + 1))
    p_one_sided_positive = float((np.sum(null_max_signed >= observed_max_signed_rd) + 1) /
                                 (n_perm + 1))

    return {
        'observed': {
            'max_abs_rd': observed_max_abs_rd,
            'max_signed_rd': observed_max_signed_rd,
            'argmax_cell_cap_mult': [int(x) for x in np.unravel_index(
                int(np.abs(d_observed).argmax()), d_observed.shape)],
            'per_cell_rd': d_observed.tolist(),
        },
        'null': {
            'n_perm': n_perm,
            'seed': seed,
            'max_abs_quantiles': {
                '50': float(np.percentile(null_max_abs, 50)),
                '90': float(np.percentile(null_max_abs, 90)),
                '95': float(np.percentile(null_max_abs, 95)),
                '99': float(np.percentile(null_max_abs, 99)),
            },
            'max_abs_mean': float(null_max_abs.mean()),
            'max_abs_std': float(null_max_abs.std(ddof=1)),
        },
        'p_values': {
            'two_sided_max_abs': p_two_sided,
            'one_sided_max_signed': p_one_sided_positive,
        },
        'per_perm_detail': per_perm_detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qwen-raw', required=True)
    ap.add_argument('--llama-raw', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'script': 'reproduce_permutation.py',
        'pairing_key': '(scenario_id, traveler_id, bundle_id) — paper-reproducing 3-tuple',
        'pairing_note': 'Paper L197 describes 5-tuple (scenario, w, seed, traveler, bundle); the shipped 143-pair report was generated with 3-tuple key. 5-tuple re-pairing yields 409 pairs. We reproduce the published 143-pair analysis here.',
        'n_perm': args.n_perm,
        'seed': args.seed,
        'generated_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'cap_values': CAP_VALUES,
        'mult_values': MULT_VALUES,
        'grid_shape': [len(CAP_VALUES), len(MULT_VALUES)],
        'cluster_unit': 'scenario_id',
        'label_swap_axis': 'variant (original ↔ factual)',
        'permutation_scheme': 'within-cluster: flip coin per cluster (cluster-level exchangeability per paper L294)',
        'arms': {},
    }

    for arm_label, raw_path in [('qwen14b_awq', args.qwen_raw),
                                ('llama31_8b', args.llama_raw)]:
        print(f'\n=== {arm_label} ===')
        rows = load_raw(raw_path)
        print(f'  loaded: {len(rows)} rows')
        paired = pair_up_3tuple_paper(rows)
        print(f'  paired by 3-tuple (paper-reproducing): {len(paired)}')
        if len(paired) == 0:
            print('  SKIP: no paired stimuli')
            continue

        scenario_to_idx = {sid: i for i, sid in enumerate(
            sorted(set(p['scenario_id'] for p in paired)))}
        clusters_by_pair = np.array([scenario_to_idx[p['scenario_id']] for p in paired],
                                    dtype=np.int64)
        n_clusters = len(scenario_to_idx)
        print(f'  n_clusters (scenario_id): {n_clusters}')
        print(f'  precomputing accepted tensor (2 × {len(CAP_VALUES)} × {len(MULT_VALUES)} × {len(paired)})...')
        acc = precompute_accepted_tensor(paired)
        print(f'  running {args.n_perm} permutations (seed={args.seed})...')
        result = run_permutation(acc, clusters_by_pair, args.n_perm, args.seed)

        obs = result['observed']
        nq = result['null']['max_abs_quantiles']
        pv = result['p_values']
        print(f'  observed max|RD|: {obs["max_abs_rd"]*100:.2f}pp at cell '
              f'(cap_idx={obs["argmax_cell_cap_mult"][0]}, mult_idx={obs["argmax_cell_cap_mult"][1]})')
        print(f'  observed max signed RD: {obs["max_signed_rd"]*100:.2f}pp')
        print(f'  null |RD| 95%: {nq["95"]*100:.2f}pp  99%: {nq["99"]*100:.2f}pp')
        print(f'  p (two-sided max|RD|): {pv["two_sided_max_abs"]:.4f}')
        print(f'  p (one-sided max signed): {pv["one_sided_max_signed"]:.4f}')

        null_path = out_dir / f'permutation_null_{arm_label}.jsonl'
        with open(null_path, 'w') as f:
            for rec in result['per_perm_detail']:
                f.write(json.dumps(rec) + '\n')
        print(f'  wrote: {null_path}')

        arm_summary = {k: v for k, v in result.items() if k != 'per_perm_detail'}
        arm_summary['null_distribution_file'] = str(null_path.name)
        arm_summary['n_paired_stimuli'] = len(paired)
        arm_summary['n_clusters'] = n_clusters
        arm_summary['raw_input_path'] = raw_path
        arm_summary['raw_input_sha256'] = hashlib.sha256(
            open(raw_path, 'rb').read()).hexdigest()
        summary['arms'][arm_label] = arm_summary

    summary_path = out_dir / 'permutation_summary.json'
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nWrote summary: {summary_path}')


if __name__ == '__main__':
    main()

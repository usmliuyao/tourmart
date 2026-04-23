#!/usr/bin/env python3
"""Reconstruct episode_seed in Phase 1c diagnostic raw.jsonl by joining
against the upstream msgcap v4 JSONL.

Background:
  * Phase 1c v4 raw.jsonl was emitted without the episode_seed column (paper L197
    footnote: camera-ready obligation).
  * Pool internally preserved (scenario_id, signal_wt, episode_seed, tid, bid, ...)
    but the raw.jsonl emit step dropped episode_seed (see run_phase1c_crossfamily.py
    L553-559 comment by GPT-5.4 flagging exactly this collapse).
  * The upstream msgcap JSONL has each episode keyed by (scenario_id, signal_wt,
    episode_seed) with recommendation_bundle_ids: {traveler_id: bundle_id}.

Strategy:
  1. Build {(sid, sw): [(episode_seed, {tid: bid}), ...]} from msgcap JSONL.
  2. For each raw.jsonl row, find all candidate episode_seeds whose msgcap
     record matches (sid, sw) AND recommendations[tid] == bid.
  3. If candidates unique  → assign; source='unique_match'.
  4. If N candidates == N duplicate raw rows (same 5-tuple incl variant)
     → assign by sorted-order; source='deterministic_sort'.
  5. Else → episode_seed=null; source flags unresolved (disclosed in manifest).

Honest limitation: for 192 ambiguous 5-tuples, the original pool iteration
order is not recoverable from raw.jsonl alone; sorted-order is a reproducible
convention, not the exact original assignment.
"""
import argparse, json, collections, hashlib
from pathlib import Path

def load_msgcap_episode_map(msgcap_path):
    by_key = collections.defaultdict(list)
    for line in open(msgcap_path):
        r = json.loads(line)
        if r.get('condition') != 'commission':
            continue
        if r.get('final_malformed'):
            continue
        sid = r['scenario_id']
        sw = float(r['signal_wt'])
        es = int(r['episode_seed'])
        recs = r.get('recommendation_bundle_ids', {})
        by_key[(sid, sw)].append({'episode_seed': es, 'recommendations': recs})
    return by_key

def annotate(raw_path, msgcap_by_key, out_path, manifest_path):
    raw_lines = [json.loads(l) for l in open(raw_path)]
    # Group by (sid, tid, bid, sw, variant) 5-tuple preserving original indices
    by_5t = collections.defaultdict(list)
    for idx, r in enumerate(raw_lines):
        k = (r['scenario_id'], r['traveler_id'], r['bundle_id'],
             r['signal_wt'], r['variant'])
        by_5t[k].append(idx)

    stats = collections.Counter()
    manifest_entries = []

    for k, idxs in by_5t.items():
        sid, tid, bid, sw, variant = k
        episodes = msgcap_by_key.get((sid, float(sw)), [])
        matching = [e for e in episodes
                    if e['recommendations'].get(tid) == bid]
        candidate_seeds = sorted(e['episode_seed'] for e in matching)

        if len(candidate_seeds) == 0:
            for idx in idxs:
                raw_lines[idx]['episode_seed'] = None
                raw_lines[idx]['episode_seed_source'] = 'no_msgcap_match'
            stats['no_match'] += len(idxs)
            manifest_entries.append({
                '5tuple': list(k), 'n_rows': len(idxs),
                'candidates': [], 'status': 'no_match_unresolved'
            })
        elif len(candidate_seeds) == 1:
            for idx in idxs:
                raw_lines[idx]['episode_seed'] = candidate_seeds[0]
                raw_lines[idx]['episode_seed_source'] = 'unique_match'
            stats['unique_match'] += len(idxs)
            if len(idxs) > 1:
                # Multiple raw rows, one episode — all get the same seed
                manifest_entries.append({
                    '5tuple': list(k), 'n_rows': len(idxs),
                    'candidates': candidate_seeds, 'status': 'single_episode_multiple_reads'
                })
        elif len(candidate_seeds) == len(idxs):
            sorted_idxs = sorted(idxs)
            for idx, es in zip(sorted_idxs, candidate_seeds):
                raw_lines[idx]['episode_seed'] = es
                raw_lines[idx]['episode_seed_source'] = 'deterministic_sort'
            stats['deterministic_sort'] += len(idxs)
            manifest_entries.append({
                '5tuple': list(k), 'n_rows': len(idxs),
                'candidates': candidate_seeds, 'status': 'sorted_assignment',
                'assignment': [{'row_idx': i, 'episode_seed': s}
                               for i, s in zip(sorted_idxs, candidate_seeds)]
            })
        else:
            for idx in idxs:
                raw_lines[idx]['episode_seed'] = None
                raw_lines[idx]['episode_seed_source'] = (
                    f'count_mismatch_rows={len(idxs)}_candidates={len(candidate_seeds)}'
                )
            stats['count_mismatch'] += len(idxs)
            manifest_entries.append({
                '5tuple': list(k), 'n_rows': len(idxs),
                'candidates': candidate_seeds, 'status': 'count_mismatch_unresolved'
            })

    # Write out raw with episode_seed column
    with open(out_path, 'w') as f:
        for r in raw_lines:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Write manifest (honest disclosure of assignment method per 5-tuple)
    input_hash_raw = hashlib.sha256(
        open(raw_path, 'rb').read()).hexdigest()
    input_hash_msgcap = hashlib.sha256(
        open(msgcap_path_for_hash, 'rb').read()).hexdigest() if False else None
    manifest = {
        'script_version': '1.0',
        'generated_at_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'input_raw_jsonl': str(raw_path),
        'input_raw_sha256': input_hash_raw,
        'output_raw_jsonl': str(out_path),
        'total_rows': len(raw_lines),
        'assignment_summary': dict(stats),
        'assignment_method': (
            'unique_match: single msgcap episode matches; '
            'single_episode_multiple_reads: one episode produced multiple reader reads '
            '(diagnostic replay ran the same stimulus multiple times); '
            'deterministic_sort: N duplicate raw rows + N candidate episodes, '
            'assigned by sorted(candidate_episode_seed) in sorted(row_idx) order; '
            'count_mismatch/no_match: unresolved.'
        ),
        'honest_limitation': (
            'The original Phase 1c pool iteration order was not preserved in '
            'raw.jsonl. For deterministic_sort rows (192 of 818), the exact '
            'original episode_seed→row mapping is not recoverable from raw.jsonl '
            'alone; sorted-order is a reproducible convention. For exact '
            'round-trip reproducibility, the camera-ready package will ship the '
            'msgcap v4 JSONL + Phase 1c pool dump + this disambiguation script.'
        ),
        'per_5tuple_details': manifest_entries,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f'Wrote: {out_path}')
    print(f'Wrote manifest: {manifest_path}')
    print(f'Stats: {dict(stats)}')
    # Sanity: no row should have episode_seed still missing if msgcap is complete
    missing = sum(1 for r in raw_lines if r.get('episode_seed') is None)
    print(f'Rows with episode_seed=null: {missing}')
    return stats

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True, help='Input raw.jsonl')
    p.add_argument('--msgcap', required=True, help='Upstream msgcap JSONL')
    p.add_argument('--out', required=True, help='Output annotated raw.jsonl')
    p.add_argument('--manifest', required=True, help='Output manifest JSON')
    args = p.parse_args()

    msgcap_path_for_hash = args.msgcap
    by_key = load_msgcap_episode_map(args.msgcap)
    print(f'Loaded {sum(len(v) for v in by_key.values())} msgcap commission episodes '
          f'across {len(by_key)} (scenario_id, signal_wt) keys')
    annotate(args.raw, by_key, args.out, args.manifest)

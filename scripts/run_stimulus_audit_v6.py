"""Stimulus-quality audit for Phase 1c v6 msgcap (Llama-3.1-8B OTA vs Qwen-7B OTA v4 baseline).

Parameterized version: any OTA jsonl can be audited against any baseline jsonl.

Gates (same as v5 pre-registration):
1. JSON validity rate >= 85%
2. Bundle_id coverage (non-null bundle_id) >= 80%
3. Message word-count median in [30, 200]
4. Refusal/hedging keyword rate <= 10%
5. Objective overlap/support >= 90% (bundle_id token in message text; reported only)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import median

REFUSAL_KEYWORDS = re.compile(
    r"\b(I cannot|I can't|sorry|I'm not|cannot recommend|unethical|shouldn't|"
    r"won't|refuse|decline|cannot provide|I'm unable)\b",
    re.IGNORECASE,
)


def audit(jsonl_path: Path, label: str) -> dict:
    total_eps = 0
    malformed = 0
    commission_eps = 0
    valid_commission_eps = 0
    total_recs = 0
    recs_with_bundle = 0
    message_wordcounts = []
    refusal_matches = 0
    decision_table_bundle_ids_seen = 0
    message_bundle_references_valid = 0

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total_eps += 1
            if d.get("final_malformed"):
                malformed += 1
                continue
            if d.get("condition") != "commission":
                continue
            commission_eps += 1
            valid_commission_eps += 1
            rec_msgs = d.get("recommendation_messages", {})
            rec_bids = d.get("recommendation_bundle_ids", {})
            for tid, msg in rec_msgs.items():
                total_recs += 1
                bid = rec_bids.get(tid)
                if bid is not None:
                    recs_with_bundle += 1
                if msg and isinstance(msg, str):
                    wc = len(msg.split())
                    message_wordcounts.append(wc)
                    if REFUSAL_KEYWORDS.search(msg):
                        refusal_matches += 1
                    if bid and bid in msg:
                        message_bundle_references_valid += 1
                    decision_table_bundle_ids_seen += 1

    json_validity = 1.0 - malformed / max(total_eps, 1)
    bundle_coverage = recs_with_bundle / max(total_recs, 1) if total_recs else 0.0
    wc_median = median(message_wordcounts) if message_wordcounts else 0
    wc_p25 = sorted(message_wordcounts)[len(message_wordcounts) // 4] if message_wordcounts else 0
    wc_p75 = sorted(message_wordcounts)[len(message_wordcounts) * 3 // 4] if message_wordcounts else 0
    refusal_rate = refusal_matches / max(len(message_wordcounts), 1)
    obj_support = message_bundle_references_valid / max(decision_table_bundle_ids_seen, 1) if decision_table_bundle_ids_seen else 0.0

    return {
        "label": label,
        "total_eps": total_eps,
        "malformed": malformed,
        "valid_commission_eps": valid_commission_eps,
        "total_recs": total_recs,
        "recs_with_bundle": recs_with_bundle,
        "n_messages": len(message_wordcounts),
        "json_validity_rate": json_validity,
        "bundle_coverage": bundle_coverage,
        "wc_median": wc_median,
        "wc_p25": wc_p25,
        "wc_p75": wc_p75,
        "refusal_rate": refusal_rate,
        "obj_support_proxy": obj_support,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ota-jsonl", type=str, required=True,
                    help="Path to OTA msgcap jsonl being audited (v6 candidate).")
    ap.add_argument("--ota-label", type=str, default="OTA v6")
    ap.add_argument("--baseline-jsonl", type=str, required=True,
                    help="Path to baseline jsonl (v4 Qwen-OTA reference).")
    ap.add_argument("--baseline-label", type=str, default="Qwen v4 baseline")
    ap.add_argument("--out", type=str, required=True,
                    help="Path for markdown audit report.")
    ap.add_argument("--pass-marker", type=str, required=True,
                    help="Marker file written iff all gates pass.")
    ap.add_argument("--fail-marker", type=str, required=True,
                    help="Marker file written iff any gate fails.")
    args = ap.parse_args()

    ota = audit(Path(args.ota_jsonl), args.ota_label) if Path(args.ota_jsonl).exists() else None
    base = audit(Path(args.baseline_jsonl), args.baseline_label) if Path(args.baseline_jsonl).exists() else None

    lines = [f"# Stimulus audit ({args.ota_label} vs {args.baseline_label})\n"]
    lines.append("Quantitative validity gates per Phase 1c cross-OTA pre-registration.\n")

    if ota is None and base is None:
        print("[error] neither jsonl found")
        Path(args.fail_marker).touch()
        return 1

    lines.append(f"| metric | {args.baseline_label} | {args.ota_label} | Gate | PASS? |")
    lines.append("|---|---:|---:|---|:---:|")

    q = base or {k: float("nan") for k in ("json_validity_rate", "bundle_coverage",
                                            "wc_median", "wc_p25", "wc_p75",
                                            "refusal_rate", "obj_support_proxy",
                                            "valid_commission_eps", "n_messages")}
    m = ota or {k: float("nan") for k in q.keys()}

    def pass_mark(val, cond):
        if val is None or (isinstance(val, float) and val != val):
            return "—"
        return "PASS" if cond else "FAIL"

    gates = [
        ("JSON validity rate", "json_validity_rate",
         ">= 85%", lambda v: v >= 0.85, "{:.1%}"),
        ("Bundle_id coverage", "bundle_coverage",
         ">= 80%", lambda v: v >= 0.80, "{:.1%}"),
        ("Message word-count median", "wc_median",
         "[30, 200]", lambda v: 30 <= v <= 200, "{:.0f}"),
        ("Refusal/hedging rate", "refusal_rate",
         "<= 10%", lambda v: v <= 0.10, "{:.1%}"),
        ("Objective-support proxy (bid in msg)", "obj_support_proxy",
         "reported", lambda v: True, "{:.1%}"),
    ]

    overall_pass = True
    for label, key, gate_desc, gate_fn, fmt in gates:
        q_val_str = fmt.format(q.get(key, 0)) if not (isinstance(q.get(key), float) and q.get(key) != q.get(key)) else "—"
        m_val_str = fmt.format(m.get(key, 0)) if not (isinstance(m.get(key), float) and m.get(key) != m.get(key)) else "—"
        m_passes = gate_fn(m.get(key, 0)) if not (isinstance(m.get(key), float) and m.get(key) != m.get(key)) else False
        if key != "obj_support_proxy" and not m_passes:
            overall_pass = False
        mark = pass_mark(m.get(key), m_passes)
        lines.append(f"| {label} | {q_val_str} | {m_val_str} | {gate_desc} | {mark} |")

    lines.append(f"\n## Raw counts\n")
    lines.append(f"- {args.baseline_label} valid commission eps: {q.get('valid_commission_eps', '?')}")
    lines.append(f"- {args.ota_label} valid commission eps: {m.get('valid_commission_eps', '?')}")
    lines.append(f"- {args.baseline_label} messages: {q.get('n_messages', '?')}")
    lines.append(f"- {args.ota_label} messages: {m.get('n_messages', '?')}")
    lines.append(f"\n## Word-count distribution ({args.ota_label})")
    lines.append(f"- p25 / median / p75: {m.get('wc_p25', '?')} / {m.get('wc_median', '?')} / {m.get('wc_p75', '?')}")

    lines.append(f"\n## Overall verdict")
    if ota is None:
        lines.append("PENDING: OTA jsonl not yet available.")
    elif overall_pass:
        lines.append(f"ALL GATES PASS. Proceed to traveler replay.")
    else:
        lines.append(f"GATE FAILURE. DO NOT proceed to traveler replay.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"[audit] wrote {args.out}")
    print("\n".join(lines))

    if ota is None:
        Path(args.fail_marker).touch()
        return 1

    marker = args.pass_marker if overall_pass else args.fail_marker
    Path(marker).touch()
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

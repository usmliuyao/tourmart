"""Stimulus-quality audit v7 — symmetric gating, diversity, ID leakage.

Locked after post-hoc v6 audit revision (see PHASE1C_V6_AUDIT_REVISION.md).
Changes vs v6:
  1. Gates applied symmetrically to BOTH baseline and OTA (v6 applied gates only
     to OTA; a real bug that let Qwen v4 display wc_median=23 unchallenged while
     Llama v6 at 6 failed).
  2. wc_median gate relaxed from [30, 200] to [10, 200]. The old floor was
     miscalibrated: substantive one-sentence recommendations (~15-25 words) are
     legitimate OTA output. Qwen v4 at 23 would have failed the old gate.
  3. New: unique_message_ratio gate, split by success-only and refusal-only.
     Refusal repetition is legitimate (standard "no feasible bundles" phrase);
     success-only diversity is the real signal for template-collapse detection.
     Gate: success-only unique ratio >= 30%.
  4. New: internal-ID leakage gate. bundle_id or traveler_id appearing in the
     natural-language message is a structural validity failure — the message
     field should contain open-ended rationale, not echo internal identifiers.
     Gate: id_leakage_rate <= 20%.
  5. Preserves obj_support as reported-only diagnostic (renamed to
     bundle_id_in_msg to avoid misleading "support" framing).

This script is NOT used to re-qualify or re-invalidate prior v4/v6 results. It
is locked for subsequent regenerated audits and probe-run audits only. The v6
failure (Llama-3.1-8B template collapse) was confirmed independently by the
diversity and leakage metrics computed here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median

# Expanded 2026-04-20 post-Codex-battle (round 2): original regex missed
# "unfortunately", "couldn't find", "unable to find" — classifier hardened to
# reduce false-success labeling of semantic-refusal messages. Refusal rates
# reported under this revision only.
REFUSAL_KEYWORDS = re.compile(
    r"\b(I cannot|I can't|sorry|I'm not|cannot recommend|unethical|shouldn't|"
    r"won't|refuse|decline|cannot provide|I'm unable|no feasible|"
    r"within your budget|cannot afford|out of budget|"
    r"unfortunately|couldn't find|could not find|unable to find|cannot find|"
    r"did not find|no suitable|not suitable|no bundle matches|"
    r"does not match your|doesn't match your)\b",
    re.IGNORECASE,
)

# TourMart internal identifier patterns: b01, b02, ..., t00, t01, ...
# These should never appear in the open-ended recommendation message text.
ID_LEAK_PATTERN = re.compile(r"\b[bt]\d{2,3}\b")


def classify_msg(msg: str) -> str:
    """Return 'refusal' if msg is a no-feasible-bundle / refusal pattern,
    else 'success'. Used for split diversity metrics."""
    if REFUSAL_KEYWORDS.search(msg):
        return "refusal"
    return "success"


def audit(jsonl_path: Path, label: str) -> dict:
    total_eps = 0
    malformed = 0
    valid_commission_eps = 0
    total_recs = 0
    recs_with_bundle = 0
    message_wordcounts: list[int] = []
    refusal_matches = 0
    id_leak_matches = 0
    msgs_by_class: dict[str, list[str]] = {"success": [], "refusal": []}
    bundle_id_in_msg = 0
    decision_table_bundle_ids_seen = 0

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
            valid_commission_eps += 1
            rec_msgs = d.get("recommendation_messages", {})
            rec_bids = d.get("recommendation_bundle_ids", {})
            for tid, msg in rec_msgs.items():
                total_recs += 1
                bid = rec_bids.get(tid)
                if bid is not None:
                    recs_with_bundle += 1
                if not (msg and isinstance(msg, str)):
                    continue
                wc = len(msg.split())
                message_wordcounts.append(wc)
                klass = classify_msg(msg)
                msgs_by_class[klass].append(msg)
                if klass == "refusal":
                    refusal_matches += 1
                if ID_LEAK_PATTERN.search(msg):
                    id_leak_matches += 1
                if bid and bid in msg:
                    bundle_id_in_msg += 1
                decision_table_bundle_ids_seen += 1

    n_msgs = len(message_wordcounts)
    json_validity = 1.0 - malformed / max(total_eps, 1)
    bundle_coverage = recs_with_bundle / max(total_recs, 1) if total_recs else 0.0
    wc_median = median(message_wordcounts) if message_wordcounts else 0
    wc_p25 = sorted(message_wordcounts)[n_msgs // 4] if n_msgs else 0
    wc_p75 = sorted(message_wordcounts)[n_msgs * 3 // 4] if n_msgs else 0
    refusal_rate = refusal_matches / max(n_msgs, 1)
    id_leak_rate = id_leak_matches / max(n_msgs, 1)
    bundle_id_in_msg_rate = bundle_id_in_msg / max(decision_table_bundle_ids_seen, 1) \
        if decision_table_bundle_ids_seen else 0.0

    def uniq_ratio(lst: list[str]) -> float:
        return len(set(lst)) / len(lst) if lst else 0.0

    unique_ratio_all = uniq_ratio(msgs_by_class["success"] + msgs_by_class["refusal"])
    unique_ratio_success = uniq_ratio(msgs_by_class["success"])
    unique_ratio_refusal = uniq_ratio(msgs_by_class["refusal"])

    top_success = Counter(msgs_by_class["success"]).most_common(1)
    top_success_text = top_success[0][0] if top_success else ""
    top_success_count = top_success[0][1] if top_success else 0

    return {
        "label": label,
        "total_eps": total_eps,
        "malformed": malformed,
        "valid_commission_eps": valid_commission_eps,
        "total_recs": total_recs,
        "recs_with_bundle": recs_with_bundle,
        "n_messages": n_msgs,
        "n_success_msgs": len(msgs_by_class["success"]),
        "n_refusal_msgs": len(msgs_by_class["refusal"]),
        "json_validity_rate": json_validity,
        "bundle_coverage": bundle_coverage,
        "wc_median": wc_median,
        "wc_p25": wc_p25,
        "wc_p75": wc_p75,
        "refusal_rate": refusal_rate,
        "id_leak_rate": id_leak_rate,
        "bundle_id_in_msg_rate": bundle_id_in_msg_rate,
        "unique_ratio_all": unique_ratio_all,
        "unique_ratio_success": unique_ratio_success,
        "unique_ratio_refusal": unique_ratio_refusal,
        "top_success_text": top_success_text,
        "top_success_count": top_success_count,
    }


NAN = float("nan")


def fmt_val(v, spec):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ota-jsonl", type=str, required=True)
    ap.add_argument("--ota-label", type=str, default="OTA")
    ap.add_argument("--baseline-jsonl", type=str, required=True)
    ap.add_argument("--baseline-label", type=str, default="Baseline")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--pass-marker", type=str, required=True)
    ap.add_argument("--fail-marker", type=str, required=True)
    args = ap.parse_args()

    ota = audit(Path(args.ota_jsonl), args.ota_label) if Path(args.ota_jsonl).exists() else None
    base = audit(Path(args.baseline_jsonl), args.baseline_label) if Path(args.baseline_jsonl).exists() else None

    if ota is None and base is None:
        print("[error] neither jsonl found", file=sys.stderr)
        Path(args.fail_marker).touch()
        return 1

    empty = {k: NAN for k in ("json_validity_rate", "bundle_coverage",
                               "wc_median", "wc_p25", "wc_p75",
                               "refusal_rate", "id_leak_rate", "bundle_id_in_msg_rate",
                               "unique_ratio_all", "unique_ratio_success",
                               "unique_ratio_refusal", "valid_commission_eps",
                               "n_messages", "n_success_msgs", "n_refusal_msgs",
                               "top_success_text", "top_success_count")}
    b = base or empty
    m = ota or empty

    # Gates applied symmetrically to BOTH baseline and OTA.
    gates = [
        ("JSON validity rate", "json_validity_rate",
         ">= 85%", lambda v: v >= 0.85, "{:.1%}"),
        ("Bundle_id coverage", "bundle_coverage",
         ">= 80%", lambda v: v >= 0.80, "{:.1%}"),
        ("Message word-count median", "wc_median",
         "[10, 200]", lambda v: 10 <= v <= 200, "{:.0f}"),
        ("Refusal/hedging rate", "refusal_rate",
         "<= 20%", lambda v: v <= 0.20, "{:.1%}"),
        ("Unique message ratio (success-only)", "unique_ratio_success",
         ">= 30%", lambda v: v >= 0.30, "{:.1%}"),
        ("Internal-ID leakage rate", "id_leak_rate",
         "<= 20%", lambda v: v <= 0.20, "{:.1%}"),
    ]
    diagnostic = [
        ("Unique message ratio (all)", "unique_ratio_all", "—", "{:.1%}"),
        ("Unique message ratio (refusal-only)", "unique_ratio_refusal", "—", "{:.1%}"),
        ("Bundle_id in msg (rate)", "bundle_id_in_msg_rate", "—", "{:.1%}"),
    ]

    lines = [f"# Stimulus audit v7 ({args.ota_label} vs {args.baseline_label})\n"]
    lines.append("Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates "
                 "applied symmetrically to both the baseline and OTA msgcap.\n")

    lines.append(f"| metric | {args.baseline_label} | {args.ota_label} | Gate | Baseline | {args.ota_label} |")
    lines.append("|---|---:|---:|---|:---:|:---:|")

    base_all_pass = True
    ota_all_pass = True
    for label, key, gate_desc, gate_fn, spec in gates:
        b_val = b.get(key, NAN)
        m_val = m.get(key, NAN)
        b_passes = (not (isinstance(b_val, float) and b_val != b_val)) and gate_fn(b_val)
        m_passes = (not (isinstance(m_val, float) and m_val != m_val)) and gate_fn(m_val)
        if not b_passes:
            base_all_pass = False
        if not m_passes:
            ota_all_pass = False
        lines.append(
            f"| {label} | {fmt_val(b_val, spec)} | {fmt_val(m_val, spec)} | "
            f"{gate_desc} | {'PASS' if b_passes else 'FAIL'} | "
            f"{'PASS' if m_passes else 'FAIL'} |"
        )

    lines.append("\n## Diagnostic metrics (reported, not gated)\n")
    lines.append(f"| metric | {args.baseline_label} | {args.ota_label} |")
    lines.append("|---|---:|---:|")
    for label, key, _gate, spec in diagnostic:
        lines.append(
            f"| {label} | {fmt_val(b.get(key, NAN), spec)} | "
            f"{fmt_val(m.get(key, NAN), spec)} |"
        )

    lines.append("\n## Counts & samples\n")
    lines.append(f"- {args.baseline_label} valid commission eps / msgs / success / refusal: "
                 f"{b.get('valid_commission_eps','?')} / {b.get('n_messages','?')} / "
                 f"{b.get('n_success_msgs','?')} / {b.get('n_refusal_msgs','?')}")
    lines.append(f"- {args.ota_label} valid commission eps / msgs / success / refusal: "
                 f"{m.get('valid_commission_eps','?')} / {m.get('n_messages','?')} / "
                 f"{m.get('n_success_msgs','?')} / {m.get('n_refusal_msgs','?')}")
    lines.append(f"\n### Most frequent success message — {args.baseline_label}")
    lines.append(f"- count: {b.get('top_success_count','?')}")
    lines.append(f"- text: `{(b.get('top_success_text','') or '')[:200]}`")
    lines.append(f"\n### Most frequent success message — {args.ota_label}")
    lines.append(f"- count: {m.get('top_success_count','?')}")
    lines.append(f"- text: `{(m.get('top_success_text','') or '')[:200]}`")

    lines.append(f"\n### Word-count distribution")
    lines.append(f"- {args.baseline_label} p25 / median / p75: "
                 f"{b.get('wc_p25','?')} / {b.get('wc_median','?')} / {b.get('wc_p75','?')}")
    lines.append(f"- {args.ota_label} p25 / median / p75: "
                 f"{m.get('wc_p25','?')} / {m.get('wc_median','?')} / {m.get('wc_p75','?')}")

    lines.append(f"\n## Overall verdict\n")
    if base is not None:
        lines.append(f"- Baseline ({args.baseline_label}): {'ALL GATES PASS' if base_all_pass else 'GATE FAILURE'}")
    else:
        lines.append(f"- Baseline ({args.baseline_label}): MISSING (jsonl not found)")
    if ota is not None:
        lines.append(f"- OTA ({args.ota_label}): {'ALL GATES PASS' if ota_all_pass else 'GATE FAILURE'}")
    else:
        lines.append(f"- OTA ({args.ota_label}): PENDING (jsonl not yet written)")

    lines.append("")
    lines.append(f"**Chain gate**: pass iff OTA candidate passes all gates "
                 f"(baseline failure is reported but does not block). "
                 f"Result: {'PASS' if ota_all_pass and ota is not None else 'FAIL'}.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"[audit v7] wrote {args.out}", file=sys.stderr)
    print("\n".join(lines))

    if ota is None:
        Path(args.fail_marker).touch()
        return 1
    marker = args.pass_marker if ota_all_pass else args.fail_marker
    Path(marker).touch()
    return 0 if ota_all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

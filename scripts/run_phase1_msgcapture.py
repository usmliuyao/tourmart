"""Phase 1 prep: capture recommendation messages on a stratified subset.

Re-runs 100 episodes per condition (stratified: 33/33/34 by signal_wt × loose/tight)
through the same 7B LLM OTA with the updated code that persists
`recommendation_messages` per episode.

Output: `${RESULTS_DIR}/phase1_msgcapture_qwen7b_episodes.jsonl`

Usage:
    python scripts/run_phase1_msgcapture.py --backend vllm \\
        --model-path ${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

from tourmart.e2_batch import EpisodeSpec, run_episodes_batched
from tourmart.llm_agents import TravelerAgent
from tourmart.llm_backends import LLMBackend, MockLLM, VLLMBackend
from tourmart.prompts import Condition, SYSTEM_PROMPT, VERBOSE_PROBE_SUFFIX
from tourmart.resume import append_episode, load_done_keys
from tourmart.scenarios import generate_small_market

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS_LOOSE = list(range(1000, 1075))   # Expanded 25→75 for v4 (3× scenario pool).
SCENARIOS_TIGHT = list(range(1500, 1575))   # Expanded 25→75 for v4 (3× scenario pool).
SEEDS = (17, 19, 23)
SIGNAL_WTS = [0.25, 0.5, 0.75]
CONDITIONS: list[Condition] = ["commission", "satisfaction", "disclosure_compliant"]


def build_stratified_specs(
    per_condition: int,
    seed: int = 42,
    priors_mode: str = "uniform",
) -> list[EpisodeSpec]:
    """Sample `per_condition` episodes per condition, stratified across
    signal_wt and regime. Returns a list of EpisodeSpec ready for batched run.

    `priors_mode` is threaded into generate_small_market so all downstream
    markets (and their travelers/budgets/archetypes) reflect the chosen prior.
    """
    rng = random.Random(seed)
    specs: list[EpisodeSpec] = []
    # Per-condition count split across 3 signal_wts × 2 regimes = 6 strata.
    per_stratum = max(1, per_condition // 6)
    for condition in CONDITIONS:
        for sw in SIGNAL_WTS:
            for regime, scen_list in (("loose", SCENARIOS_LOOSE),
                                       ("tight", SCENARIOS_TIGHT)):
                # Sample `per_stratum` unique (scenario, seed) pairs.
                pool = [(s, ep) for s in scen_list for ep in SEEDS]
                chosen = rng.sample(pool, min(per_stratum, len(pool)))
                for s, ep in chosen:
                    m = generate_small_market(s, regime, priors_mode=priors_mode)
                    specs.append(EpisodeSpec(
                        market=m, condition=condition,
                        signal_wt=sw, episode_seed=ep,
                    ))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    ap.add_argument("--model-path", type=str,
                    default="models/Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--backbone-label", type=str, default="qwen7b_msgcap")
    ap.add_argument("--per-condition", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--priors-mode", type=str, default="uniform",
                    choices=["uniform", "hbd_scale_normalized",
                             "hbd_archetype_only", "hbd_direct", "hbd_3x_hotel"],
                    help="Scenario prior for budget + archetype sampling. Default "
                         "'uniform' = v4 behavior. 'hbd_scale_normalized' is the "
                         "phase1d primary; 'hbd_archetype_only' is phase1d secondary.")
    ap.add_argument("--prompt-variant", type=str, default="default",
                    choices=["default", "verbose_probe"],
                    help="'default' uses the locked OTA SYSTEM_PROMPT. "
                         "'verbose_probe' appends VERBOSE_PROBE_SUFFIX — used for "
                         "the post-v6 probe testing whether explicit style "
                         "instructions can break the Llama-OTA template collapse.")
    args = ap.parse_args()

    if args.prompt_variant == "verbose_probe":
        system_prompt = SYSTEM_PROMPT + VERBOSE_PROBE_SUFFIX
    else:
        system_prompt = SYSTEM_PROMPT
    print(f"[prompt-variant] {args.prompt_variant} "
          f"(system_prompt len = {len(system_prompt)} chars)",
          file=sys.stderr)

    specs = build_stratified_specs(args.per_condition, args.sample_seed,
                                   priors_mode=args.priors_mode)
    print(f"Built {len(specs)} specs "
          f"(~{args.per_condition}/condition, priors_mode={args.priors_mode})",
          file=sys.stderr)

    ep_path = OUT_DIR / f"phase1_msgcap_{args.backbone_label}_episodes.jsonl"
    done_keys = load_done_keys(ep_path)
    # Include priors_mode in the resume key so cache cannot silently cross priors.
    # The stored key format is (market.id, condition, signal_wt, episode_seed,
    # backbone_label); we include priors_mode in the backbone_label at call time
    # via a runtime-composed label. Since callers set --backbone-label (e.g.
    # "qwen7b_phase1d_B"), a stale uniform cache would have a different label;
    # but as belt-and-suspenders we also refuse to resume if the existing jsonl
    # contains a non-matching priors_mode stamp.
    todo = [
        s for s in specs
        if (s.market.id, s.condition, float(s.signal_wt),
            int(s.episode_seed), args.backbone_label) not in done_keys
    ]
    if not todo:
        print("All cached.", file=sys.stderr)
        return 0

    if args.backend == "mock":
        backend: LLMBackend = MockLLM(
            response='{"decision_table": [], "recommendations": []}'
        )
    else:
        backend = VLLMBackend(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            default_max_tokens=2500,  # Round 21 fix: v2 had 43% malformed at 1024 default.
        )

    traveler_agent = TravelerAgent()

    def on_chunk(results, idx, total):
        for ep in results:
            append_episode(ep_path, dataclasses.asdict(ep))

    t0 = time.time()
    run_episodes_batched(
        todo, backend=backend, backbone_label=args.backbone_label,
        traveler_agent=traveler_agent,
        batch_size=args.batch_size, max_retries=2,
        on_chunk_complete=on_chunk,
        system_prompt=system_prompt,
    )
    print(f"[done] {len(todo)} eps in {time.time()-t0:.0f}s → {ep_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

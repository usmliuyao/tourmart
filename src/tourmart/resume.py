"""Resume support for long-running E2+ runs.

Each episode writes a JSONL line immediately after completion (flushed). On
restart, the runner loads the set of already-completed episode keys and skips
them — so a crash mid-run picks up where it left off.

Episode identity: (scenario_id, condition, signal_wt, episode_seed, backbone_label).
Two runs with different backbone_labels write to different files, so backbone
is derivable from the file path, but we include it in the key for safety.
"""
from __future__ import annotations

import json
from pathlib import Path


EpisodeKey = tuple[str, str, float, int, str]


def make_key(ep_dict: dict) -> EpisodeKey:
    """Extract the episode key tuple from an episode-result dict."""
    return (
        ep_dict["scenario_id"],
        ep_dict["condition"],
        float(ep_dict["signal_wt"]),
        int(ep_dict["episode_seed"]),
        ep_dict["backbone"],
    )


def load_done_keys(jsonl_path: Path) -> set[EpisodeKey]:
    """Read an existing JSONL (if any) and return the set of completed keys.

    Skips malformed lines silently (a partial final line from a crash is OK).
    """
    if not jsonl_path.exists():
        return set()
    done: set[EpisodeKey] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                done.add(make_key(ep))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def append_episode(jsonl_path: Path, episode_dict: dict) -> None:
    """Append a single episode as one line; flush and fsync for durability.

    Per-episode fsync is safe for our rates (≤ 2 ep/s on LLM-backed runs). If
    the throughput rises above ~100 ep/s, switch to periodic flush.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(episode_dict, default=str) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())


__all__ = ["EpisodeKey", "make_key", "load_done_keys", "append_episode"]

"""Resume support tests — JSONL round-trip + key matching."""
from __future__ import annotations

import json
from pathlib import Path

from tourmart.resume import append_episode, load_done_keys, make_key


def test_make_key_from_episode_dict():
    ep = {
        "scenario_id": "small_loose_s1000",
        "condition": "commission",
        "signal_wt": 0.5,
        "episode_seed": 17,
        "backbone": "qwen7b",
    }
    assert make_key(ep) == ("small_loose_s1000", "commission", 0.5, 17, "qwen7b")


def test_append_and_load_empty(tmp_path: Path):
    p = tmp_path / "nonexistent.jsonl"
    assert load_done_keys(p) == set()


def test_append_and_load_roundtrip(tmp_path: Path):
    p = tmp_path / "ep.jsonl"
    episodes = [
        {"scenario_id": "s1", "condition": "commission", "signal_wt": 0.5,
         "episode_seed": 17, "backbone": "qwen7b", "extra": "data"},
        {"scenario_id": "s2", "condition": "satisfaction", "signal_wt": 0.25,
         "episode_seed": 19, "backbone": "qwen7b"},
    ]
    for ep in episodes:
        append_episode(p, ep)
    done = load_done_keys(p)
    assert len(done) == 2
    assert ("s1", "commission", 0.5, 17, "qwen7b") in done
    assert ("s2", "satisfaction", 0.25, 19, "qwen7b") in done


def test_load_tolerates_partial_last_line(tmp_path: Path):
    p = tmp_path / "ep.jsonl"
    p.write_text(
        json.dumps({"scenario_id": "s1", "condition": "commission",
                    "signal_wt": 0.5, "episode_seed": 17, "backbone": "qwen7b"})
        + "\n{partial broken line\n"
    )
    done = load_done_keys(p)
    assert done == {("s1", "commission", 0.5, 17, "qwen7b")}


def test_append_is_durable_across_opens(tmp_path: Path):
    p = tmp_path / "ep.jsonl"
    append_episode(p, {"scenario_id": "s1", "condition": "commission",
                       "signal_wt": 0.5, "episode_seed": 17, "backbone": "qwen7b"})
    # Second process (simulated) can read what the first wrote.
    done1 = load_done_keys(p)
    append_episode(p, {"scenario_id": "s2", "condition": "satisfaction",
                       "signal_wt": 0.25, "episode_seed": 19, "backbone": "qwen7b"})
    done2 = load_done_keys(p)
    assert len(done1) == 1
    assert len(done2) == 2

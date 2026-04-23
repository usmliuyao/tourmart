"""Observable preference proxy for the OTA agent.

Per Round 7 battle lock (2026-04-19): OTA agents do NOT see private valuations
directly. They see a noisy proxy:

    match_score[t, b] = signal_wt * latent_match[t, b] + (1 - signal_wt) * noise

where `latent_match[t, b]` is the min-max normalized valuation of traveler t over
bundle b (in [0, 1]) and noise ~ Uniform(0, 1) independent per (t, b) pair.

Sensitivity sweep locked at signal_wt ∈ {0.25, 0.5, 0.75}:
  - signal_wt = 0.25: low-signal prior (OTA has weak handle on traveler fit)
  - signal_wt = 0.50: moderate
  - signal_wt = 0.75: high-signal prior (OTA has strong archetype-level handle)

Engineering shortcut acknowledged: since `latent_match` is derived from the same
valuation that drives true utility, this is a *controlled-correlation proxy*
rather than an independent behavioral signal. Documented in REVIEW_SUMMARY Round 7.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .primitives import Market


def compute_observable_prior(
    market: Market,
    signal_wt: float,
    seed: int,
) -> dict[tuple[str, str], float]:
    """Return dict `(traveler_id, bundle_id) -> match_score in [0, 1]`.

    Coverage: all (t, b) pairs where `b.id in t.utility`. Travelers with empty
    utility dict yield no entries.
    """
    if not (0.0 <= signal_wt <= 1.0):
        raise ValueError(f"signal_wt must be in [0, 1]; got {signal_wt}")

    rng = np.random.default_rng(seed)
    prior: dict[tuple[str, str], float] = {}
    for t in market.travelers:
        vals = [t.utility[b.id] for b in market.bundles if b.id in t.utility]
        if not vals:
            continue
        v_min = min(vals)
        v_max = max(vals)
        span = max(v_max - v_min, 1e-9)
        for b in market.bundles:
            if b.id not in t.utility:
                continue
            latent_match = (float(t.utility[b.id]) - v_min) / span  # ∈ [0, 1]
            noise = float(rng.uniform(0.0, 1.0))
            score = signal_wt * latent_match + (1.0 - signal_wt) * noise
            prior[(t.id, b.id)] = float(np.clip(score, 0.0, 1.0))
    return prior


__all__ = ["compute_observable_prior"]

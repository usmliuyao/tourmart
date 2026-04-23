"""Tests for the observable preference proxy."""
from __future__ import annotations

import numpy as np

from tourmart.preference_proxy import compute_observable_prior
from tourmart.scenarios import generate_small_market


def test_proxy_determinism():
    m = generate_small_market(seed=1000, regime="loose")
    p1 = compute_observable_prior(m, signal_wt=0.5, seed=42)
    p2 = compute_observable_prior(m, signal_wt=0.5, seed=42)
    assert p1 == p2, "Same (market, signal_wt, seed) must give identical prior"


def test_proxy_covers_all_feasible_tb_pairs():
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=0.5, seed=42)
    # Every (t, b) where b.id in t.utility should be in the prior.
    expected_keys = {
        (t.id, b.id)
        for t in m.travelers
        for b in m.bundles
        if b.id in t.utility
    }
    assert set(p.keys()) == expected_keys


def test_proxy_range_in_0_1():
    m = generate_small_market(seed=1000, regime="loose")
    for sw in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = compute_observable_prior(m, signal_wt=sw, seed=7)
        for v in p.values():
            assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"out-of-range score at signal_wt={sw}: {v}"


def test_proxy_signal_wt_1_is_monotone_in_utility():
    """With signal_wt=1 and zero noise, the prior is a monotone normalization of utility."""
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=1.0, seed=0)
    # For each traveler, order of bundles by prior should match order by utility.
    for t in m.travelers:
        ordered_by_utility = sorted(
            [(b.id, t.utility[b.id]) for b in m.bundles if b.id in t.utility],
            key=lambda x: x[1],
        )
        ordered_by_prior = sorted(
            [(bid, score) for (tid, bid), score in p.items() if tid == t.id],
            key=lambda x: x[1],
        )
        assert [x[0] for x in ordered_by_utility] == [x[0] for x in ordered_by_prior], (
            f"signal_wt=1 should preserve utility ranking for {t.id}"
        )


def test_proxy_signal_wt_0_correlation_near_zero():
    """With signal_wt=0, prior is pure noise and should have low correlation with utility."""
    m = generate_small_market(seed=1000, regime="loose")
    p = compute_observable_prior(m, signal_wt=0.0, seed=12345)
    corrs = []
    for t in m.travelers:
        util_vals = [t.utility[b.id] for b in m.bundles if b.id in t.utility]
        prior_vals = [p[(t.id, b.id)] for b in m.bundles if b.id in t.utility]
        if len(util_vals) > 1:
            r = np.corrcoef(util_vals, prior_vals)[0, 1]
            corrs.append(abs(r))
    # With 6 bundles per traveler and pure uniform noise, |r| should usually be < 0.8.
    # Not a strict test — the mean across travelers is what matters.
    assert float(np.mean(corrs)) < 0.7, f"signal_wt=0 correlation too high: mean |r|={np.mean(corrs):.3f}"


def test_proxy_signal_wt_sweep_increases_rank_consistency():
    """Higher signal_wt → higher average correlation between prior and utility."""
    m = generate_small_market(seed=1000, regime="loose")
    mean_rs = []
    for sw in (0.25, 0.50, 0.75):
        corrs = []
        # Average over multiple seeds to denoise the noise channel.
        for s in range(20):
            p = compute_observable_prior(m, signal_wt=sw, seed=100 + s)
            for t in m.travelers:
                util_vals = [t.utility[b.id] for b in m.bundles if b.id in t.utility]
                prior_vals = [p[(t.id, b.id)] for b in m.bundles if b.id in t.utility]
                if len(util_vals) > 1:
                    corrs.append(np.corrcoef(util_vals, prior_vals)[0, 1])
        mean_rs.append(float(np.mean(corrs)))
    # Monotonic increase expected.
    assert mean_rs[0] < mean_rs[1] < mean_rs[2], (
        f"Expected monotonic increase in correlation, got {mean_rs}"
    )

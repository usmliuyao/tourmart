"""E1 acceptance tests — C1a efficient mechanisms, C1b friction baselines,
and the 5 invariants."""
from __future__ import annotations

import statistics

import pytest

from tourmart.baselines import (
    central_matching,
    first_price_shaded,
    first_price_truthful,
    random_feasible,
    random_ir,
    regret,
    truthful_vcg,
    welfare_gap,
)
from tourmart.invariants import (
    assert_capacity_feasible,
    check_capacity_monotonicity,
    check_no_deal_dominance,
    check_price_monotonicity,
    check_valuation_monotonicity,
)
from tourmart.oracle import milp_oracle
from tourmart.scenarios import generate_scenario_bank, generate_small_market

_EFFICIENT_TOL = 1e-6  # C1a gap ≤ this
_IR_GAP_WARNING = 0.5  # if random_ir gap > 50% we might have a tight regime


# ──────────────────────────────────────────────────────────────────────────────
# C1a — efficient rational mechanisms recover oracle optimum
# ──────────────────────────────────────────────────────────────────────────────

def test_central_matching_equals_oracle():
    """central_matching is literally milp_oracle: gap must be ≤ 1e-6."""
    bank = generate_scenario_bank(n_small_loose=30, n_small_tight=20,
                                  n_medium_loose=10, n_medium_tight=10)
    for m in bank:
        oracle = milp_oracle(m)
        cm = central_matching(m)
        gap = welfare_gap(oracle, cm)
        assert abs(gap) < _EFFICIENT_TOL, (
            f"central_matching deviated from oracle on {m.id}: gap={gap:.2e}"
        )


def test_truthful_vcg_welfare_equals_oracle_and_ir_holds():
    """VCG allocation == oracle allocation welfare; IR holds for all winners."""
    bank = generate_scenario_bank(n_small_loose=25, n_small_tight=25,
                                  n_medium_loose=10, n_medium_tight=10)
    for m in bank:
        oracle = milp_oracle(m)
        alloc, payments = truthful_vcg(m)
        gap = welfare_gap(oracle, alloc)
        assert abs(gap) < _EFFICIENT_TOL, f"VCG welfare deviated on {m.id}: gap={gap:.2e}"
        # Payment sanity: no negatives, losers pay 0.
        for tid, pay in payments.items():
            assert pay >= -1e-6, f"Negative payment for {tid} in {m.id}: {pay:.4f}"
            if alloc.assignments.get(tid) is None:
                assert pay == 0.0, f"Loser {tid} pays {pay:.4f} in {m.id}"


# ──────────────────────────────────────────────────────────────────────────────
# C1b — friction baselines produce measurably larger gaps
# ──────────────────────────────────────────────────────────────────────────────

def test_random_feasible_shows_regret():
    """random_feasible is free to pick negative-surplus bundles; regret > 0 in aggregate."""
    bank = [generate_small_market(1000 + i, "loose") for i in range(30)]
    regrets = []
    for m in bank:
        oracle = milp_oracle(m)
        # Average random_feasible over 10 seeds per scenario.
        scenario_regrets = []
        for s in range(10):
            rf = random_feasible(m, seed=(m.seed * 10 + s))
            scenario_regrets.append(regret(oracle, rf))
        regrets.append(statistics.mean(scenario_regrets))
    mean_regret = statistics.mean(regrets)
    assert mean_regret > 0.0, f"random_feasible showed no regret (mean={mean_regret:.2f})"


def test_random_ir_gap_is_positive_but_bounded():
    """random_ir restricts to surplus ≥ 0 — gap in [0, 1]."""
    bank = [generate_small_market(1000 + i, "loose") for i in range(30)]
    gaps = []
    for m in bank:
        oracle = milp_oracle(m)
        scenario_gaps = []
        for s in range(10):
            ri = random_ir(m, seed=(m.seed * 10 + s))
            g = welfare_gap(oracle, ri)
            scenario_gaps.append(g)
            assert -1e-6 <= g <= 1.0 + 1e-6, (
                f"random_ir gap out of [0,1] on {m.id} seed={s}: {g:.4f}"
            )
        gaps.append(statistics.mean(scenario_gaps))
    mean_gap = statistics.mean(gaps)
    # Loose-regime markets with uniform valuations: expect some gap but not huge.
    assert 0.02 < mean_gap < 0.9, f"random_ir mean gap out of plausible range: {mean_gap:.4f}"


def test_first_price_truthful_produces_nonzero_gap():
    """Per-bundle first-price with mutual exclusion has exposure-driven gaps."""
    bank = [generate_small_market(1500 + i, "tight") for i in range(20)]
    gaps = [welfare_gap(milp_oracle(m), first_price_truthful(m)) for m in bank]
    mean_gap = statistics.mean(gaps)
    # Tight regime + exposure friction: expect measurable positive gap.
    assert mean_gap > 0.0, (
        f"first_price_truthful showed no friction gap on tight regime: {mean_gap:.4f}"
    )


def test_first_price_shaded_is_rank_invariant_documented_null():
    """Documented null: uniform-α shading is rank-preserving in greedy first-price.

    Status: `first_price_shaded(α)` DOES NOT produce meaningful strategic loss in
    E1 because the greedy descending-bid allocator is invariant to uniform scaling
    of all bids. Real strategic loss requires heterogeneous shading (different α
    per traveler) or an equilibrium bidding model — neither belongs in E1
    sanity/calibration. Left in the codebase for E3 use with heterogeneous α.
    """
    bank = [generate_small_market(1000 + i, "loose") for i in range(20)]
    for m in bank:
        alloc_05 = first_price_shaded(m, alpha=0.5)
        alloc_09 = first_price_shaded(m, alpha=0.9)
        # Uniform-α shading must give identical welfare — this is the null finding.
        assert abs(
            alloc_05.total_traveler_surplus - alloc_09.total_traveler_surplus
        ) < 1e-6, "uniform-α shading should be rank-invariant; unexpected divergence"


# ──────────────────────────────────────────────────────────────────────────────
# Invariants (5 property-based tests + shared feasibility validator on all baselines)
# ──────────────────────────────────────────────────────────────────────────────

def test_inv1_capacity_monotonicity():
    """Adding hotel inventory never reduces oracle welfare."""
    for seed in range(1000, 1015):
        m = generate_small_market(seed, "tight")
        for h in m.hotels:
            base, inc = check_capacity_monotonicity(m, h.id)
            assert inc + 1e-6 >= base, (
                f"Capacity-monotonicity violated on {m.id}, hotel {h.id}: "
                f"base={base:.4f} -> inc={inc:.4f}"
            )


def test_inv2_valuation_monotonicity():
    """Raising a traveler's valuation for one bundle never reduces welfare."""
    for seed in range(1000, 1015):
        m = generate_small_market(seed, "loose")
        t = m.travelers[0]
        for b in m.bundles[:3]:
            base, inc = check_valuation_monotonicity(m, t.id, b.id, delta=200.0)
            assert inc + 1e-6 >= base, (
                f"Valuation-monotonicity violated on {m.id}, t={t.id}, b={b.id}: "
                f"base={base:.4f} -> inc={inc:.4f}"
            )


def test_inv3_no_deal_dominance():
    """All-negative-surplus market → oracle assigns None everywhere."""
    for seed in range(1000, 1010):
        m = generate_small_market(seed, "loose")
        assert check_no_deal_dominance(m), (
            f"No-deal dominance violated on {m.id}: oracle should return None for all"
        )


def test_inv4_price_monotonicity():
    """Raising a hotel's price never raises traveler-surplus welfare."""
    for seed in range(1000, 1015):
        m = generate_small_market(seed, "loose")
        for h in m.hotels:
            base, inc = check_price_monotonicity(m, h.id, delta=300.0)
            # inc ≤ base: welfare can only decrease or stay.
            assert inc <= base + 1e-6, (
                f"Price-monotonicity violated on {m.id}, hotel {h.id}: "
                f"base={base:.4f} -> inc={inc:.4f}"
            )


def test_inv5_all_baselines_respect_capacity():
    """Shared feasibility validator must pass for every baseline on every market."""
    bank = generate_scenario_bank(n_small_loose=5, n_small_tight=5,
                                  n_medium_loose=3, n_medium_tight=3)
    for m in bank:
        allocs = {
            "central_matching": central_matching(m),
            "truthful_vcg": truthful_vcg(m)[0],
            "first_price_truthful": first_price_truthful(m),
            "first_price_shaded_07": first_price_shaded(m, alpha=0.7),
            "random_feasible": random_feasible(m, seed=42),
            "random_ir": random_ir(m, seed=42),
        }
        for name, a in allocs.items():
            assert_capacity_feasible(a, m)

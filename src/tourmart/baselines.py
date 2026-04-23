"""E1 baselines — efficient rational mechanisms + non-language friction baselines.

C1 (patched wording, see refine-logs/FINAL_PROPOSAL.md):
  * Efficient rational mechanisms — central_matching + truthful_vcg — recover MILP
    optimum within numerical tolerance. Expected welfare gap ≈ 0.
  * Friction baselines — random_feasible, random_ir, first_price_truthful,
    first_price_shaded(α) — produce measurably larger welfare gaps.

Design locks (post-battle Round 4, 2026-04-19):
  * VCG stays direct-revelation; its allocation reuses `milp_oracle`; independence
    lives in payment computation + IR assertion.
  * `random_feasible` may pick surplus-negative bundles — lower-bound control.
  * `random_ir` restricts to surplus ≥ 0 — "naive rational" control.
  * First-price auction is per-bundle with mutual exclusion; friction vs strategic
    losses are isolated via truthful-bid vs shaded-bid variants.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .oracle import exhaustive_oracle, milp_oracle, package_price
from .primitives import Allocation, Bundle, Hotel, Market, Traveler

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _traveler_candidate_bundles(t: Traveler, market: Market) -> list[Bundle]:
    """Budget-feasible + hard-constraint-satisfying bundles for traveler t."""
    min_nights = int(t.hard_constraints.get("min_nights", 1))
    if market.nights < min_nights:
        return []
    cands = []
    for b in market.bundles:
        if b.id not in t.utility:
            continue
        if package_price(b, market) <= t.budget + 1e-9:
            cands.append(b)
    return cands


def _ir_candidate_bundles(t: Traveler, market: Market) -> list[Bundle]:
    """Feasible bundles with nonnegative surplus (individually rational)."""
    out = []
    for b in _traveler_candidate_bundles(t, market):
        surplus = float(t.utility[b.id]) - package_price(b, market)
        if surplus >= 0:
            out.append(b)
    return out


def _summarize_allocation(
    assignments: dict[str, Optional[str]],
    market: Market,
) -> Allocation:
    """Compute surplus / valuation / platform revenue for a given assignment map."""
    total_surplus = 0.0
    total_valuation = 0.0
    platform_revenue = 0.0
    bundle_by_id = {b.id: b for b in market.bundles}
    traveler_by_id = {t.id: t for t in market.travelers}
    for tid, bid in assignments.items():
        if bid is None:
            continue
        t = traveler_by_id[tid]
        b = bundle_by_id[bid]
        v = float(t.utility[bid])
        p = package_price(b, market)
        total_valuation += v
        total_surplus += v - p
        platform_revenue += market.commission_rate * p
    return Allocation(
        assignments=dict(assignments),
        total_traveler_surplus=total_surplus,
        total_valuation=total_valuation,
        platform_revenue=platform_revenue,
    )


def _capacity_check(assignments: dict[str, Optional[str]], market: Market) -> bool:
    """Shared feasibility validator — same rules all baselines must respect."""
    bundle_by_id = {b.id: b for b in market.bundles}
    hotel_cap = {h.id: h.inventory for h in market.hotels}
    flight_cap = {a.id: a.seats for a in market.airlines}
    for bid in assignments.values():
        if bid is None:
            continue
        b = bundle_by_id[bid]
        hotel_cap[b.hotel_id] -= 1
        flight_cap[b.flight_id] -= 1
        if hotel_cap[b.hotel_id] < 0 or flight_cap[b.flight_id] < 0:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Efficient rational mechanisms (expected welfare gap ≈ 0)
# ──────────────────────────────────────────────────────────────────────────────

def central_matching(market: Market) -> Allocation:
    """Centralized welfare-max allocation = MILP oracle. C1a baseline."""
    return milp_oracle(market)


def _market_minus_traveler(market: Market, tid: str) -> Market:
    """Return a copy of the market with traveler `tid` removed."""
    new_travelers = tuple(t for t in market.travelers if t.id != tid)
    return Market(
        id=f"{market.id}_drop_{tid}",
        seed=market.seed,
        travelers=new_travelers,
        hotels=market.hotels,
        airlines=market.airlines,
        bundles=market.bundles,
        commission_rate=market.commission_rate,
        nights=market.nights,
    )


def truthful_vcg(market: Market) -> tuple[Allocation, dict[str, float]]:
    """Direct-revelation VCG. Allocation is welfare-optimal (from milp_oracle).

    Returns `(allocation, payments)` where `payments[tid]` is the VCG externality
    payment (nonnegative, zero for losers). Asserts IR: no winner pays more than
    their gross valuation.

    Payment formula (quasi-linear, surplus-based welfare):
        p_i = max-welfare-without-i − (welfare-with-i − surplus_i)
            = S' − S + s_i
    where S = oracle welfare with all travelers, S' = oracle welfare excluding i,
    and s_i = traveler i's surplus in the oracle allocation.
    """
    alloc = milp_oracle(market)
    payments: dict[str, float] = {t.id: 0.0 for t in market.travelers}
    S = alloc.total_traveler_surplus
    bundle_by_id = {b.id: b for b in market.bundles}

    for t in market.travelers:
        chosen = alloc.assignments.get(t.id)
        if chosen is None:
            payments[t.id] = 0.0
            continue
        chosen_bundle = bundle_by_id[chosen]
        valuation = float(t.utility[chosen])
        price = package_price(chosen_bundle, market)
        surplus_i = valuation - price
        market_without = _market_minus_traveler(market, t.id)
        S_prime = milp_oracle(market_without).total_traveler_surplus if market_without.travelers else 0.0
        payment = S_prime - (S - surplus_i)
        # Numerical noise: clamp tiny negatives to zero.
        if -1e-6 <= payment < 0:
            payment = 0.0
        assert payment >= -1e-6, (
            f"VCG payment negative (>1e-6): {t.id} pays {payment:.6f}"
        )
        # IR: winner's net utility = valuation − price − payment ≥ 0 (i.e. surplus ≥ payment).
        assert surplus_i + 1e-6 >= payment, (
            f"VCG IR violated for {t.id}: surplus_i={surplus_i:.4f} < payment={payment:.4f}"
        )
        payments[t.id] = float(max(0.0, payment))
    return alloc, payments


# ──────────────────────────────────────────────────────────────────────────────
# Friction baselines (expected welfare gap > 0)
# ──────────────────────────────────────────────────────────────────────────────

def random_feasible(market: Market, seed: int) -> Allocation:
    """Each traveler picks uniformly from feasible bundles + no-deal. May produce
    negative surplus. Capacity conflicts resolved by random tiebreak on travelers."""
    rng = np.random.default_rng(seed)
    traveler_order = list(market.travelers)
    rng.shuffle(traveler_order)
    assignments: dict[str, Optional[str]] = {t.id: None for t in market.travelers}
    hotel_cap = {h.id: h.inventory for h in market.hotels}
    flight_cap = {a.id: a.seats for a in market.airlines}
    bundle_by_id = {b.id: b for b in market.bundles}
    for t in traveler_order:
        cands = _traveler_candidate_bundles(t, market)
        # Uniform over cands + None.
        choices: list[Optional[Bundle]] = [*cands, None]
        pick = choices[int(rng.integers(0, len(choices)))]
        if pick is None:
            continue
        if hotel_cap[pick.hotel_id] >= 1 and flight_cap[pick.flight_id] >= 1:
            assignments[t.id] = pick.id
            hotel_cap[pick.hotel_id] -= 1
            flight_cap[pick.flight_id] -= 1
        # else: capacity-out, no-deal.
    assert _capacity_check(assignments, market), "random_feasible produced infeasible alloc"
    return _summarize_allocation(assignments, market)


def random_ir(market: Market, seed: int) -> Allocation:
    """Uniform over IR-feasible (surplus ≥ 0) + no-deal. 'Naive rational' control."""
    rng = np.random.default_rng(seed)
    traveler_order = list(market.travelers)
    rng.shuffle(traveler_order)
    assignments: dict[str, Optional[str]] = {t.id: None for t in market.travelers}
    hotel_cap = {h.id: h.inventory for h in market.hotels}
    flight_cap = {a.id: a.seats for a in market.airlines}
    for t in traveler_order:
        cands = _ir_candidate_bundles(t, market)
        choices: list[Optional[Bundle]] = [*cands, None]
        pick = choices[int(rng.integers(0, len(choices)))]
        if pick is None:
            continue
        if hotel_cap[pick.hotel_id] >= 1 and flight_cap[pick.flight_id] >= 1:
            assignments[t.id] = pick.id
            hotel_cap[pick.hotel_id] -= 1
            flight_cap[pick.flight_id] -= 1
    assert _capacity_check(assignments, market), "random_ir produced infeasible alloc"
    return _summarize_allocation(assignments, market)


def _first_price_auction(market: Market, bid_of: dict[tuple[str, str], float]) -> Allocation:
    """Shared core: greedy descending-bid assignment with mutual exclusion + capacity.

    `bid_of[(traveler_id, bundle_id)]` = traveler's bid for that bundle. Pairs not
    in the dict are "no bid" and skipped.
    """
    assignments: dict[str, Optional[str]] = {t.id: None for t in market.travelers}
    hotel_cap = {h.id: h.inventory for h in market.hotels}
    flight_cap = {a.id: a.seats for a in market.airlines}
    bundle_by_id = {b.id: b for b in market.bundles}
    assigned_travelers: set[str] = set()
    # Sort by bid desc; ties broken deterministically by (traveler_id, bundle_id).
    ranked = sorted(
        bid_of.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    for (tid, bid), _ in ranked:
        if tid in assigned_travelers:
            continue
        b = bundle_by_id[bid]
        if hotel_cap[b.hotel_id] >= 1 and flight_cap[b.flight_id] >= 1:
            assignments[tid] = bid
            hotel_cap[b.hotel_id] -= 1
            flight_cap[b.flight_id] -= 1
            assigned_travelers.add(tid)
    assert _capacity_check(assignments, market), "first-price auction produced infeasible alloc"
    return _summarize_allocation(assignments, market)


def first_price_truthful(market: Market) -> Allocation:
    """Per-bundle first-price auction with mutual exclusion. Travelers bid their
    true valuation on each IR-feasible bundle. Losses vs oracle = exposure/menu
    friction (no strategic shading)."""
    bids: dict[tuple[str, str], float] = {}
    for t in market.travelers:
        for b in _ir_candidate_bundles(t, market):
            bids[(t.id, b.id)] = float(t.utility[b.id])
    return _first_price_auction(market, bids)


def first_price_shaded(market: Market, alpha: float) -> Allocation:
    """First-price with bid = alpha × valuation. Difference vs `first_price_truthful`
    on same scenario isolates the strategic-shading loss. α ∈ (0, 1]."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    bids: dict[tuple[str, str], float] = {}
    for t in market.travelers:
        for b in _ir_candidate_bundles(t, market):
            bids[(t.id, b.id)] = alpha * float(t.utility[b.id])
    return _first_price_auction(market, bids)


# ──────────────────────────────────────────────────────────────────────────────
# Gap / regret metrics
# ──────────────────────────────────────────────────────────────────────────────

def welfare_gap(oracle: Allocation, baseline: Allocation) -> float:
    """Normalized gap: (oracle − baseline) / max(|oracle|, 1e-6).

    Warning: if `baseline` admits negative surplus (e.g. random_feasible), gap can
    exceed 1.0. For that case prefer `regret()`.
    """
    return (oracle.total_traveler_surplus - baseline.total_traveler_surplus) / max(
        abs(oracle.total_traveler_surplus), 1e-6
    )


def regret(oracle: Allocation, baseline: Allocation) -> float:
    """Raw welfare loss relative to oracle. Unbounded; sign-insensitive."""
    return oracle.total_traveler_surplus - baseline.total_traveler_surplus


__all__ = [
    "central_matching",
    "truthful_vcg",
    "random_feasible",
    "random_ir",
    "first_price_truthful",
    "first_price_shaded",
    "welfare_gap",
    "regret",
]

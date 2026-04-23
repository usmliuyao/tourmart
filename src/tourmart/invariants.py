"""E1 economic invariance tests — 5 property-based checks + shared feasibility validator.

Each invariant holds by economic reasoning and should pass on any correctly
implemented welfare simulator. Failure indicates a simulator bug, not a modelling
choice.

Cut from GPT's original list of 10 to 5 (post-battle Round 4):
  * Dropped budget-monotonicity as dual of price-monotonicity.
  * Dropped VCG-IR (moved to `baselines.truthful_vcg` assertion).
  * Dropped MILP==exhaustive parity (already in E0 `test_oracle_consistency`).
  * Dropped scenario-nontriviality (already checked in E0 tight-regime test).
  * Dropped no-deal dominance → kept one clean version.
"""
from __future__ import annotations

from dataclasses import replace as _replace

from .oracle import milp_oracle, package_price
from .primitives import Airline, Allocation, Bundle, Hotel, Market, Traveler


# ──────────────────────────────────────────────────────────────────────────────
# Shared feasibility validator
# ──────────────────────────────────────────────────────────────────────────────

def assert_capacity_feasible(allocation: Allocation, market: Market) -> None:
    """Raise AssertionError if `allocation` violates hotel inventory or airline seats.

    Invariant #5 (shared): all baselines AND the oracle must produce allocations
    that respect the same capacity rules.
    """
    bundle_by_id = {b.id: b for b in market.bundles}
    hotel_load: dict[str, int] = {}
    flight_load: dict[str, int] = {}
    for tid, bid in allocation.assignments.items():
        if bid is None:
            continue
        b = bundle_by_id[bid]
        hotel_load[b.hotel_id] = hotel_load.get(b.hotel_id, 0) + 1
        flight_load[b.flight_id] = flight_load.get(b.flight_id, 0) + 1
    for h in market.hotels:
        assert hotel_load.get(h.id, 0) <= h.inventory, (
            f"Hotel {h.id} overloaded: {hotel_load.get(h.id, 0)} > cap {h.inventory}"
        )
    for f in market.airlines:
        assert flight_load.get(f.id, 0) <= f.seats, (
            f"Airline {f.id} overloaded: {flight_load.get(f.id, 0)} > cap {f.seats}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Invariants 1-4 (welfare monotonicity properties)
# ──────────────────────────────────────────────────────────────────────────────

def check_capacity_monotonicity(market: Market, hotel_id: str) -> tuple[float, float]:
    """Inv 1: increasing a hotel's inventory never reduces oracle welfare.

    Returns `(welfare_base, welfare_inc)`. Caller asserts `welfare_inc >= welfare_base`.
    """
    base = milp_oracle(market).total_traveler_surplus
    new_hotels = tuple(
        _replace(h, inventory=h.inventory + 1) if h.id == hotel_id else h
        for h in market.hotels
    )
    inc_market = _replace(market, hotels=new_hotels)
    inc = milp_oracle(inc_market).total_traveler_surplus
    return base, inc


def check_valuation_monotonicity(
    market: Market, traveler_id: str, bundle_id: str, delta: float = 100.0,
) -> tuple[float, float]:
    """Inv 2: increasing traveler.utility[bundle_id] by delta never lowers welfare."""
    base = milp_oracle(market).total_traveler_surplus
    new_travelers = list(market.travelers)
    for i, t in enumerate(new_travelers):
        if t.id == traveler_id and bundle_id in t.utility:
            new_utility = dict(t.utility)
            new_utility[bundle_id] = new_utility[bundle_id] + delta
            new_travelers[i] = _replace(t, utility=new_utility)
            break
    inc_market = _replace(market, travelers=tuple(new_travelers))
    inc = milp_oracle(inc_market).total_traveler_surplus
    return base, inc


def check_no_deal_dominance(market: Market) -> bool:
    """Inv 3: if ALL feasible bundles have negative surplus for every traveler,
    the oracle must assign `None` to all travelers.

    We synthesize a market where every traveler's valuation is set below-price.
    """
    # For each traveler, set utility < price for all bundles.
    new_travelers = []
    for t in market.travelers:
        new_utility = {}
        for b in market.bundles:
            price = package_price(b, market)
            new_utility[b.id] = price - 10.0  # strictly below price → negative surplus
        new_travelers.append(_replace(t, utility=new_utility))
    forced_market = _replace(market, travelers=tuple(new_travelers))
    alloc = milp_oracle(forced_market)
    return all(v is None for v in alloc.assignments.values())


def check_price_monotonicity(
    market: Market, hotel_id: str, delta: float = 100.0,
) -> tuple[float, float]:
    """Inv 4: increasing a hotel's nightly price never raises traveler-surplus welfare."""
    base = milp_oracle(market).total_traveler_surplus
    new_hotels = tuple(
        _replace(h, nightly_price=h.nightly_price + delta) if h.id == hotel_id else h
        for h in market.hotels
    )
    inc_market = _replace(market, hotels=new_hotels)
    inc = milp_oracle(inc_market).total_traveler_surplus
    return base, inc


__all__ = [
    "assert_capacity_feasible",
    "check_capacity_monotonicity",
    "check_valuation_monotonicity",
    "check_no_deal_dominance",
    "check_price_monotonicity",
]

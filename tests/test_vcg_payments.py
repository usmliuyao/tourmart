"""Hand-built VCG payment regression tests — two tiny markets with known answers.

Tests the VCG externality calculation directly, not via the welfare claim.
Added per Round 5 peer-debate: under uniform-valuation scenarios the standard
IR assertion can never fail, so payment correctness needs a dedicated check.
"""
from __future__ import annotations

from tourmart.baselines import truthful_vcg
from tourmart.oracle import package_price
from tourmart.primitives import (
    Airline,
    Bundle,
    Hotel,
    Market,
    Traveler,
    TravelerArchetype,
)


_ARCH = TravelerArchetype(id="solo", vibe_tags=("solo",), companion_structure="solo")


def _build_two_traveler_single_capacity_market() -> Market:
    """T1 value=1000, T2 value=700, one bundle with capacity 1.
    Oracle welfare winner: T1. VCG payment for T1 = T2's displaced surplus = 700 - price."""
    hotels = (Hotel(id="H", city="x", star=4, inventory=1, nightly_price=100.0),)
    airlines = (Airline(id="F", route=("A", "B"), seats=1, base_price=200.0),)
    bundles = (Bundle(id="B", hotel_id="H", flight_id="F", extras=(), extras_price=0.0),)
    # nights=1 → price = 100*1 + 200 + 0 = 300.
    travelers = (
        Traveler(id="T1", archetype=_ARCH, budget=2000.0, utility={"B": 1000.0},
                 hard_constraints={"min_nights": 1}),
        Traveler(id="T2", archetype=_ARCH, budget=2000.0, utility={"B": 700.0},
                 hard_constraints={"min_nights": 1}),
    )
    return Market(id="vcg_two_t_cap1", seed=0, travelers=travelers, hotels=hotels,
                  airlines=airlines, bundles=bundles, commission_rate=0.15, nights=1)


def _build_one_traveler_no_displaced_market() -> Market:
    """T1 has the only feasible positive-surplus bundle. No displaced bidder → payment=0."""
    hotels = (Hotel(id="H", city="x", star=4, inventory=1, nightly_price=100.0),)
    airlines = (Airline(id="F", route=("A", "B"), seats=1, base_price=200.0),)
    bundles = (Bundle(id="B", hotel_id="H", flight_id="F", extras=(), extras_price=0.0),)
    travelers = (
        Traveler(id="T1", archetype=_ARCH, budget=2000.0, utility={"B": 1000.0},
                 hard_constraints={"min_nights": 1}),
        # T2 has no utility entry for B → cannot compete.
        Traveler(id="T2", archetype=_ARCH, budget=2000.0, utility={},
                 hard_constraints={"min_nights": 1}),
    )
    return Market(id="vcg_one_winner_no_rival", seed=0, travelers=travelers, hotels=hotels,
                  airlines=airlines, bundles=bundles, commission_rate=0.15, nights=1)


def test_vcg_payment_equals_displaced_surplus():
    m = _build_two_traveler_single_capacity_market()
    alloc, payments = truthful_vcg(m)
    # T1 wins.
    assert alloc.assignments["T1"] == "B"
    assert alloc.assignments["T2"] is None
    # Price = 300. T2's surplus if they had won = 700 - 300 = 400.
    # T1's VCG payment = T2's displaced surplus = 400.
    price = package_price(m.bundles[0], m)
    assert price == 300.0
    expected_payment_t1 = 700.0 - price  # 400
    assert abs(payments["T1"] - expected_payment_t1) < 1e-6, (
        f"T1 VCG payment: expected {expected_payment_t1}, got {payments['T1']}"
    )
    assert payments["T2"] == 0.0


def test_vcg_payment_zero_when_no_displaced_bidder():
    m = _build_one_traveler_no_displaced_market()
    alloc, payments = truthful_vcg(m)
    assert alloc.assignments["T1"] == "B"
    assert alloc.assignments["T2"] is None
    # No one was displaced by T1 → externality = 0.
    assert payments["T1"] == 0.0, f"Expected 0 payment with no rival, got {payments['T1']}"
    assert payments["T2"] == 0.0

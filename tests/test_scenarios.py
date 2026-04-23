"""Scenario-generator sanity tests: determinism, sizes, regime differences."""
from tourmart.scenarios import (
    generate_medium_market,
    generate_scenario_bank,
    generate_small_market,
)


def test_small_market_size_and_determinism():
    m1 = generate_small_market(seed=42, regime="loose")
    m2 = generate_small_market(seed=42, regime="loose")
    assert m1 == m2, "Same seed must produce identical market"
    assert len(m1.travelers) == 3
    assert len(m1.hotels) == 4
    assert len(m1.airlines) == 2
    assert len(m1.bundles) == 6
    assert m1.nights == 3


def test_different_seeds_differ():
    m1 = generate_small_market(seed=1, regime="loose")
    m2 = generate_small_market(seed=2, regime="loose")
    assert m1 != m2


def test_tight_regime_has_shared_hotel_flight():
    """Tight-regime bundles should share the popular hotel/flight."""
    m = generate_small_market(seed=100, regime="tight")
    hotels_used = [b.hotel_id for b in m.bundles]
    flights_used = [b.flight_id for b in m.bundles]
    assert max(hotels_used.count(h) for h in set(hotels_used)) >= 2
    assert max(flights_used.count(f) for f in set(flights_used)) >= 2


def test_medium_market_size():
    m = generate_medium_market(seed=2000, regime="loose")
    assert len(m.travelers) == 10
    assert len(m.hotels) == 15
    assert len(m.airlines) == 8
    assert len(m.bundles) == 30


def test_scenario_bank_composition():
    bank = generate_scenario_bank(
        n_small_loose=5, n_small_tight=5, n_medium_loose=2, n_medium_tight=2,
    )
    assert len(bank) == 14
    labels = [m.id.split("_")[0] + "_" + m.id.split("_")[1] for m in bank]
    assert labels.count("small_loose") == 5
    assert labels.count("small_tight") == 5
    assert labels.count("medium_loose") == 2
    assert labels.count("medium_tight") == 2

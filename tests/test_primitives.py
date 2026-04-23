"""Basic sanity tests for frozen dataclass primitives."""
from tourmart import (
    Airline,
    Allocation,
    Bundle,
    Hotel,
    Market,
    Traveler,
    TravelerArchetype,
)


def test_archetype_equality_and_immutability():
    a = TravelerArchetype(id="x", vibe_tags=("solo",), companion_structure="solo")
    b = TravelerArchetype(id="x", vibe_tags=("solo",), companion_structure="solo")
    assert a == b
    try:
        a.id = "y"
    except Exception:
        pass
    else:
        raise AssertionError("TravelerArchetype should be frozen")


def test_traveler_equality():
    arch = TravelerArchetype(id="a1", vibe_tags=("solo",), companion_structure="solo")
    t1 = Traveler(id="t0", archetype=arch, budget=2000.0, utility={"b0": 100.0}, hard_constraints={"min_nights": 2})
    t2 = Traveler(id="t0", archetype=arch, budget=2000.0, utility={"b0": 100.0}, hard_constraints={"min_nights": 2})
    assert t1 == t2


def test_allocation_fields_present():
    alloc = Allocation(
        assignments={"t0": "b0", "t1": None},
        total_traveler_surplus=123.4,
        total_valuation=567.8,
        platform_revenue=89.1,
    )
    assert alloc.total_traveler_surplus == 123.4
    assert alloc.total_valuation == 567.8
    assert alloc.platform_revenue == 89.1
    assert alloc.assignments["t1"] is None

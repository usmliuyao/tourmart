"""Frozen dataclasses for TourMart market primitives.

Design rule: every scenario-level object is immutable and hashable-by-equality.
No behaviour here; all semantics live in oracle.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class TravelerArchetype:
    id: str
    vibe_tags: tuple[str, ...]
    companion_structure: str  # "solo" | "couple" | "family3" | "family4"


@dataclass(frozen=True)
class Traveler:
    """A traveler with private valuations over bundles.

    `utility[bundle_id]` is the traveler's private valuation for that bundle.
    A bundle not in `utility` is considered unacceptable (utility = -inf in effect).
    Feasibility (budget, hard constraints) is checked by the oracle at solve time.
    """
    id: str
    archetype: TravelerArchetype
    budget: float
    utility: Mapping[str, float]
    hard_constraints: Mapping[str, Any]  # V1: {"min_nights": int}


@dataclass(frozen=True)
class Hotel:
    id: str
    city: str
    star: int              # 3 / 4 / 5
    inventory: int         # rooms available for this market
    nightly_price: float


@dataclass(frozen=True)
class Airline:
    id: str
    route: tuple[str, str]  # (origin, destination)
    seats: int
    base_price: float       # round-trip


@dataclass(frozen=True)
class Bundle:
    """A sellable package pinning one hotel and one flight plus optional extras."""
    id: str
    hotel_id: str
    flight_id: str
    extras: tuple[str, ...]
    extras_price: float


@dataclass(frozen=True)
class Market:
    id: str
    seed: int
    travelers: tuple[Traveler, ...]
    hotels: tuple[Hotel, ...]
    airlines: tuple[Airline, ...]
    bundles: tuple[Bundle, ...]
    commission_rate: float  # e.g. 0.15 = 15%
    nights: int             # trip length (uniform across travelers in V1)


@dataclass(frozen=True)
class Allocation:
    """Oracle output: one bundle (or no-deal) per traveler, with summary metrics.

    Design note: `Traveler.utility[bundle_id]` is gross valuation (willingness-to-pay).
    The welfare objective that oracles optimize is `total_traveler_surplus` = gross
    valuation minus price, summed over filled assignments. Storing both surplus and
    valuation lets downstream code (E2 OTA objective sweep, C3 ManipulationPremium)
    compute either metric without re-running the oracle.
    """
    assignments: Mapping[str, Optional[str]]   # traveler_id -> bundle_id | None
    total_traveler_surplus: float              # welfare objective: Σ (valuation - price)
    total_valuation: float                     # gross: Σ valuation (no price subtracted)
    platform_revenue: float                    # Σ commission_rate * price over filled

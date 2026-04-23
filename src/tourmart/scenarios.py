"""Scenario generators for TourMart E0.

V1 keeps utility distributions uniform iid (synthetic, but mathematically tight for
E0 correctness testing). Feature-based valuations (star / extras / archetype match)
are deferred to the scenario-realism pass (E8 expert audit).

Two regimes per size:
- `loose`: inventory/seats sampled from {2, 3} — capacity rarely binds.
- `tight`: bundles share popular hotels/flights AND key inventories are set to 1 —
  capacity binds on most scenarios.

`generate_scenario_bank` mixes regimes so E1 baseline calibration exercises both.

Phase 1d adds `priors_mode` — see PHASE1D_HBD_CALIBRATED_PREREG.md. Default
`"uniform"` preserves v4 bit-for-bit; non-default modes alter `_sample_travelers`.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .primitives import (
    Airline,
    Bundle,
    Hotel,
    Market,
    Traveler,
    TravelerArchetype,
)

# Fixed archetype roster — shared across V1 scenarios.
_ARCHETYPES: tuple[TravelerArchetype, ...] = (
    TravelerArchetype(id="solo_leisure", vibe_tags=("solo", "leisure"), companion_structure="solo"),
    TravelerArchetype(id="couple_romance", vibe_tags=("couple", "romance"), companion_structure="couple"),
    TravelerArchetype(id="family3_kids", vibe_tags=("family", "kids"), companion_structure="family3"),
    TravelerArchetype(id="family4_kids", vibe_tags=("family", "kids"), companion_structure="family4"),
    TravelerArchetype(id="solo_business", vibe_tags=("solo", "business"), companion_structure="solo"),
    TravelerArchetype(id="couple_cultural", vibe_tags=("couple", "cultural"), companion_structure="couple"),
)

# ─── Phase 1d: HBD-derived priors (Kaggle Hotel Booking Demand, 119K) ─────────
# Source: external_data/proposed_scenarios_priors.json (frozen).
# HBD companion_structure distribution split symmetrically within each class
# across the 6 TourMart archetypes, then renormalized after dropping "other".
# See PHASE1D_HBD_CALIBRATED_PREREG.md §2.3 for derivation and mapping hash.
_HBD_ARCHETYPE_RAW: dict[str, float] = {
    "solo_leisure":    0.10722,
    "couple_romance":  0.32895,
    "family3_kids":    0.09534,
    "family4_kids":    0.03199,
    "solo_business":   0.10722,
    "couple_cultural": 0.32895,
}
# Raw HBD quantization sums to 0.99967; renormalize to exact 1.0 for sampling.
_HBD_S = sum(_HBD_ARCHETYPE_RAW.values())
HBD_ARCHETYPE_WEIGHTS: dict[str, float] = {
    k: v / _HBD_S for k, v in _HBD_ARCHETYPE_RAW.items()
}
assert abs(sum(HBD_ARCHETYPE_WEIGHTS.values()) - 1.0) < 1e-9, \
    "HBD_ARCHETYPE_WEIGHTS must sum to 1.0"

# Scale-normalized HBD-shape budget prior.
# HBD raw LogNormal(μ=5.538, σ=0.844) has mean €354 — *hotel booking spend*, not
# total trip budget. We preserve the right-skew (σ=0.844) but anchor the
# distribution mean to v4 Uniform(1500, 5000)'s mean of €3250 so support overlaps
# TourMart package prices. Derivation: mean = exp(μ + σ²/2) = 3250 → μ' = 7.731.
HBD_BUDGET_LOGNORMAL_MU_SCALE_NORMALIZED: float = 7.7310
HBD_BUDGET_LOGNORMAL_SIGMA: float = 0.8435

PriorsMode = Literal["uniform", "hbd_scale_normalized", "hbd_archetype_only",
                    "hbd_direct", "hbd_3x_hotel"]


def _hbd_archetype_index_list() -> np.ndarray:
    """Ordered weights aligned with `_ARCHETYPES`, as numpy array."""
    ids = [a.id for a in _ARCHETYPES]
    w = np.array([HBD_ARCHETYPE_WEIGHTS[i] for i in ids], dtype=np.float64)
    return w / w.sum()

_EXTRAS_MENU: tuple[tuple[str, ...], ...] = (
    (),
    ("breakfast",),
    ("airport_shuttle",),
    ("breakfast", "airport_shuttle"),
    ("spa",),
)

Regime = Literal["loose", "tight"]


def _sample_hotels(rng: np.random.Generator, n: int, regime: Regime) -> tuple[Hotel, ...]:
    cities = ["beijing", "chengdu", "dali"]
    if regime == "loose":
        inventory_choices = (2, 3)
    else:
        # Tight: most hotels have inventory 1; a couple have 2 to avoid impossibility.
        inventory_choices = (1, 1, 1, 2)
    hotels = []
    for i in range(n):
        hotels.append(Hotel(
            id=f"h{i:02d}",
            city=str(rng.choice(cities)),
            star=int(rng.choice([3, 4, 5])),
            inventory=int(rng.choice(inventory_choices)),
            nightly_price=float(rng.uniform(200.0, 800.0)),
        ))
    return tuple(hotels)


def _sample_airlines(rng: np.random.Generator, n: int, regime: Regime) -> tuple[Airline, ...]:
    routes = [("PEK", "CTU"), ("PEK", "DLI"), ("SHA", "CTU"), ("CAN", "DLI")]
    if regime == "loose":
        seats_choices = (2, 3)
    else:
        seats_choices = (1, 1, 2)
    airlines = []
    for i in range(n):
        route = routes[i % len(routes)]
        airlines.append(Airline(
            id=f"f{i:02d}",
            route=route,
            seats=int(rng.choice(seats_choices)),
            base_price=float(rng.uniform(400.0, 1500.0)),
        ))
    return tuple(airlines)


def _sample_bundles(
    rng: np.random.Generator,
    hotels: tuple[Hotel, ...],
    airlines: tuple[Airline, ...],
    n: int,
    regime: Regime,
) -> tuple[Bundle, ...]:
    """Bundles pin (hotel_id, flight_id). In tight regime, bias toward sharing one
    popular hotel and one popular flight so capacity binds on those entities."""
    bundles: list[Bundle] = []
    if regime == "tight" and len(hotels) >= 1 and len(airlines) >= 1:
        popular_hotel = hotels[0].id
        popular_flight = airlines[0].id
        # Guarantee ≥2 bundles share the popular hotel and ≥2 share the popular flight.
        forced_pairs: list[tuple[str, str]] = [
            (popular_hotel, airlines[0].id),
            (popular_hotel, airlines[1 % len(airlines)].id),
            (hotels[1 % len(hotels)].id, popular_flight),
            (hotels[2 % len(hotels)].id, popular_flight),
        ]
    else:
        forced_pairs = []

    # Remaining bundles sampled uniformly with replacement over the (h, f) grid.
    all_pairs = [(h.id, f.id) for h in hotels for f in airlines]
    # Forced pairs first, then fill to n.
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for p in forced_pairs:
        if p not in seen and len(pairs) < n:
            pairs.append(p)
            seen.add(p)
    # Fill remaining by random choice without replacement.
    remaining = [p for p in all_pairs if p not in seen]
    rng.shuffle(remaining)
    for p in remaining:
        if len(pairs) >= n:
            break
        pairs.append(p)
        seen.add(p)

    for i, (h_id, f_id) in enumerate(pairs[:n]):
        extras = _EXTRAS_MENU[int(rng.integers(0, len(_EXTRAS_MENU)))]
        bundles.append(Bundle(
            id=f"b{i:02d}",
            hotel_id=h_id,
            flight_id=f_id,
            extras=extras,
            extras_price=float(rng.uniform(0.0, 200.0)),
        ))
    return tuple(bundles)


def _sample_travelers(
    rng: np.random.Generator,
    n: int,
    bundles: tuple[Bundle, ...],
    priors_mode: PriorsMode = "uniform",
) -> tuple[Traveler, ...]:
    """Sample `n` travelers under the given priors mode.

    Budget and archetype distributions depend on `priors_mode`:
      - "uniform": v4 default — Uniform(1500, 5000) budget, uniform archetype.
      - "hbd_scale_normalized": LogNormal(7.731, 0.844) budget (mean=€3250),
          HBD archetype weights. Primary phase1d prior.
      - "hbd_archetype_only": Uniform(1500, 5000) budget, HBD archetype
          weights. Secondary phase1d prior — isolates demographic effect.
      - "hbd_direct": LogNormal(5.538, 0.844) budget (mean=€354), HBD
          archetype weights. Preflight-only (support collapses with v4 prices).
      - "hbd_3x_hotel": LogNormal(5.538 + ln(3), 0.844) budget (mean=€1062), HBD
          archetype weights. Preflight-only scale-diagnostic.

    Utility is unchanged across modes (v1 Uniform[500, 4500]).
    """
    if priors_mode == "uniform":
        arch_weights = None  # uniform over _ARCHETYPES
    else:
        arch_weights = _hbd_archetype_index_list()

    travelers: list[Traveler] = []
    for i in range(n):
        if arch_weights is None:
            arch = _ARCHETYPES[int(rng.integers(0, len(_ARCHETYPES)))]
        else:
            arch = _ARCHETYPES[int(rng.choice(len(_ARCHETYPES), p=arch_weights))]

        if priors_mode in ("uniform", "hbd_archetype_only"):
            budget = float(rng.uniform(1500.0, 5000.0))
        elif priors_mode == "hbd_scale_normalized":
            budget = float(rng.lognormal(
                mean=HBD_BUDGET_LOGNORMAL_MU_SCALE_NORMALIZED,
                sigma=HBD_BUDGET_LOGNORMAL_SIGMA,
            ))
        elif priors_mode == "hbd_direct":
            budget = float(rng.lognormal(mean=5.5377, sigma=HBD_BUDGET_LOGNORMAL_SIGMA))
        elif priors_mode == "hbd_3x_hotel":
            budget = float(rng.lognormal(
                mean=5.5377 + math.log(3.0),
                sigma=HBD_BUDGET_LOGNORMAL_SIGMA,
            ))
        else:
            raise ValueError(f"unknown priors_mode: {priors_mode}")

        # Full utility matrix over all bundles. iid Uniform[500, 4500] for V1.
        # Range chosen to overlap with price range (≈ [1000, 4100]) so roughly half
        # of (traveler, bundle) pairs have positive surplus; otherwise IR filtering
        # collapses all baselines to no-deal and C1b gaps are trivially 0.
        # Feature-based valuations are deferred to V2 (E8 expert audit).
        utility = {b.id: float(rng.uniform(500.0, 4500.0)) for b in bundles}
        hard_constraints: dict[str, int] = {"min_nights": 2}
        travelers.append(Traveler(
            id=f"t{i:02d}",
            archetype=arch,
            budget=budget,
            utility=utility,
            hard_constraints=hard_constraints,
        ))
    return tuple(travelers)


def _generate(
    seed: int,
    n_travelers: int,
    n_hotels: int,
    n_airlines: int,
    n_bundles: int,
    nights: int,
    commission_rate: float,
    regime: Regime,
    size_label: str,
    priors_mode: PriorsMode = "uniform",
) -> Market:
    rng = np.random.default_rng(seed)
    hotels = _sample_hotels(rng, n_hotels, regime)
    airlines = _sample_airlines(rng, n_airlines, regime)
    bundles = _sample_bundles(rng, hotels, airlines, n_bundles, regime)
    travelers = _sample_travelers(rng, n_travelers, bundles, priors_mode=priors_mode)
    return Market(
        id=f"{size_label}_{regime}_s{seed}",
        seed=seed,
        travelers=travelers,
        hotels=hotels,
        airlines=airlines,
        bundles=bundles,
        commission_rate=commission_rate,
        nights=nights,
    )


def generate_small_market(
    seed: int,
    regime: Regime = "loose",
    priors_mode: PriorsMode = "uniform",
) -> Market:
    """3 travelers / 4 hotels / 2 airlines / 6 bundles / nights=3 / commission=15%."""
    return _generate(
        seed=seed,
        n_travelers=3, n_hotels=4, n_airlines=2, n_bundles=6,
        nights=3, commission_rate=0.15,
        regime=regime, size_label="small",
        priors_mode=priors_mode,
    )


def generate_medium_market(
    seed: int,
    regime: Regime = "loose",
    priors_mode: PriorsMode = "uniform",
) -> Market:
    """10 travelers / 15 hotels / 8 airlines / 30 bundles / nights=3 / commission=15%."""
    return _generate(
        seed=seed,
        n_travelers=10, n_hotels=15, n_airlines=8, n_bundles=30,
        nights=3, commission_rate=0.15,
        regime=regime, size_label="medium",
        priors_mode=priors_mode,
    )


def generate_scenario_bank(
    n_small_loose: int = 50,
    n_small_tight: int = 50,
    n_medium_loose: int = 25,
    n_medium_tight: int = 25,
    priors_mode: PriorsMode = "uniform",
) -> list[Market]:
    """Default bank: 100 small (50 loose + 50 tight), 50 medium (25 + 25).

    Seed ranges (deterministic, disjoint across regimes):
    - small loose : seeds 1000 .. 1000+n_small_loose-1
    - small tight : seeds 1500 .. 1500+n_small_tight-1
    - medium loose: seeds 2000 .. 2000+n_medium_loose-1
    - medium tight: seeds 2500 .. 2500+n_medium_tight-1
    """
    bank: list[Market] = []
    for i in range(n_small_loose):
        bank.append(generate_small_market(1000 + i, "loose", priors_mode=priors_mode))
    for i in range(n_small_tight):
        bank.append(generate_small_market(1500 + i, "tight", priors_mode=priors_mode))
    for i in range(n_medium_loose):
        bank.append(generate_medium_market(2000 + i, "loose", priors_mode=priors_mode))
    for i in range(n_medium_tight):
        bank.append(generate_medium_market(2500 + i, "tight", priors_mode=priors_mode))
    return bank

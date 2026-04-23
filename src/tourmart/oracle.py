"""Welfare oracles for TourMart E0.

Two oracles compute the **welfare-optimal feasible allocation**:
  * `milp_oracle(market)`: sparse-candidate MILP via OR-Tools CBC.
  * `exhaustive_oracle(market)`: brute-force Cartesian product; tractable only on small markets.

Both return an `Allocation` with three summary fields:
  * `total_traveler_surplus` — the welfare objective, Σ (valuation - price) over filled assignments.
  * `total_valuation`        — Σ gross valuation over filled assignments.
  * `platform_revenue`       — Σ commission_rate * price over filled assignments.

Feasibility of a (traveler, bundle) pair:
  1. `bundle.id in traveler.utility` (traveler has a valuation for it).
  2. `package_price(bundle, market) ≤ traveler.budget`.
  3. `market.nights ≥ traveler.hard_constraints["min_nights"]`.

Capacity constraints (fixed-window semantics for V1 — all travelers share the same date window):
  * For each hotel h: Σ assignments over bundles with `b.hotel_id == h` ≤ `h.inventory`.
  * For each airline f: Σ assignments over bundles with `b.flight_id == f` ≤ `f.seats`.
"""
from __future__ import annotations

import itertools
import math
from typing import Mapping, Optional

from ortools.linear_solver import pywraplp

from .primitives import Allocation, Bundle, Market, Traveler

_MAX_EXHAUSTIVE_COMBOS = 1_000_000
_MILP_EXHAUSTIVE_TOL = 1e-6
_MILP_EXHAUSTIVE_WARN = 1e-4  # diagnostic threshold; caller may use as soft gate


def package_price(bundle: Bundle, market: Market) -> float:
    """Total traveler-side price of a bundle for this market's itinerary window."""
    hotel = next(h for h in market.hotels if h.id == bundle.hotel_id)
    airline = next(a for a in market.airlines if a.id == bundle.flight_id)
    return hotel.nightly_price * market.nights + airline.base_price + bundle.extras_price


def _feasible_bundles_for(traveler: Traveler, market: Market) -> list[Bundle]:
    """Pre-filter bundles: present in utility, within budget, hard constraints met."""
    min_nights = int(traveler.hard_constraints.get("min_nights", 1))
    if market.nights < min_nights:
        return []
    out: list[Bundle] = []
    for b in market.bundles:
        if b.id not in traveler.utility:
            continue
        if package_price(b, market) <= traveler.budget + 1e-9:
            out.append(b)
    return out


def _summarize(
    assignments: Mapping[str, Optional[str]],
    market: Market,
) -> Allocation:
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
        valuation = float(t.utility[bid])
        price = package_price(b, market)
        total_valuation += valuation
        total_surplus += valuation - price
        platform_revenue += market.commission_rate * price
    return Allocation(
        assignments=dict(assignments),
        total_traveler_surplus=total_surplus,
        total_valuation=total_valuation,
        platform_revenue=platform_revenue,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MILP oracle
# ──────────────────────────────────────────────────────────────────────────────

def milp_oracle(market: Market) -> Allocation:
    """Welfare-optimal allocation via OR-Tools CBC.

    Decision: x[t, b] ∈ {0, 1} over (traveler, feasible_bundle) pairs.
    Objective: max Σ x[t, b] * (valuation[t, b] - price(b)).
    Constraints:
      Σ_b x[t, b] ≤ 1                         per traveler
      Σ_{t, b: hotel_id==h} x[t, b] ≤ inv(h)  per hotel
      Σ_{t, b: flight_id==f} x[t, b] ≤ seats(f)  per airline
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver is unavailable in this Python env.")

    # Build sparse candidate set.
    candidates: dict[str, list[Bundle]] = {}
    for t in market.travelers:
        candidates[t.id] = _feasible_bundles_for(t, market)

    # Variables.
    x: dict[tuple[str, str], pywraplp.Variable] = {}
    for t in market.travelers:
        for b in candidates[t.id]:
            x[(t.id, b.id)] = solver.BoolVar(f"x_{t.id}_{b.id}")

    if not x:
        # No feasible (traveler, bundle) pairs → all no-deal.
        assignments = {t.id: None for t in market.travelers}
        return _summarize(assignments, market)

    # One bundle per traveler.
    for t in market.travelers:
        cand = candidates[t.id]
        if cand:
            solver.Add(solver.Sum(x[(t.id, b.id)] for b in cand) <= 1)

    # Hotel capacity.
    for h in market.hotels:
        terms = [
            x[(t.id, b.id)]
            for t in market.travelers
            for b in candidates[t.id]
            if b.hotel_id == h.id
        ]
        if terms:
            solver.Add(solver.Sum(terms) <= h.inventory)

    # Airline capacity.
    for f in market.airlines:
        terms = [
            x[(t.id, b.id)]
            for t in market.travelers
            for b in candidates[t.id]
            if b.flight_id == f.id
        ]
        if terms:
            solver.Add(solver.Sum(terms) <= f.seats)

    # Objective: welfare = valuation - price.
    obj_terms = []
    for t in market.travelers:
        for b in candidates[t.id]:
            surplus_coef = float(t.utility[b.id]) - package_price(b, market)
            obj_terms.append(surplus_coef * x[(t.id, b.id)])
    solver.Maximize(solver.Sum(obj_terms))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"CBC did not find a feasible solution (status={status}).")

    assignments: dict[str, Optional[str]] = {t.id: None for t in market.travelers}
    for (tid, bid), var in x.items():
        if var.solution_value() > 0.5:
            assignments[tid] = bid
    return _summarize(assignments, market)


# ──────────────────────────────────────────────────────────────────────────────
# Exhaustive oracle
# ──────────────────────────────────────────────────────────────────────────────

def _exhaustive_combo_count(candidates_per_traveler: list[list[Bundle]]) -> int:
    """Exact Cartesian-product size including 'no-deal' option per traveler."""
    total = 1
    for cands in candidates_per_traveler:
        total *= (len(cands) + 1)
        if total > _MAX_EXHAUSTIVE_COMBOS:
            return total  # short-circuit; caller will raise
    return total


def exhaustive_oracle(market: Market) -> Allocation:
    """Brute-force welfare optimum. Raises if the combo space exceeds 10^6."""
    candidates_per_traveler: list[list[Bundle]] = [
        _feasible_bundles_for(t, market) for t in market.travelers
    ]
    combos = _exhaustive_combo_count(candidates_per_traveler)
    if combos > _MAX_EXHAUSTIVE_COMBOS:
        raise ValueError(
            f"Exhaustive oracle refuses: {combos} combos > {_MAX_EXHAUSTIVE_COMBOS}. "
            f"Use milp_oracle on markets of this size."
        )

    best_surplus = -math.inf
    best_assignments: dict[str, Optional[str]] = {t.id: None for t in market.travelers}

    # Per-traveler choice iterable: each bundle + None (no-deal).
    per_traveler_choices: list[list[Optional[Bundle]]] = [
        [*cands, None] for cands in candidates_per_traveler
    ]
    traveler_ids = [t.id for t in market.travelers]
    bundle_prices = {b.id: package_price(b, market) for b in market.bundles}
    utility_lookup = {t.id: t.utility for t in market.travelers}

    for combo in itertools.product(*per_traveler_choices):
        # Check capacity.
        hotel_load: dict[str, int] = {}
        flight_load: dict[str, int] = {}
        feasible = True
        for b in combo:
            if b is None:
                continue
            hotel_load[b.hotel_id] = hotel_load.get(b.hotel_id, 0) + 1
            flight_load[b.flight_id] = flight_load.get(b.flight_id, 0) + 1
        for h in market.hotels:
            if hotel_load.get(h.id, 0) > h.inventory:
                feasible = False
                break
        if not feasible:
            continue
        for f in market.airlines:
            if flight_load.get(f.id, 0) > f.seats:
                feasible = False
                break
        if not feasible:
            continue
        # Compute surplus for this combo.
        surplus = 0.0
        for tid, b in zip(traveler_ids, combo):
            if b is None:
                continue
            surplus += float(utility_lookup[tid][b.id]) - bundle_prices[b.id]
        if surplus > best_surplus:
            best_surplus = surplus
            best_assignments = {
                tid: (b.id if b is not None else None)
                for tid, b in zip(traveler_ids, combo)
            }
    if best_surplus == -math.inf:
        best_surplus = 0.0  # no-deal for everyone (trivial)
    return _summarize(best_assignments, market)

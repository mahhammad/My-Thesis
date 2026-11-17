#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the 2D circles BEFORE rendering the Persian §3.9 schematic.

This is a *Euclidean schematic* of the Hilbert-space argument, NOT the
numerical JS embedding of training examples. It does not create an image.
The image/TikZ source must be generated only after these assertions pass.

B1 = fixed orange ball, B2 = green ball recomputed without one GREEN sample,
B3 = larger fixed blue ball above both. Every ball has exactly three plotted
boundary samples, with one or two additional interior samples. The candidate
green sample stays in the raw data/graph, but is excluded from B2's fit.

No external dependencies. Run: python img/chap3/js_overlap_geometry_check.py
"""
from __future__ import annotations

from itertools import combinations
from math import cos, degrees, hypot, pi, radians, sin

Point = tuple[float, float]
EPS = 1e-8


def polar(center: Point, radius: float, degrees_: tuple[float, ...]) -> list[Point]:
    return [
        (center[0] + radius * cos(radians(a)),
         center[1] + radius * sin(radians(a)))
        for a in degrees_
    ]


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def diameter_circle(p: Point, q: Point) -> tuple[Point, float]:
    center = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return center, distance(center, p)


def circumcircle(p: Point, q: Point, r: Point) -> tuple[Point, float] | None:
    ax, ay = p
    bx, by = q
    cx, cy = r
    det = 2 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    if abs(det) < 1e-12:
        return None
    b = bx * bx + by * by - ax * ax - ay * ay
    c = cx * cx + cy * cy - ax * ax - ay * ay
    ux = (b * (cy - ay) - c * (by - ay)) / det
    uy = ((bx - ax) * c - (cx - ax) * b) / det
    center = (ux, uy)
    return center, distance(center, p)


def enclosing_ball(points: dict[str, Point]) -> tuple[Point, float, set[str]]:
    """Exact 2D minimum-enclosing-circle by enumerating 1-, 2-, 3-point fits.

    For this small, fixed educational diagram, enumerating all support subsets
    is clearer and safer than using an approximate iterative circle.
    """
    pairs = list(points.items())
    if not pairs:
        raise ValueError("Empty point set")
    candidates: list[tuple[Point, float]] = [
        (p, 0.0) for _, p in pairs
    ]
    candidates += [
        diameter_circle(p, q)
        for (_, p), (_, q) in combinations(pairs, 2)
    ]
    candidates += [
        circle
        for (_, p), (_, q), (_, r) in combinations(pairs, 3)
        if (circle := circumcircle(p, q, r)) is not None
    ]
    covering = [
        (c, rad) for c, rad in candidates
        if all(distance(c, p) <= rad + EPS for _, p in pairs)
    ]
    if not covering:
        raise AssertionError("No enclosing circle found")
    center, radius = min(covering, key=lambda item: item[1])
    boundary = {
        name for name, p in pairs
        if abs(distance(center, p) - radius) < EPS
    }
    return center, radius, boundary


# All coordinates are illustrative Euclidean drawing coordinates.
ORANGE = {
    **{f"O-rim-{j}": p for j, p in enumerate(
        polar((0.0, 0.0), 1.05, (70.0, 190.0, 310.0)), start=1)},
    "O-inside-1": (0.13, 0.22),
    "O-inside-2": (-0.32, -0.20),
}

BLUE = {
    **{f"B-rim-{j}": p for j, p in enumerate(
        polar((1.10, 2.75), 0.80, (90.0, 210.0, 330.0)), start=1)},
    "B-inside-1": (1.07, 2.65),
    "B-inside-2": (1.26, 2.90),
}

GREEN_AFTER = {
    **{f"G-rim-after-{j}": p for j, p in enumerate(
        polar((2.17, 0.0), 0.64, (65.0, 185.0, 305.0)), start=1)},
    "G-inside-1": (2.13, -0.07),
    "G-inside-2": (2.30, 0.12),
}
CANDIDATE_NAME = "G-green-boundary-to-exclude"
CANDIDATE = (0.87, 0.24)
GREEN_BEFORE = {**GREEN_AFTER, CANDIDATE: CANDIDATE}
# IMPORTANT: Candidate is still an existing GREEN data point after screening.
# It is excluded ONLY from the dict of points used for the green fitted circle.


def gap(c1: Point, r1: float, c2: Point, r2: float) -> float:
    return distance(c1, c2) - r1 - r2


def main() -> None:
    oc, ro, bo = enclosing_ball(ORANGE)
    bc, rb, bb = enclosing_ball(BLUE)
    gc0, rg0, bg0 = enclosing_ball(GREEN_BEFORE)
    gc1, rg1, bg1 = enclosing_ball(GREEN_AFTER)

    # Three TRUE boundary points in EACH circle in BOTH panels.
    assert bo == {"O-rim-1", "O-rim-2", "O-rim-3"}, bo
    assert bb == {"B-rim-1", "B-rim-2", "B-rim-3"}, bb
    assert bg0 == {"G-rim-after-1", "G-rim-after-3", CANDIDATE_NAME}, bg0
    assert bg1 == {"G-rim-after-1", "G-rim-after-2", "G-rim-after-3"}, bg1

    # Correct color, physical location and effect of removing this candidate.
    assert CANDIDATE_NAME.startswith("G-")
    assert distance(CANDIDATE, gc0) == rg0 or abs(
        distance(CANDIDATE, gc0) - rg0) < EPS
    assert distance(CANDIDATE, oc) < ro - EPS  # inside ORANGE rival
    assert distance(CANDIDATE, gc1) > rg1 + EPS  # outside NEW GREEN ball
    assert rg1 < rg0 - EPS
    assert distance(gc0, gc1) > EPS
    assert gap(oc, ro, gc0, rg0) < -EPS  # orange / green overlap before
    assert gap(oc, ro, gc1, rg1) > EPS   # orange / green separate after
    assert gap(oc, ro, bc, rb) > EPS     # blue distinct from orange
    assert gap(gc0, rg0, bc, rb) > EPS  # blue distinct from green before
    assert gap(gc1, rg1, bc, rb) > EPS  # blue distinct from green after
    assert len(GREEN_BEFORE) == 6
    assert len(GREEN_AFTER) == 5
    assert len(GREEN_BEFORE) - len(bg0) == 3
    assert len(GREEN_AFTER) - len(bg1) == 2
    assert len(ORANGE) - len(bo) == 2
    assert len(BLUE) - len(bb) == 2

    circles = (
        ("orange B1 (fixed)", oc, ro, bo),
        ("green B2 BEFORE", gc0, rg0, bg0),
        ("green B2 AFTER", gc1, rg1, bg1),
        ("blue B3 (fixed)", bc, rb, bb),
    )
    for name, center, radius, boundary in circles:
        print(f"{name:20} center={center!r} R={radius:.9f}")
        print(" " * 22 + "boundary=" + ", ".join(sorted(boundary)))
    print(f"orange/green gap BEFORE: {gap(oc,ro,gc0,rg0):+.9f}")
    print(f"orange/green gap AFTER:  {gap(oc,ro,gc1,rg1):+.9f}")
    print("GREEN candidate (same position in both panels):", CANDIDATE)
    print("GREEN candidate inside orange:", distance(CANDIDATE,oc) < ro)
    print("GREEN candidate outside new green:", distance(CANDIDATE,gc1) > rg1)
    print("PASS: geometry exact; no picture produced.")


if __name__ == "__main__":
    main()

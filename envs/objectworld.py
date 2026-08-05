"""Objectworld (Levine, Popovic & Koltun, 2011): the classic IRL benchmark, per
plan.md Phase 5's env list. An NxN grid with C colors of randomly placed objects;
the feature at a cell is, for each color, the (negated, so closer=higher) distance
to the nearest object of that color -- smooth/continuous features, in contrast to
envs/gridworld.py's hard Voronoi region indicators (whose hard boundaries were
found, in notes/phase5_gridworld_result.md's earlier exploration, to concentrate
policy sensitivity rather than smooth it out). Objectworld's distance features give
a genuinely different, smoother feature landscape worth comparing against.
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP

_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _cell_to_state(r, c, size):
    return r * size + c


def _clip(r, c, size):
    return max(0, min(size - 1, r)), max(0, min(size - 1, c))


def build_objectworld(size: int, n_colors: int, n_objects_per_color: int, gamma: float,
                       rng: np.random.Generator, slip_prob: float = 0.1,
                       start: tuple[int, int] | None = None,
                       max_distance: float | None = None) -> TabularMDP:
    """Returns a TabularMDP with feature_dim = n_colors: phi(s, a)[c] is the
    (negated, capped) distance from the expected successor state to the nearest
    object of color c -- closer objects give a HIGHER feature value (so a positive
    theta[c] rewards proximity to color c), a standard Objectworld convention.
    """
    S = size * size
    A = 4
    if max_distance is None:
        max_distance = 2 * (size - 1)  # max possible Manhattan distance on the grid

    # place n_objects_per_color objects of each color at random distinct cells
    all_cells = rng.choice(S, size=min(S, n_colors * n_objects_per_color), replace=False)
    objects_by_color = {c: [] for c in range(n_colors)}
    for i, cell in enumerate(all_cells):
        color = i % n_colors
        objects_by_color[color].append((cell // size, cell % size))

    def dist_to_nearest(r, c, color):
        cells = objects_by_color[color]
        if not cells:
            return max_distance
        dists = [abs(r - orow) + abs(c - ocol) for orow, ocol in cells]
        return min(min(dists), max_distance)

    # feature per grid cell: negated normalized distance per color (closer = higher)
    cell_feature = np.zeros((S, n_colors))
    for r in range(size):
        for c in range(size):
            s = _cell_to_state(r, c, size)
            for color in range(n_colors):
                d = dist_to_nearest(r, c, color)
                cell_feature[s, color] = 1.0 - d / max_distance  # in [0, 1]

    P = np.zeros((S, A, S))
    phi = np.zeros((S, A, n_colors))
    for r in range(size):
        for c in range(size):
            s = _cell_to_state(r, c, size)
            for a, (dr, dc) in enumerate(_DELTAS):
                targets = []
                nr, nc = _clip(r + dr, c + dc, size)
                targets.append((1 - slip_prob, _cell_to_state(nr, nc, size)))
                perp = [(dc, dr), (-dc, -dr)] if (dr, dc) != (0, 0) else [(0, 0), (0, 0)]
                for (pdr, pdc) in perp:
                    pr, pc = _clip(r + pdr, c + pdc, size)
                    targets.append((slip_prob / 2, _cell_to_state(pr, pc, size)))
                for prob, s_next in targets:
                    P[s, a, s_next] += prob
                phi[s, a, :] = sum(prob * cell_feature[s_next] for prob, s_next in targets)

    if start is None:
        start = (0, 0)
    mu0 = np.zeros(S)
    mu0[_cell_to_state(*start, size)] = 1.0

    return TabularMDP(P, phi, gamma, mu0)

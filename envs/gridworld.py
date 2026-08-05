"""Region-based gridworld: a more realistic Phase 5 testbed than envs.mdp.random_mdp.

Motivation (notes/phase2_end_to_end_finding.md): random_mdp's i.i.d. per-(s,a)
Gaussian features give an MDP with NO spatial structure -- nearby reward directions
can induce completely different optimal policies at a large fraction of states,
making robust hedging pay a large, somewhat artificial price. Real-world reward
structure (and CIO's own shortest-path/knapsack testbeds) is spatially/structurally
correlated: nearby states or nearby decisions tend to have SIMILAR features, so small
reward-direction perturbations tend to have LOCALIZED, not global, effects on the
optimal policy.

This env: an NxN grid. Reward depends on the REGION of the cell the agent moves INTO
(a Voronoi tessellation of the grid from K random seed cells, giving spatially
contiguous regions -- adjacent cells are usually in the same region), each region
carrying its own feature vector, so r_theta(s,a) = theta . phi(s,a) where phi(s,a) is
the feature vector of the region containing the successor state (in expectation, if
transitions are stochastic).
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP

# actions: 0=up, 1=down, 2=left, 3=right
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _cell_to_state(r, c, size):
    return r * size + c


def _clip(r, c, size):
    return max(0, min(size - 1, r)), max(0, min(size - 1, c))


def build_region_gridworld(size: int, n_regions: int, feature_dim: int, gamma: float,
                            rng: np.random.Generator, slip_prob: float = 0.1,
                            start: tuple[int, int] | None = None) -> tuple[TabularMDP, np.ndarray]:
    """Returns (mdp, region_id_per_state). Region features are random unit-scale
    vectors in R^feature_dim, one per region (feature_dim can differ from n_regions --
    features need not be one-hot, giving genuine low-dimensional theta while regions
    remain spatially contiguous).
    """
    S = size * size
    A = 4

    # Voronoi tessellation from n_regions random seed cells (Euclidean nearest-seed).
    seeds = rng.choice(S, size=n_regions, replace=False)
    seed_coords = np.array([(s // size, s % size) for s in seeds])
    region_id = np.zeros(S, dtype=int)
    for s in range(S):
        r, c = s // size, s % size
        dists = np.sum((seed_coords - np.array([r, c])) ** 2, axis=1)
        region_id[s] = np.argmin(dists)

    region_features = rng.normal(size=(n_regions, feature_dim))
    region_features /= np.linalg.norm(region_features, axis=1, keepdims=True)

    P = np.zeros((S, A, S))
    phi = np.zeros((S, A, feature_dim))
    for r in range(size):
        for c in range(size):
            s = _cell_to_state(r, c, size)
            for a, (dr, dc) in enumerate(_DELTAS):
                # intended successor, plus slip to the two perpendicular directions
                targets = []
                nr, nc = _clip(r + dr, c + dc, size)
                targets.append((1 - slip_prob, _cell_to_state(nr, nc, size)))
                perp = [(dc, dr), (-dc, -dr)] if (dr, dc) != (0, 0) else [(0, 0), (0, 0)]
                for (pdr, pdc) in perp:
                    pr, pc = _clip(r + pdr, c + pdc, size)
                    targets.append((slip_prob / 2, _cell_to_state(pr, pc, size)))
                for prob, s_next in targets:
                    P[s, a, s_next] += prob
                phi[s, a, :] = sum(prob * region_features[region_id[s_next]]
                                    for prob, s_next in targets)

    if start is None:
        start = (0, 0)
    mu0 = np.zeros(S)
    mu0[_cell_to_state(*start, size)] = 1.0

    mdp = TabularMDP(P, phi, gamma, mu0)
    return mdp, region_id

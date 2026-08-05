import numpy as np
import pytest

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration


@pytest.mark.parametrize("seed", range(5))
def test_gridworld_is_valid_mdp(seed):
    rng = np.random.default_rng(seed)
    mdp, region_id = build_region_gridworld(size=6, n_regions=4, feature_dim=3,
                                             gamma=0.9, rng=rng)
    assert np.allclose(mdp.P.sum(axis=2), 1.0)
    assert mdp.S == 36
    assert region_id.shape == (36,)
    assert region_id.min() >= 0 and region_id.max() < 4


def test_gridworld_features_are_spatially_contiguous():
    """Sanity check on the actual structural property this env provides: adjacent
    cells usually share a region (hence identical or near-identical successor
    features), unlike random_mdp's i.i.d. per-(s,a) features."""
    rng = np.random.default_rng(0)
    size = 8
    mdp, region_id = build_region_gridworld(size=size, n_regions=4, feature_dim=3,
                                              gamma=0.9, rng=rng)
    same_region_neighbor_fraction = []
    for r in range(size):
        for c in range(size):
            s = r * size + c
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * size + c)
            if r < size - 1:
                neighbors.append((r + 1) * size + c)
            if c > 0:
                neighbors.append(r * size + c - 1)
            if c < size - 1:
                neighbors.append(r * size + c + 1)
            same = sum(region_id[n] == region_id[s] for n in neighbors)
            same_region_neighbor_fraction.append(same / len(neighbors))
    mean_frac = np.mean(same_region_neighbor_fraction)
    # with only 4 regions on a 64-cell grid, most cells should share a region with
    # most of their neighbors (Voronoi cells are spatially contiguous by construction)
    assert mean_frac > 0.6, f"regions unexpectedly fragmented: {mean_frac}"


def test_gridworld_policy_sensitivity_measured_honestly():
    """NOTE ON AN INITIAL HYPOTHESIS THAT DID NOT HOLD: this env was originally
    built expecting spatially-correlated (region-based) features to give LOWER
    policy sensitivity to reward direction than envs.mdp.random_mdp (motivated by
    notes/phase2_end_to_end_finding.md). A 20-seed measurement showed this is FALSE
    for this specific hard-Voronoi-boundary construction: gridworld's mean
    sensitivity (0.176, std 0.209) was actually HIGHER and much more variable than
    random_mdp's (0.143, std 0.060) at a 14-degree perturbation -- likely because
    hard region boundaries concentrate ALL the sensitivity at boundary cells rather
    than smoothing it out, and regions are otherwise perfectly homogeneous inside.
    This test only checks the env is structurally valid and does NOT assert a
    sensitivity ordering that isn't actually true. A smoother alternative (soft,
    distance-based features rather than hard region indicators, as in Objectworld's
    "distance to nearest object of each color") is a follow-up if a genuinely
    lower-sensitivity testbed is still needed -- not built yet."""
    rng = np.random.default_rng(0)
    mdp, _ = build_region_gridworld(size=6, n_regions=5, feature_dim=4, gamma=0.9, rng=rng)
    theta0 = rng.uniform(0.5, 2.0, size=mdp.d)
    theta0 /= np.linalg.norm(theta0)
    _, _, pol0 = value_iteration(mdp, theta0)
    assert pol0.shape == (mdp.S,)

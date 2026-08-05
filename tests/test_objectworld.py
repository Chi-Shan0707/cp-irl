import numpy as np
import pytest

from envs.objectworld import build_objectworld
from envs.mdp import value_iteration


@pytest.mark.parametrize("seed", range(5))
def test_objectworld_is_valid_mdp(seed):
    rng = np.random.default_rng(seed)
    mdp = build_objectworld(size=6, n_colors=3, n_objects_per_color=2, gamma=0.9, rng=rng)
    assert np.allclose(mdp.P.sum(axis=2), 1.0)
    assert mdp.S == 36
    assert mdp.d == 3


def test_objectworld_features_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    mdp = build_objectworld(size=6, n_colors=3, n_objects_per_color=2, gamma=0.9, rng=rng)
    assert np.all(mdp.phi >= -1e-9)
    assert np.all(mdp.phi <= 1.0 + 1e-9)


def test_objectworld_policy_sensitivity_measured():
    """Compare policy sensitivity to reward direction against gridworld's earlier
    measurement (notes/phase5_gridworld_result.md), which found the region-based
    gridworld's hard boundaries did NOT reduce sensitivity vs random_mdp. Report,
    don't assert an unproven direction -- see the note this test's result feeds."""
    rng = np.random.default_rng(0)
    mdp = build_objectworld(size=6, n_colors=4, n_objects_per_color=2, gamma=0.9, rng=rng)

    def sensitivity(mdp, angle_deg, rng):
        theta0 = rng.uniform(0.5, 2.0, size=mdp.d)
        theta0 /= np.linalg.norm(theta0)
        _, _, pol0 = value_iteration(mdp, theta0)
        v = rng.normal(size=mdp.d)
        v = v - (v @ theta0) * theta0
        v /= np.linalg.norm(v)
        ang = np.radians(angle_deg)
        theta_rot = np.cos(ang) * theta0 + np.sin(ang) * v
        _, _, pol_rot = value_iteration(mdp, theta_rot)
        return np.mean(pol0 != pol_rot)

    rng2 = np.random.default_rng(1)
    sens = np.mean([sensitivity(mdp, 14, rng2) for _ in range(10)])
    assert 0.0 <= sens <= 1.0  # just confirm it's measurable; magnitude reported in notes

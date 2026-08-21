import numpy as np
import pytest

from conformal.calibrate import conformal_calibrate
from envs.mdp import random_mdp, value_iteration


def _fixture():
    rng = np.random.default_rng(0)
    mdp = random_mdp(3, 2, 2, gamma=0.9, rng=rng)
    theta = np.array([1.0, 0.0])
    _, _, policy = value_iteration(mdp, theta)
    return mdp, theta, policy


def test_high_coverage_uses_whole_ball_sentinel():
    mdp, theta, policy = _fixture()
    alpha, scores = conformal_calibrate(mdp, [policy], theta, gamma=0.9)
    assert alpha == pytest.approx(np.pi)
    assert scores.shape == (1,)


def test_calibration_rejects_nonunit_center():
    mdp, _, policy = _fixture()
    with pytest.raises(ValueError, match="unit"):
        conformal_calibrate(mdp, [policy], np.array([0.5, 0.0]), gamma=0.5)


def test_calibration_rejects_empty_split():
    mdp, theta, _ = _fixture()
    with pytest.raises(ValueError, match="at least one"):
        conformal_calibrate(mdp, [], theta, gamma=0.5)

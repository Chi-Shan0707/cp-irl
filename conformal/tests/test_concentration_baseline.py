"""Sanity + efficiency comparison: the DKW concentration baseline must be VALID
(coverage >= gamma) but is expected to be looser (larger alpha, i.e. less efficient)
than exact split-conformal calibration -- this is the point of including it as the
"honest robust baseline" in plan.md Phase 5.
"""
import numpy as np
import pytest

from conformal.concentration_baseline import concentration_calibrate, dkw_quantile_threshold


def test_dkw_threshold_in_valid_range():
    rng = np.random.default_rng(0)
    c_ks = rng.uniform(0.5, 1.0, size=100)
    thresh = dkw_quantile_threshold(c_ks, gamma=0.8, delta=0.05)
    assert -1.0 <= thresh <= 1.0


@pytest.mark.parametrize("seed", range(10))
def test_concentration_alpha_is_looser_than_conformal(seed):
    """For the same nominal gamma, the DKW-based alpha should typically be >= the
    exact conformal alpha, since DKW pays an extra sqrt(log(2/delta)/(2N)) margin
    for simultaneous (uniform-in-x) validity that split-conformal's direct order-
    statistic argument doesn't need."""
    from envs.mdp import random_mdp, value_iteration
    from irl.point_estimate import classic_irl
    from conformal.calibrate import conformal_calibrate

    rng = np.random.default_rng(seed)
    S, A, d = 6, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d)

    def gen_pop(N):
        pols = []
        for _ in range(N):
            p = rng.uniform(0.5, 2.0, size=d)
            eps = rng.normal(0, 0.5, size=d)
            th = np.maximum(theta_star * p + eps, 0.0) + 0.1
            _, _, pi = value_iteration(mdp, th)
            pols.append(pi)
        return pols

    pols_train = gen_pop(60)
    pols_val = gen_pop(60)
    theta_bar = classic_irl(mdp, pols_train)

    alpha_conformal, c_ks = conformal_calibrate(mdp, pols_val, theta_bar, gamma=0.8, n_jobs=4)
    alpha_concentration = concentration_calibrate(c_ks, gamma=0.8, delta=0.05)

    assert alpha_concentration >= alpha_conformal - 1e-6, (
        f"seed={seed}: expected concentration baseline (alpha={alpha_concentration:.4f}) "
        f">= conformal (alpha={alpha_conformal:.4f})"
    )

"""Correctness checks for classic IO and the conformal (Theorem 1) calibration."""
import numpy as np
import pytest

from cio.network import build_layered_network
from cio.shortest_path import solve_forward
from cio.io_pipeline import Demonstration, generate_population, classic_io, _c_k, conformal_calibrate


def _small_net():
    return build_layered_network(m_s=2, w1=4, w2=4, m_t=2)


def test_c_k_of_own_theta_hat_is_at_least_projection():
    """theta_hat itself (normalized) is always feasible for xhat's ΘOPT set (it's the
    reward that generated xhat), so c_k >= (theta_hat/||theta_hat||) . theta_bar for
    ANY theta_bar — in particular this must hold, giving a cheap correctness floor."""
    rng = np.random.default_rng(0)
    net = _small_net()
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    demos = generate_population(net, theta_star, N=5, rng=rng)

    theta_bar = rng.normal(size=net.d)
    theta_bar /= np.linalg.norm(theta_bar)

    for dem in demos:
        c = _c_k(net=net, dem=dem, theta_bar=theta_bar)
        theta_hat_unit = dem.theta_hat / np.linalg.norm(dem.theta_hat)
        floor = theta_hat_unit @ theta_bar
        assert c >= floor - 1e-4, f"c_k={c} below feasibility floor {floor}"


def test_c_k_is_at_most_one():
    rng = np.random.default_rng(1)
    net = _small_net()
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    demos = generate_population(net, theta_star, N=5, rng=rng)
    theta_bar = rng.normal(size=net.d)
    theta_bar /= np.linalg.norm(theta_bar)
    for dem in demos:
        c = _c_k(net, dem, theta_bar)
        assert c <= 1.0 + 1e-6


def test_classic_io_recovers_direction_with_low_noise():
    """With near-zero perception noise, classic IO's point estimate should be close
    (in cosine similarity) to the (normalized) ground truth theta*."""
    rng = np.random.default_rng(2)
    net = build_layered_network(m_s=2, w1=5, w2=5, m_t=2)
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    demos = generate_population(net, theta_star, N=60, rng=rng, noise_std=0.01, eps0=0.0)
    theta_bar = classic_io(net, demos)

    cos_sim = theta_bar @ theta_star_unit / np.linalg.norm(theta_bar)
    assert cos_sim > 0.9, f"cos similarity {cos_sim} too low for near-noiseless data"


def test_conformal_calibration_alpha_in_range():
    rng = np.random.default_rng(3)
    net = _small_net()
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    demos_train = generate_population(net, theta_star, N=20, rng=rng)
    demos_val = generate_population(net, theta_star, N=20, rng=rng)
    theta_bar = classic_io(net, demos_train)

    alpha, c_ks = conformal_calibrate(net, demos_val, theta_bar, gamma=0.8)
    assert 0.0 <= alpha <= np.pi
    assert len(c_ks) == len(demos_val)


def test_conformal_alpha_shrinks_as_gamma_decreases():
    """Smaller target coverage gamma -> smaller (less conservative) uncertainty set."""
    rng = np.random.default_rng(4)
    net = _small_net()
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    demos_train = generate_population(net, theta_star, N=20, rng=rng)
    demos_val = generate_population(net, theta_star, N=30, rng=rng)
    theta_bar = classic_io(net, demos_train)

    alpha_hi, _ = conformal_calibrate(net, demos_val, theta_bar, gamma=0.95)
    alpha_lo, _ = conformal_calibrate(net, demos_val, theta_bar, gamma=0.5)
    assert alpha_lo <= alpha_hi + 1e-9

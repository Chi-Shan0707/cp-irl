"""Phase 5 ablation (plan.md Sec 6): bounded-rationality margin sweep. Demonstrators
are Boltzmann-rational (temperature `temp` controls how close to exactly-optimal
their realized, sampled action is at each state) rather than exact optimizers --
CIO's own Remark 3 notes real demonstrators may be bounded-rational, and this tests
how the calibration pipeline responds as demonstrations move away from exact
optimality. temp -> infinity recovers exact optimization; temp -> 0 is uniform
random.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_phase5_rationality_ablation.py
"""
from __future__ import annotations

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def sample_boltzmann_policy(mdp, theta, temp, rng):
    """Deterministic policy formed by sampling each state's action from the
    temp-scaled softmax over Q(s, .) -- a realized bounded-rational demonstration."""
    _, Q, _ = value_iteration(mdp, theta)
    m = (temp * Q).max(axis=1, keepdims=True)
    probs = np.exp(temp * Q - m)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(mdp.A, p=probs[s]) for s in range(mdp.S)])


def gen_population(mdp, theta_star, N, rng, temp, noise_std=0.5, eps0=0.05):
    policies = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        pi = sample_boltzmann_policy(mdp, theta_hat, temp, rng)
        policies.append(pi)
    return policies


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, temp, gamma_target=0.8, size=6, n_regions=4, feature_dim=4, N=30):
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, N, rng, temp)
    pols_val = gen_population(mdp, theta_star, N, rng, temp)
    theta_bar = classic_irl(mdp, pols_train)

    # c_k may be infeasible for a bounded-rational (non-Bellman-optimal) observed
    # policy; treat infeasible points as never-coverable (c_k = -1) rather than
    # crashing the calibration -- an explicit, documented margin-of-error handling.
    c_ks = []
    for pi in pols_val:
        try:
            c_ks.append(c_k_fast(mdp, pi, theta_bar))
        except Exception:
            c_ks.append(-1.0)
    c_ks = np.array(c_ks)
    tau = int(np.ceil(gamma_target * (len(c_ks) + 1)))
    tau = min(max(tau, 1), len(c_ks))
    c_tau = np.clip(np.sort(c_ks)[::-1][tau - 1], -1.0, 1.0)
    alpha = np.arccos(c_tau)

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_classic = v_star - occ_value(mdp, d_classic, theta_star_unit)
    aog_conformal = v_star - occ_value(mdp, d_conformal, theta_star_unit)
    n_infeasible = int(np.sum(c_ks < -0.999))
    return alpha, aog_classic, aog_conformal, n_infeasible


def main():
    temps = [0.5, 2.0, 5.0, 10.0, 30.0]
    seeds = list(range(5))

    print(f"{'temp':>6} {'mean alpha':>11} {'mean AOG_classic':>17} "
          f"{'mean AOG_conformal':>19} {'mean n_infeasible':>18}")
    for temp in temps:
        alphas, aog_cs, aog_rs, n_infs = [], [], [], []
        for seed in seeds:
            a, ac, ar, ninf = run_once(seed, temp)
            alphas.append(a)
            aog_cs.append(ac)
            aog_rs.append(ar)
            n_infs.append(ninf)
        print(f"{temp:>6.1f} {np.mean(alphas):>11.4f} {np.mean(aog_cs):>17.4f} "
              f"{np.mean(aog_rs):>19.4f} {np.mean(n_infs):>18.1f}")


if __name__ == "__main__":
    main()

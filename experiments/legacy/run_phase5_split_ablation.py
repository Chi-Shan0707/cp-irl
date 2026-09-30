"""Phase 5 ablation (plan.md Sec 6): train/val split ratio sweep, fixed total N=60,
gridworld env, LP-IRL, gamma_target=0.8.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_phase5_split_ablation.py
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
TOTAL_N = 60


def gen_population(mdp, theta_star, N, rng, noise_std=0.5, eps0=0.05):
    policies = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, val_frac, gamma_target=0.8, size=6, n_regions=4, feature_dim=4):
    N_val = int(TOTAL_N * val_frac)
    N_train = TOTAL_N - N_val
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, max(N_train, 3), rng)
    pols_val = gen_population(mdp, theta_star, max(N_val, 3), rng)
    theta_bar = classic_irl(mdp, pols_train)
    alpha, _ = conformal_calibrate(mdp, pols_val, theta_bar, gamma_target, n_jobs=N_JOBS)

    pols_cov = gen_population(mdp, theta_star, 30, rng)
    c_ks_cov = np.array([c_k_fast(mdp, pi, theta_bar) for pi in pols_cov])
    coverage = float(np.mean(c_ks_cov >= np.cos(alpha) - 1e-9))

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_classic = v_star - occ_value(mdp, d_classic, theta_star_unit)
    aog_conformal = v_star - occ_value(mdp, d_conformal, theta_star_unit)
    return alpha, coverage, aog_classic, aog_conformal


def main():
    val_fracs = [0.2, 0.35, 0.5, 0.65, 0.8]
    seeds = list(range(5))

    print(f"{'val_frac':>9} {'mean alpha':>11} {'mean coverage':>14} "
          f"{'mean AOG_classic':>17} {'mean AOG_conformal':>19}")
    for val_frac in val_fracs:
        alphas, covs, aog_cs, aog_rs = [], [], [], []
        for seed in seeds:
            a, c, ac, ar = run_once(seed, val_frac)
            alphas.append(a)
            covs.append(c)
            aog_cs.append(ac)
            aog_rs.append(ar)
        print(f"{val_frac:>9.2f} {np.mean(alphas):>11.4f} {np.mean(covs):>14.3f} "
              f"{np.mean(aog_cs):>17.4f} {np.mean(aog_rs):>19.4f}")


if __name__ == "__main__":
    main()

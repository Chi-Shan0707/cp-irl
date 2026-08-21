"""Does value-gap (Euclidean-ball) calibration reverse the paper's Q2 negative
result (angular-cap CP-IRL increases true-reward regret vs. the plain point
estimate, in every generic environment tested)? See
notes/2026-08-09_value_gap_redesign.md Sec 5.

Same region-gridworld generator, demonstrator-population sampler, and three point
estimators as experiments/run_phase5_gridworld.py, extended with a fourth decision
rule (value-ball CP-IRL) and a value-gap coverage check. Reports AOG for all four
decision rules per estimator, honestly, whichever way it comes out.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_value_gap_comparison.py
"""
from __future__ import annotations

import time

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from irl.bayesian_irl import bayesian_irl
from conformal.calibrate import conformal_calibrate
from conformal.value_gap import calibrate_value_gap, value_gap_score, ball_radius
from robust.mdp_robust import solve_robust_mdp
from robust.value_ball_robust import solve_robust_mdp_ball

N_JOBS = 8


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


POINT_ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
    "Bayesian": lambda mdp, pols: bayesian_irl(mdp, pols, n_samples=400, burn_in=150,
                                                rng=np.random.default_rng(0)),
}


def run_once(seed, gamma_target=0.8, N_train=30, N_val=30, N_test=30,
             size=6, n_regions=4, feature_dim=4):
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, N_train, rng)
    pols_val = gen_population(mdp, theta_star, N_val, rng)

    _, v_star = occupancy_lp(mdp, theta_star_unit)

    rows = []
    for est_name, estimator in POINT_ESTIMATORS.items():
        t0 = time.time()
        theta_bar = estimator(mdp, pols_train)
        t_fit = time.time() - t0
        theta_bar_unit = theta_bar / max(np.linalg.norm(theta_bar), 1e-12)

        # --- angular-cap CP-IRL (existing) ---
        alpha_conf, _ = conformal_calibrate(mdp, pols_val, theta_bar_unit, gamma_target,
                                             n_jobs=N_JOBS)
        d_angular, _ = solve_robust_mdp(mdp, theta_bar_unit, alpha_conf)

        # --- value-ball CP-IRL (new) ---
        q_gamma, _ = calibrate_value_gap(mdp, pols_val, theta_bar, gamma_target)
        rho = ball_radius(q_gamma, mdp)
        rho_capped = rho if np.isfinite(rho) else 10.0 * np.linalg.norm(theta_bar)
        d_ball, _ = solve_robust_mdp_ball(mdp, theta_bar, rho_capped)

        # --- plain point estimate ---
        d_classic, _ = occupancy_lp(mdp, theta_bar)

        # --- value-gap coverage check on a fresh population ---
        pols_cov = gen_population(mdp, theta_star, N_test, rng)
        v_star_theta_bar, _, _ = value_iteration(mdp, theta_bar)
        v_star_theta_bar_s0 = float(mdp.mu0 @ v_star_theta_bar)
        c_cov = np.array([value_gap_score(mdp, pi, theta_bar) for pi in pols_cov])
        value_gap_coverage = float(np.mean(c_cov <= q_gamma + 1e-9))

        for policy_name, d_occ in [("point_estimate", d_classic),
                                    ("angular_cap", d_angular),
                                    ("value_ball", d_ball)]:
            aog = v_star - occ_value(mdp, d_occ, theta_star_unit)
            rows.append(dict(seed=seed, estimator=est_name, policy=policy_name,
                              aog=aog, alpha=alpha_conf, rho=rho,
                              value_gap_coverage=value_gap_coverage, t_fit=t_fit))

    return rows


def main():
    seeds = list(range(8))
    all_rows = []
    t0 = time.time()
    for seed in seeds:
        all_rows.extend(run_once(seed))
    elapsed = time.time() - t0
    print(f"Total: {elapsed:.1f}s for {len(seeds)} seeds\n")

    print(f"{'estimator':>10} {'policy':>16} {'mean AOG':>10} {'std AOG':>10} "
          f"{'mean alpha':>11} {'mean rho':>10} {'val-gap cov':>12}")
    combos = sorted(set((r["estimator"], r["policy"]) for r in all_rows))
    for est, pol in combos:
        subset = [r for r in all_rows if r["estimator"] == est and r["policy"] == pol]
        aogs = [r["aog"] for r in subset]
        alphas = [r["alpha"] for r in subset]
        rhos = [r["rho"] for r in subset if np.isfinite(r["rho"])]
        covs = [r["value_gap_coverage"] for r in subset]
        rho_str = f"{np.mean(rhos):.4f}" if rhos else "inf"
        print(f"{est:>10} {pol:>16} {np.mean(aogs):>10.4f} {np.std(aogs):>10.4f} "
              f"{np.mean(alphas):>11.4f} {rho_str:>10} {np.mean(covs):>12.3f}")

    print("\nPer-estimator: does value_ball beat angular_cap? Does either beat point_estimate?")
    for est in POINT_ESTIMATORS:
        pe = np.mean([r["aog"] for r in all_rows if r["estimator"] == est and r["policy"] == "point_estimate"])
        ac = np.mean([r["aog"] for r in all_rows if r["estimator"] == est and r["policy"] == "angular_cap"])
        vb = np.mean([r["aog"] for r in all_rows if r["estimator"] == est and r["policy"] == "value_ball"])
        print(f"  {est:>10}: point_estimate={pe:.4f}  angular_cap={ac:.4f} ({'worse' if ac>pe else 'better'})  "
              f"value_ball={vb:.4f} ({'worse' if vb>pe else 'better'})")


if __name__ == "__main__":
    main()

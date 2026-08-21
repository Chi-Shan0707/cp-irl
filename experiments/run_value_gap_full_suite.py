"""Full-rigor version of run_value_gap_comparison.py: does value-gap (Euclidean-ball)
calibration reverse the paper's Q2 negative result, across all three environments the
paper itself uses (random tabular MDP, region gridworld, Objectworld), all three
point estimators, 30 seeds (not 8), with a DKW concentration baseline for the
value-gap score (the value-space analogue of the paper's existing DKW-vs-conformal
comparison for the angular score) and a vacuous-set (noise-sweep) check for the
calibrated ball radius. See notes/2026-08-09_value_gap_redesign.md.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_value_gap_full_suite.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.objectworld import build_objectworld
from envs.mdp import TabularMDP, random_mdp, value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from irl.bayesian_irl import bayesian_irl
from conformal.calibrate import conformal_calibrate
from conformal.value_gap import (
    calibrate_value_gap, dkw_calibrate_value_gap, value_gap_score, ball_radius,
)
from robust.mdp_robust import solve_robust_mdp
from robust.value_ball_robust import solve_robust_mdp_ball

N_JOBS = 8
GAMMA_TARGET = 0.8
N_SEEDS = 30

POINT_ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
    "Bayesian": lambda mdp, pols: bayesian_irl(mdp, pols, n_samples=400, burn_in=150,
                                                rng=np.random.default_rng(0)),
}


def build_random_mdp_env(rng):
    mdp = random_mdp(6, 3, 3, gamma=0.9, rng=rng)
    return mdp


def build_gridworld_env(rng):
    mdp, _ = build_region_gridworld(size=6, n_regions=4, feature_dim=4, gamma=0.9, rng=rng)
    return mdp


def build_objectworld_env(rng):
    mdp = build_objectworld(size=6, n_colors=4, n_objects_per_color=2, gamma=0.9, rng=rng)
    return mdp


ENV_BUILDERS = {
    "random_mdp": build_random_mdp_env,
    "gridworld": build_gridworld_env,
    "objectworld": build_objectworld_env,
}


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


_ENV_SEED_OFFSET = {"random_mdp": 0, "gridworld": 100_000, "objectworld": 200_000}


def run_once(env_name, seed, N_train=30, N_val=30, N_test=30):
    rng = np.random.default_rng(_ENV_SEED_OFFSET[env_name] + seed)
    mdp = ENV_BUILDERS[env_name](rng)
    theta_star = rng.uniform(0.5, 2.0, size=mdp.d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, N_train, rng)
    pols_val = gen_population(mdp, theta_star, N_val, rng)

    _, v_star = occupancy_lp(mdp, theta_star_unit)

    rows = []
    for est_name, estimator in POINT_ESTIMATORS.items():
        theta_bar = estimator(mdp, pols_train)
        theta_bar_unit = theta_bar / max(np.linalg.norm(theta_bar), 1e-12)

        # angular-cap CP-IRL (existing)
        alpha_conf, _ = conformal_calibrate(mdp, pols_val, theta_bar_unit, GAMMA_TARGET,
                                             n_jobs=N_JOBS)
        d_angular, _ = solve_robust_mdp(mdp, theta_bar_unit, alpha_conf)

        # value-ball CP-IRL, exact conformal quantile (new)
        q_gamma, c_ks_vg = calibrate_value_gap(mdp, pols_val, theta_bar, GAMMA_TARGET)
        rho = ball_radius(q_gamma, mdp)
        rho_capped = rho if np.isfinite(rho) else 10.0 * np.linalg.norm(theta_bar)
        d_ball, _ = solve_robust_mdp_ball(mdp, theta_bar, rho_capped)

        # value-ball CP-IRL, DKW baseline (new, the value-gap analogue of the
        # paper's existing angular DKW-vs-conformal comparison)
        q_dkw = dkw_calibrate_value_gap(c_ks_vg, GAMMA_TARGET, delta=0.05)
        rho_dkw = ball_radius(q_dkw, mdp)
        rho_dkw_capped = rho_dkw if np.isfinite(rho_dkw) else 10.0 * np.linalg.norm(theta_bar)
        d_ball_dkw, _ = solve_robust_mdp_ball(mdp, theta_bar, rho_dkw_capped)

        # plain point estimate
        d_classic, _ = occupancy_lp(mdp, theta_bar)

        # value-gap coverage check on a fresh population
        pols_cov = gen_population(mdp, theta_star, N_test, rng)
        c_cov = np.array([value_gap_score(mdp, pi, theta_bar) for pi in pols_cov])
        value_gap_coverage = float(np.mean(c_cov <= q_gamma + 1e-9))

        for policy_name, d_occ in [("point_estimate", d_classic),
                                    ("angular_cap", d_angular),
                                    ("value_ball", d_ball),
                                    ("value_ball_dkw", d_ball_dkw)]:
            aog = v_star - occ_value(mdp, d_occ, theta_star_unit)
            rows.append(dict(env=env_name, seed=seed, estimator=est_name, policy=policy_name,
                              aog=aog, alpha=alpha_conf, rho=rho, rho_dkw=rho_dkw,
                              value_gap_coverage=value_gap_coverage))

    return rows


def _gen_population_unified_noise(mdp, theta_star, N, rng, spread):
    """Unified-noise population generator (irl/tests/test_efficiency_t6.py's fix,
    reused verbatim): gen_population's default multiplicative p~U(0.5,2.0) is an
    ALWAYS-PRESENT noise source independent of noise_std, so sweeping noise_std
    alone plateaus at a nonzero floor instead of vanishing (confirmed below on a
    first attempt with the un-unified generator -- same pathology T6 already
    diagnosed and fixed for the angular alpha). Here every noise source shares one
    `spread` parameter, so theta_hat -> theta_star exactly as spread -> 0.
    """
    policies = []
    for _ in range(N):
        p = 1.0 + rng.uniform(-spread, spread, size=theta_star.shape)
        eps = rng.normal(0, spread, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + 0.01 * spread
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def noise_sweep_vacuousness_check(env_name="gridworld", n_seeds=8, estimator_name="MaxEnt"):
    """Vacuous-set check for the value-ball radius: does rho shrink to 0 as
    demonstrator-population noise vanishes, mirroring the paper's existing T6-style
    unified-noise sweep for the angular alpha (Q1). Uses MaxEnt (not LP-IRL): the
    exact-fit LP estimator degenerates to c_k=0 for nearly all validation
    demonstrators even at HIGH noise on this low-feature-diversity gridworld (the
    same exact-fit tie-breaking pathology behind the paper's retracted degenerate
    separation example, \\S\\ref{sec:degenerate-diagnostic}), which saturates the
    check at its floor from the start and makes it uninformative about noise.
    """
    spreads = [1.0, 0.5, 0.2, 0.1, 0.05]
    estimator = POINT_ESTIMATORS[estimator_name]
    results = {}
    for spread in spreads:
        rhos = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(5000 + seed)
            mdp = ENV_BUILDERS[env_name](rng)
            theta_star = rng.uniform(0.5, 2.0, size=mdp.d)
            pols_train = _gen_population_unified_noise(mdp, theta_star, 30, rng, spread)
            pols_val = _gen_population_unified_noise(mdp, theta_star, 30, rng, spread)
            theta_bar = estimator(mdp, pols_train)
            q_gamma, _ = calibrate_value_gap(mdp, pols_val, theta_bar, GAMMA_TARGET)
            rho = ball_radius(q_gamma, mdp)
            if np.isfinite(rho):
                rhos.append(rho)
        results[spread] = float(np.mean(rhos)) if rhos else float("nan")
    return results


def main():
    t0 = time.time()
    all_rows = []
    for env_name in ENV_BUILDERS:
        for seed in range(N_SEEDS):
            all_rows.extend(run_once(env_name, seed))
        print(f"  ...finished env={env_name} ({time.time()-t0:.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s for {N_SEEDS} seeds x {len(ENV_BUILDERS)} envs\n")

    with open("experiments/value_gap_full_suite_results.json", "w") as f:
        json.dump(all_rows, f)

    print(f"{'env':>12} {'estimator':>10} {'policy':>16} {'mean AOG':>10} {'std AOG':>10} "
          f"{'val-gap cov':>12}")
    combos = sorted(set((r["env"], r["estimator"], r["policy"]) for r in all_rows))
    for env, est, pol in combos:
        subset = [r for r in all_rows if r["env"] == env and r["estimator"] == est and r["policy"] == pol]
        aogs = [r["aog"] for r in subset]
        covs = [r["value_gap_coverage"] for r in subset]
        print(f"{env:>12} {est:>10} {pol:>16} {np.mean(aogs):>10.4f} {np.std(aogs):>10.4f} "
              f"{np.mean(covs):>12.3f}")

    print("\nSummary: point_estimate vs angular_cap vs value_ball vs value_ball_dkw")
    for env in ENV_BUILDERS:
        for est in POINT_ESTIMATORS:
            pe = np.mean([r["aog"] for r in all_rows if r["env"] == env and r["estimator"] == est and r["policy"] == "point_estimate"])
            ac = np.mean([r["aog"] for r in all_rows if r["env"] == env and r["estimator"] == est and r["policy"] == "angular_cap"])
            vb = np.mean([r["aog"] for r in all_rows if r["env"] == env and r["estimator"] == est and r["policy"] == "value_ball"])
            vbd = np.mean([r["aog"] for r in all_rows if r["env"] == env and r["estimator"] == est and r["policy"] == "value_ball_dkw"])
            print(f"  {env:>12} {est:>10}: point={pe:.4f}  angular={ac:.4f} "
                  f"(x{ac/max(pe,1e-9):.2f})  value_ball={vb:.4f} (x{vb/max(pe,1e-9):.2f})  "
                  f"value_ball_dkw={vbd:.4f} (x{vbd/max(pe,1e-9):.2f})")

    print("\nVacuous-set check (value-ball radius vs. population noise, gridworld, 8 seeds):")
    sweep = noise_sweep_vacuousness_check()
    for noise, rho in sorted(sweep.items(), reverse=True):
        print(f"  noise_std={noise:>5.2f}  mean rho={rho:.4f}")
    with open("experiments/value_gap_noise_sweep.json", "w") as f:
        json.dump(sweep, f)


if __name__ == "__main__":
    main()

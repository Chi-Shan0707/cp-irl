"""Scope check for Theorem 4.3 (Separation): does the unbounded-AOG failure mode of
classic point-estimate IRL in the Example-1 MDP (irl/example1_mdp.py) depend on the
specific LP sub-optimality-loss estimator (irl/point_estimate.py::classic_irl), or
is it a generic property of collapsing a heterogeneous demonstrator population to a
single point estimate?

Runs THREE point estimators -- LP-IRL (classic_irl), MaxEnt IRL (maxent_irl), and
Bayesian IRL posterior mean (bayesian_irl) -- through the SAME Example-1 MDP
construction used by experiments/run_t5_separation.py, each followed by the plain
(non-robust) decision under its point estimate, and reports AOG as u grows.

This addresses a specific reviewer concern: Theorem 4.3 is proved only for the LP
estimator's degenerate tie-breaking on the facet x1 + u*x2 = u; whether MaxEnt/
Bayesian IRL exhibit the same qualitative failure is left open in the paper. This
script provides the missing empirical evidence (not a proof) either way.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_separation_estimator_scope.py
"""
from __future__ import annotations

import numpy as np

from envs.mdp import value_iteration, occupancy_lp
from irl.example1_mdp import (
    build_example1_mdp, sample_theta_hat_example1, THETA_STAR_EXAMPLE1,
)
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from irl.bayesian_irl import bayesian_irl

ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=200),
    "Bayesian": lambda mdp, pols: bayesian_irl(
        mdp, pols, n_samples=600, burn_in=200, rng=np.random.default_rng(0)),
}


def gen_population(mdp, N, rng):
    policies = []
    for _ in range(N):
        theta_hat = sample_theta_hat_example1(rng)
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(u, seed, N_train=1000):
    rng = np.random.default_rng(seed)
    mdp = build_example1_mdp(u)
    theta_star_unit = THETA_STAR_EXAMPLE1 / np.linalg.norm(THETA_STAR_EXAMPLE1)
    pols_train = gen_population(mdp, N_train, rng)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aogs = {}
    for name, estimator in ESTIMATORS.items():
        theta_bar = estimator(mdp, pols_train)
        d_hat, _ = occupancy_lp(mdp, theta_bar)
        aogs[name] = v_star - occ_value(mdp, d_hat, theta_star_unit)
    return aogs


def main():
    us = [2, 5, 10, 25, 50, 100]
    seeds = list(range(10))
    names = list(ESTIMATORS.keys())

    print(f"{'u':>5}" + "".join(f"{n + '_mean':>14}{n + '_std':>12}" for n in names))
    results = {}
    for u in us:
        per_est = {n: [] for n in names}
        for seed in seeds:
            aogs = run_once(u, seed)
            for n in names:
                per_est[n].append(aogs[n])
        row = f"{u:>5}"
        for n in names:
            vals = np.array(per_est[n])
            row += f"{vals.mean():>14.4f}{vals.std():>12.4f}"
        print(row)
        results[u] = {n: (float(np.mean(per_est[n])), float(np.std(per_est[n])))
                       for n in names}

    print("\nInterpretation: if a column's mean AOG grows with u, that estimator "
          "shares the LP estimator's degenerate-tie-break failure mode; if it stays "
          "flat/bounded, Theorem 4.3's failure mode is specific to LP sub-optimality "
          "loss and does not generalize to that estimator.")
    return results


if __name__ == "__main__":
    main()

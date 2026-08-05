"""RLHF-relabeling demonstration (paper/draft.md Sec 7, "Discussion and
Limitations" -- flagged as "we believe transfers directly but have not run").

Claim being tested: the per-demonstrator exchangeable-unit design (plan.md P1,
unit A) is structurally identical to "one annotator, one calibration point" in a
heterogeneous-preference RLHF population -- no new machinery is needed, only a
relabeling of what the population represents. This script demonstrates that
literally: it reuses irl/point_estimate.py, conformal/calibrate.py, and
robust/mdp_robust.py UNCHANGED, with the only difference from earlier experiments
being the interpretation of the population (each "demonstrator" is now an
"annotator" whose revealed preference reflects their own idiosyncratic reward
weighting over response features) and the coverage statement's framing.

Setting: a small "response selection" MDP -- a single decision state where the
policy must choose among candidate responses (actions), each with a feature
vector (e.g. helpfulness, verbosity, safety-caution -- generic reward-relevant
attributes), reward = theta . phi(response). A population of N annotators, each
with their own idiosyncratic preference weighting theta_hat_k over these
attributes, each "labels" (selects) their preferred response among the
candidates -- exactly the CIO/CP-IRL calibration data model, relabeled.

The resulting guarantee, stated in the annotator population's terms: the deployed
response-selection policy is calibrated so that a randomly drawn NEW annotator's
own preferred response is covered by the calibrated reward set with probability
>= gamma, without ever observing that annotator's reward weighting directly --
i.e. "the policy is provably aligned with at least a gamma-fraction of the
annotator population's own (unobserved) preferences."

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_rlhf_relabeling_demo.py
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP, value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def build_response_selection_mdp(n_responses: int, n_attributes: int, gamma: float,
                                  rng: np.random.Generator) -> TabularMDP:
    """Single decision state; each action = selecting one candidate response;
    features = that response's attribute vector (helpfulness, verbosity, safety-
    caution, etc. -- generic, not meaningfully labeled here, just distinct
    directions in attribute space)."""
    P = np.ones((1, n_responses, 1))
    phi = rng.uniform(0.2, 1.0, size=(1, n_responses, n_attributes))
    mu0 = np.array([1.0])
    return TabularMDP(P, phi, gamma, mu0)


def gen_annotator_population(mdp, theta_star, N, rng, noise_std=0.4, eps0=0.05):
    """Each annotator's preference weighting theta_hat_k is a noisy perturbation
    of the population's central tendency theta_star; their 'label' is which
    response they'd select under their own weighting -- an idiosyncratic,
    individually-rational choice, never a direct observation of theta_hat_k
    itself, exactly mirroring the IRL demonstrator population model."""
    annotations = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, chosen = value_iteration(mdp, theta_hat)
        annotations.append(chosen)
    return annotations


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, gamma_target, n_responses=6, n_attributes=4, N=40):
    rng = np.random.default_rng(seed)
    mdp = build_response_selection_mdp(n_responses, n_attributes, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=n_attributes)  # population's central preference
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    annot_train = gen_annotator_population(mdp, theta_star, N, rng)
    annot_val = gen_annotator_population(mdp, theta_star, N, rng)

    # theta_bar = the "reward model" fit from the annotator population
    theta_bar = classic_irl(mdp, annot_train)
    alpha, _ = conformal_calibrate(mdp, annot_val, theta_bar, gamma_target, n_jobs=N_JOBS)

    # coverage, stated in the annotator population's terms: for a FRESH annotator,
    # does the calibrated reward set contain a weighting consistent with their own
    # (never-observed) choice?
    annot_test = gen_annotator_population(mdp, theta_star, N, rng)
    c_ks = np.array([c_k_fast(mdp, a, theta_bar) for a in annot_test])
    coverage = float(np.mean(c_ks >= np.cos(alpha) - 1e-9))

    d_policy_model, _ = occupancy_lp(mdp, theta_bar)
    d_policy_cirl, _ = solve_robust_mdp(mdp, theta_bar, alpha)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_model = v_star - occ_value(mdp, d_policy_model, theta_star_unit)
    aog_cirl = v_star - occ_value(mdp, d_policy_cirl, theta_star_unit)
    return coverage, alpha, aog_model, aog_cirl


def main():
    seeds = list(range(8))
    gammas = [0.5, 0.7, 0.8, 0.9]

    print("RLHF-relabeling demo: 'annotator population' framing of CP-IRL,\n"
          "reusing irl/point_estimate.py, conformal/calibrate.py, robust/mdp_robust.py\n"
          "UNCHANGED -- only the population's interpretation is relabeled.\n")
    print(f"{'gamma':>6} {'mean coverage':>14} {'mean alpha':>11} "
          f"{'mean AOG_model':>15} {'mean AOG_cirl':>14}")
    for gamma in gammas:
        covs, alphas, aog_ms, aog_cs = [], [], [], []
        for seed in seeds:
            c, a, am, ac = run_once(seed, gamma)
            covs.append(c)
            alphas.append(a)
            aog_ms.append(am)
            aog_cs.append(ac)
        print(f"{gamma:>6.2f} {np.mean(covs):>14.3f} {np.mean(alphas):>11.4f} "
              f"{np.mean(aog_ms):>15.4f} {np.mean(aog_cs):>14.4f}")

    print("\nInterpretation: 'mean coverage' is the empirical realization of "
          "'this policy is calibrated so that a fresh annotator's own preferred "
          "response is covered by the calibrated reward set with probability "
          ">= gamma' -- the exact guarantee the RLHF relabeling motivates, "
          "achieved with zero new machinery.")


if __name__ == "__main__":
    main()

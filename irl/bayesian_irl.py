"""Bayesian IRL (Ramachandran & Amir, 2007) point estimator: posterior mean over
theta under a Boltzmann-rational demonstrator likelihood, sampled via Metropolis-
Hastings. The last of plan.md's Phase 5 baseline list ("Bayesian IRL posterior-mean")
not yet implemented.

Likelihood model (standard in the Bayesian IRL literature): demonstrator k's policy
pi_k is Boltzmann-rational w.r.t. Q_theta,
    P(pi_k | theta) = prod_s  exp(temp * Q_theta(s, pi_k(s))) / sum_a exp(temp * Q_theta(s,a)),
`temp` a rationality temperature (higher = closer to optimal). This is a genuinely
different model from irl/point_estimate.py's sub-optimality-loss LP (a point
estimate minimizing worst-case loss) and irl/maxent.py's MaxEnt IRL (matches feature
expectations exactly) -- Bayesian IRL instead maintains a full posterior and reports
its mean, providing the "principled Bayesian alternative" contrast plan.md's Phase 5
baseline list calls for.
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP, value_iteration


def log_likelihood(mdp: TabularMDP, theta: np.ndarray, policies: list[np.ndarray],
                    temp: float = 5.0) -> float:
    """Sum of per-demonstrator log-likelihoods under the Boltzmann-rational model."""
    _, Q, _ = value_iteration(mdp, theta)  # Q well-defined even off the ball boundary
    log_partition = np.log(np.exp(temp * Q - temp * Q.max(axis=1, keepdims=True)).sum(axis=1)) \
        + temp * Q.max(axis=1)
    total = 0.0
    for pi in policies:
        chosen_Q = temp * Q[np.arange(mdp.S), pi]
        total += float(np.sum(chosen_Q - log_partition))
    return total


def bayesian_irl(mdp: TabularMDP, policies: list[np.ndarray], temp: float = 5.0,
                  n_samples: int = 2000, burn_in: int = 500, step_std: float = 0.15,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Metropolis-Hastings posterior sampling over theta (uniform prior on the unit
    sphere), returning the posterior mean, L2-renormalized for consistency with the
    rest of the pipeline (scale is not identified by the likelihood alone, since Q
    and the softmax are invariant to positive rescaling combined with temp -- we
    fix scale via the unit-sphere constraint, standard in this project).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = mdp.d

    theta = rng.normal(size=d)
    theta /= np.linalg.norm(theta)
    cur_ll = log_likelihood(mdp, theta, policies, temp)

    samples = []
    for i in range(n_samples):
        proposal = theta + rng.normal(0, step_std, size=d)
        proposal /= np.linalg.norm(proposal)
        prop_ll = log_likelihood(mdp, proposal, policies, temp)
        if np.log(rng.uniform()) < prop_ll - cur_ll:
            theta, cur_ll = proposal, prop_ll
        if i >= burn_in:
            samples.append(theta.copy())

    posterior_mean = np.mean(samples, axis=0)
    norm = np.linalg.norm(posterior_mean)
    if norm > 1e-12:
        posterior_mean = posterior_mean / norm
    return posterior_mean

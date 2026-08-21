"""The "honest robust baseline" from plan.md Phase 5: a robust MDP whose uncertainty
set radius comes from a classical (non-conformal) concentration inequality on the
same c_k scores, rather than the exact finite-sample split-conformal quantile
(conformal/calibrate.py::conformal_calibrate). This is the natural point of
comparison for demonstrating CP-IRL's calibration is not just VALID but EFFICIENT
(tighter for the same confidence level) -- CIO's own related-work discussion (plan.md
Sec 2) makes exactly this contrast against Petrik & Russell's Bayesian/concentration-
based ambiguity sets for robust MDPs.

Method: Dvoretzky-Kiefer-Wolfowitz (DKW) inequality. Given N i.i.d. draws of a score
c_k (here, the calibration scores from irl/feasible_set.py::c_k), the empirical CDF
F_hat satisfies, with probability >= 1-delta over the draw of the N scores,
    sup_x |F_hat(x) - F(x)| <= eps(N, delta) := sqrt(log(2/delta) / (2N))
uniformly over x (Dvoretzky, Kiefer & Wolfowitz 1956; tight constant per Massart
1990). To guarantee the TRUE gamma-quantile of the score's distribution is covered
with confidence >= 1-delta, it suffices to take the empirical (gamma + eps)-quantile
(clipped to 1) as a conservative upper bound: since F_hat(x) >= F(x) - eps uniformly,
the point where F_hat first reaches gamma+eps has true CDF value >= gamma, i.e. is at
or below the true gamma-quantile... more precisely: with probability >= 1-delta,
F(q_hat) >= F_hat(q_hat) - eps = (gamma+eps) - eps = gamma, so the empirical
(gamma+eps)-quantile q_hat upper-bounds the true gamma-quantile of scores that are
LOWER (since we want a threshold BELOW which the true gamma-quantile lies, and we
need C(theta_bar, alpha) to be big enough to cover c_new with probability gamma, we
need alpha = arccos(the gamma-quantile of the DISTRIBUTION of c_k), so we want a
threshold that UNDER-estimates c's gamma-quantile => use the empirical
(gamma - eps)-quantile, clipped to >= 0, as the DKW-conservative threshold on c
itself (smaller c-threshold => larger alpha => more conservative/wider cap)).
"""
from __future__ import annotations

import numpy as np


def dkw_quantile_threshold(c_ks: np.ndarray, gamma: float, delta: float = 0.05) -> float:
    """DKW-conservative threshold c_tau on the c_k scores such that, with
    probability >= 1-delta (over the draw of c_ks), the TRUE population satisfies
    P(c >= c_tau) >= gamma -- i.e. c_tau conservatively estimates the (1-gamma)
    quantile of the score distribution from below (a SMALLER c_tau is more
    conservative / gives a bigger alpha, matching conformal_calibrate's convention
    where alpha = arccos(c_tau)).

    We need the strict CDF F_<(c_tau)=P(c<c_tau) to be at most 1-gamma.
    Under the DKW event, choosing an ascending order statistic whose empirical
    left limit is at most 1-gamma-eps guarantees this inequality, including for
    atomic score distributions. If that target is negative, the sentinel -1
    gives certain (but uninformative) coverage. Smaller delta -> larger eps ->
    more conservative (smaller) c_tau -> larger alpha; smaller N -> looser threshold
    (eps grows as 1/sqrt(N), vs conformal's exact O(1/N) finite-sample validity with
    no confidence-level parameter at all).
    """
    c_ks = np.asarray(c_ks, dtype=float)
    N = len(c_ks)
    if N == 0:
        raise ValueError("c_ks must contain at least one score")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie strictly between 0 and 1, got {delta}")
    eps = np.sqrt(np.log(2.0 / delta) / (2.0 * N))
    left_mass = (1.0 - gamma) - eps
    if left_mass < 0.0:
        return -1.0
    rank = int(np.floor(N * left_mass)) + 1  # one-indexed ascending rank
    return float(np.sort(c_ks)[rank - 1])


def concentration_calibrate(c_ks: np.ndarray, gamma: float,
                             delta: float = 0.05) -> float:
    """DKW-based alpha (the 'honest concentration-based robust MDP baseline'),
    analogous in role to conformal/calibrate.py::conformal_calibrate's alpha_gamma
    but built from a classical concentration inequality instead of the exact
    split-conformal quantile lemma. Returns alpha (radians).
    """
    c_tau = dkw_quantile_threshold(c_ks, gamma, delta)
    c_tau = np.clip(c_tau, -1.0, 1.0)
    return float(np.arccos(c_tau))

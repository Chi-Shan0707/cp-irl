"""The MDP analogue of CIO's degenerate Example 1 (plan.md Sec 5, T5 Separation).

CIO's Example 1 (their Sec 3.2): a shortest-path-style LP
    FO(theta, u): minimize theta_1 x_1 + theta_2 x_2
                  s.t. x_1 + u x_2 >= u, 0 <= x_1 <= u, 0 <= x_2 <= 2
with theta* = (cos(pi/4), sin(pi/4)) and demonstrator population theta_hat drawn
uniformly on the unit circle's first quadrant. Their Lemma 1 proves classic IO's
sub-optimality-loss point estimate converges (as N -> infinity) to
    theta_u := (1/sqrt(1+u^2), u/sqrt(1+u^2))
-- NOT theta* -- and Proposition 1 shows this makes AOG and POG of classic IO
UNBOUNDED as u -> infinity. Their Lemma 2 proves conformal IO (robust decision against
the calibrated cap) keeps AOG = 0 and POG bounded, for any alpha in (0, pi/2),
regardless of u.

Translating to a reward-MAXIMIZING MDP requires negating the features: CIO's problem
MINIMIZES cost, so the "safe" (theta*-preferred) choice is the one with BOUNDED
feature magnitude (x=(0,1)), and the "risky" choice (x=(u,0)) has magnitude growing
with u -- for a cost minimizer, larger u makes that choice WORSE, aligning
theta*'s preference with what a worst-case-averse (robust) decision-maker would also
avoid. If features are used as-is in a reward-MAXIMIZING MDP, growing u makes the
u-scaled action *more* attractive (since maximizing wants a LARGER dot product), which
inverts the whole mechanism -- confirmed empirically (see notes/phase2_t5_construction.md):
a naive (non-negated) translation shows classic IO staying near-perfect while robustified
IO's AOG blows up, the OPPOSITE of CIO's intended result and NOT what plan.md's T5 wants.

The fix (implemented below): reward(s, a) = -cost_CIO(s, a), i.e. phi(s, a) is the
NEGATION of CIO's cost features. Maximizing -cost is exactly minimizing cost, so this
exactly reproduces CIO's Example 1 inside the reward-maximization MDP framework used
throughout this project.
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP

THETA_STAR_EXAMPLE1 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])


def build_example1_mdp(u: float, gamma: float = 0.9) -> TabularMDP:
    """Single-state, 2-action MDP mirroring CIO's Example 1 (negated features, see
    module docstring). Action 0 <-> CIO's x=(0,1) (bounded, theta*-preferred).
    Action 1 <-> CIO's x=(u,0) (feature magnitude grows with u).
    """
    P = np.ones((1, 2, 1))
    phi = np.array([[[0.0, -1.0], [-u, 0.0]]])
    mu0 = np.array([1.0])
    return TabularMDP(P, phi, gamma, mu0)


def sample_theta_hat_example1(rng: np.random.Generator) -> np.ndarray:
    """theta_hat uniform on the unit circle, first quadrant (CIO's Theta domain in
    Example 1: {(cos delta, sin delta) : delta in (0, pi/2)})."""
    delta = rng.uniform(1e-6, np.pi / 2 - 1e-6)
    return np.array([np.cos(delta), np.sin(delta)])


def theta_u_degenerate_direction(u: float) -> np.ndarray:
    """The direction CIO's Lemma 1 proves classic IO's point estimate converges to
    (NOT theta*), for cross-checking the point-estimate implementation reproduces it."""
    return np.array([1.0, u]) / np.sqrt(1 + u ** 2)

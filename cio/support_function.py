"""Support function of the spherical-cap uncertainty set
    C(theta_bar, alpha) = {theta : ||theta||_2 = 1, theta^T theta_bar >= cos(alpha)}
(theta_bar a unit vector, alpha in (0, pi/2)), i.e.
    h_C(x) = max_{theta in C(theta_bar, alpha)} theta^T x.

Closed form derivation (verified against a cvxpy oracle in tests): decompose
x = a * theta_bar + x_perp with a = x . theta_bar and x_perp orthogonal to theta_bar,
r = ||x_perp||. Writing theta at angle gamma from theta_bar (gamma in [0, pi], aligned
in azimuth with x_perp), theta^T x = a cos(gamma) + r sin(gamma) =: f(gamma), which is
concave on [0, pi] and maximized unconstrained at gamma* = atan2(r, a) =: beta (the
angle between x and theta_bar). Restricting to gamma in [0, alpha] (the cap):
  - if beta <= alpha: the unconstrained maximizer is feasible, h_C(x) = f(beta) = ||x||.
  - if beta > alpha: f is still increasing on [0, alpha] (alpha is before the peak),
    so the max is at the boundary gamma = alpha, h_C(x) = f(alpha) = a cos(alpha)
    + r sin(alpha).
"""
from __future__ import annotations

import numpy as np


def support_value(x: np.ndarray, theta_bar: np.ndarray, alpha: float) -> float:
    """h_C(x) via the closed form above. theta_bar must be unit norm."""
    assert np.isclose(np.linalg.norm(theta_bar), 1.0, atol=1e-6)
    a = float(x @ theta_bar)
    x_perp = x - a * theta_bar
    r = float(np.linalg.norm(x_perp))
    beta = np.arctan2(r, a)
    if beta <= alpha:
        return float(np.linalg.norm(x))
    return a * np.cos(alpha) + r * np.sin(alpha)


def support_maximizer(x: np.ndarray, theta_bar: np.ndarray, alpha: float) -> np.ndarray:
    """The argmax theta achieving h_C(x), for diagnostics / sanity checks."""
    a = float(x @ theta_bar)
    x_perp = x - a * theta_bar
    r = float(np.linalg.norm(x_perp))
    beta = np.arctan2(r, a)
    norm_x = np.linalg.norm(x)
    if beta <= alpha:
        if norm_x < 1e-12:
            return theta_bar.copy()
        return x / norm_x
    gamma = alpha
    u = x_perp / r if r > 1e-12 else np.zeros_like(x)
    return np.cos(gamma) * theta_bar + np.sin(gamma) * u

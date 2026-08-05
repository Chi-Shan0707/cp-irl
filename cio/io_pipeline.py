"""Classic IO (sub-optimality loss) and conformal calibration (CIO Theorem 1) on the
shortest-path network. See plan.md Sec 1 for the mapping from the CIO paper's notation.

Theta domain: the unit ball {theta in R^d : ||theta||_2 <= 1} (no sign constraint —
the network is a DAG so the forward LP is well-posed for any real edge costs).
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .network import ShortestPathNetwork
from .shortest_path import path_edges, solve_forward


@dataclass
class Demonstration:
    theta_hat: np.ndarray  # perceived reward of this decision-maker (not observed)
    s: int
    t: int
    x_hat: np.ndarray  # observed decision (edge-flow / path indicator)


def generate_population(net: ShortestPathNetwork, theta_star: np.ndarray, N: int,
                         rng: np.random.Generator, p_range=(0.5, 2.0),
                         noise_std: float = 1.0, eps0: float = 0.1) -> list[Demonstration]:
    """Mirrors the CIO paper's Sec. 5 data-generation process:
        theta_hat_i = relu(theta_star_i * p_i + eps_i) + eps0,  p_i ~ U(p_range), eps_i ~ N(0,1)
    """
    demos = []
    od_pairs = net.all_od_pairs()
    for _ in range(N):
        p = rng.uniform(*p_range, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        s, t = od_pairs[rng.integers(len(od_pairs))]
        x_hat = solve_forward(net, theta_hat, s, t)
        demos.append(Demonstration(theta_hat=theta_hat, s=s, t=t, x_hat=x_hat))
    return demos


def classic_io(net: ShortestPathNetwork, demos: list[Demonstration],
                normalize_l2: bool = True) -> np.ndarray:
    """Point estimate theta_bar minimizing the average sub-optimality loss:
        (1/N) sum_k [theta^T xhat_k - min_{x in X(u_k)} theta^T x]
    Reformulated via strong LP duality (min_x theta^Tx = max_y{b_k^Ty : A^Ty<=theta})
    as a single joint convex program over (theta, {y_k}):
        min_{theta, y_k} (1/N) sum_k [theta^T xhat_k - b_k^T y_k]
        s.t. A^T y_k <= theta  for all k,   theta in Theta

    Theta domain: the simplex {theta >= 0, sum(theta) = 1}, NOT the L2 ball
    {||theta||_2 <= 1}. The inequality-ball relaxation is fine for the conformal
    calibration step (Theorem 1's per-point maximization, where shrinking theta can
    only hurt the objective), but is unsound here: since the objective is linear and
    homogeneous in theta and theta=0 is always feasible in {||theta||_2 <= 1}, the
    sub-optimality loss collapses to the trivial zero-loss solution theta=0 (verified
    empirically before this fix — cio/tests/test_io_pipeline.py caught it). The
    simplex excludes the origin and matches the network's nonnegative-edge-cost
    semantics; the returned point estimate is L2-normalized afterward (by default) to
    feed into the L2-normalized machinery downstream (conformal_calibrate, RFO),
    which is valid because the forward LP's optimal solution is invariant to positive
    rescaling of theta.
    """
    N = len(demos)
    d = net.d
    theta = cp.Variable(d)
    ys = [cp.Variable(net.num_nodes) for _ in range(N)]

    obj_terms = []
    constraints = [theta >= 0, cp.sum(theta) == 1]
    for k, dem in enumerate(demos):
        b_k = net.rhs(dem.s, dem.t)
        obj_terms.append(theta @ dem.x_hat - b_k @ ys[k])
        constraints.append(net.A.T @ ys[k] <= theta)

    objective = cp.Minimize(cp.sum(obj_terms) / N)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if theta.value is None:
        raise RuntimeError(f"classic_io failed to solve: status={prob.status}")
    theta_hat = np.maximum(theta.value, 0.0)  # clip small negative solver noise
    if normalize_l2:
        theta_hat = theta_hat / np.linalg.norm(theta_hat)
    return theta_hat


def _c_k(net: ShortestPathNetwork, dem: Demonstration, theta_bar: np.ndarray) -> float:
    """c_k = max_{theta_k, y} theta_k^T theta_bar
             s.t. y[s]=0,
                  y[v]-y[u] = theta_k[e]   for e=(u,v) on the observed path,
                  y[v]-y[u] <= theta_k[e]  for e=(u,v) off the path,
                  ||theta_k||_2 <= 1
    This is the ΘOPT(x_hat_k, u_k) feasibility set (CIO Theorem 1), specialized to the
    shortest-path LP's optimality conditions (dual feasibility + complementary slackness).
    """
    d = net.d
    on_path = path_edges(dem.x_hat)

    theta_k = cp.Variable(d)
    y = cp.Variable(net.num_nodes)

    constraints = [cp.norm(theta_k, 2) <= 1, y[dem.s] == 0]
    for e, (u, v) in enumerate(net.edges):
        lhs = y[v] - y[u]
        if on_path[e]:
            constraints.append(lhs == theta_k[e])
        else:
            constraints.append(lhs <= theta_k[e])

    prob = cp.Problem(cp.Maximize(theta_k @ theta_bar), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.value is None:
        raise RuntimeError(f"c_k computation infeasible: status={prob.status}")
    return float(prob.value)


def conformal_calibrate(net: ShortestPathNetwork, demos_val: list[Demonstration],
                         theta_bar: np.ndarray, gamma: float,
                         n_jobs: int | None = None) -> tuple[float, np.ndarray]:
    """CIO Theorem 1: alpha_gamma = arccos(Gamma_tau({c_k})), tau = ceil(gamma*(N+1)).
    c_k computed per validation point (embarrassingly parallel — Theorem 1's point).
    Returns (alpha_gamma, array of c_k).
    """
    N = len(demos_val)
    tau = int(np.ceil(gamma * (N + 1)))
    tau = min(max(tau, 1), N)

    if n_jobs is None or n_jobs == 1:
        c_ks = np.array([_c_k(net, dem, theta_bar) for dem in demos_val])
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            c_ks = np.array(pool.starmap(_c_k, [(net, dem, theta_bar) for dem in demos_val]))

    # Gamma_tau: the tau-th LARGEST value.
    sorted_desc = np.sort(c_ks)[::-1]
    c_tau = sorted_desc[tau - 1]
    c_tau = np.clip(c_tau, -1.0, 1.0)
    alpha_gamma = np.arccos(c_tau)
    return float(alpha_gamma), c_ks

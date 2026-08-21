"""The MDP analogue of CIO's inverse-feasible set Theta^OPT(x, u), and the calibration
score c_k built from it (plan.md Sec 4 P2, Sec 5 T1). See notes/T7_sketch.md and
notes/phase2_resolvent.md for the derivation and the acceleration argument below.

Core fact (exact, not an approximation): for a FIXED deterministic policy pi in a
tabular MDP with known transitions P and reward r_theta(s,a) = phi(s,a)^T theta, the
policy's own induced value function V^pi_theta is a LINEAR function of theta:

    V^pi_theta = (I - gamma P_pi)^{-1} Phi_pi theta =: W_pi theta      (S x d matrix W_pi)

where Phi_pi(s, :) = phi(s, pi(s)). Substituting into the Bellman equation for the
Q-function gives Q^pi_theta(s, a) = M_pi(s, a) . theta, with

    M_pi(s, a) = phi(s, a) + gamma * P(s, a, :) @ W_pi          (S x A x d tensor)

By the policy improvement theorem, pi is optimal for theta IF AND ONLY IF it has no
profitable one-step deviation under its OWN value function:

    Q^pi_theta(s, a) <= Q^pi_theta(s, pi(s))   for all s, a != pi(s)

This is EXACT MDP theory (no relaxation): a policy that is greedy w.r.t. its own value
function is optimal. So "pi_k is optimal under theta" is exactly the set of LINEAR
constraints  M_pi(s,a) . theta <= M_pi(s,pi(s)) . theta, once M_pi is precomputed.

This collapses CIO's Theorem-1-style calibration score

    c_k = max_theta { theta . theta_bar : ||theta||_2 <= 1, pi_k optimal for theta }

from a joint SOCP over (theta, V) of size d + S (the naive formulation, kept below as
`c_k_reference` for correctness testing) to an SOCP over theta ALONE of size d, with
S*(A-1) precomputed linear constraints and one one-time resolvent solve per distinct
policy -- the acceleration plan.md Sec 2.5/notes/phase2_resolvent.md set out to build.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP


def resolvent(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    """(I - gamma * P_pi)^{-1}, shape (S, S). O(S^3) exact solve."""
    S = mdp.S
    P_pi = mdp.P[np.arange(S), policy, :]
    return np.linalg.inv(np.eye(S) - mdp.gamma * P_pi)


def q_linear_operator(mdp: TabularMDP, policy: np.ndarray,
                       W_pi: np.ndarray | None = None) -> np.ndarray:
    """M_pi such that Q^pi_theta(s, a) = M_pi[s, a] @ theta. Shape (S, A, d).

    If W_pi (the S x d matrix with V^pi_theta = W_pi @ theta) is not supplied, it is
    computed here via `resolvent`; pass it in to reuse a resolvent already computed
    for this policy (e.g. across multiple demonstrators sharing the same policy).
    """
    if W_pi is None:
        Phi_pi = mdp.phi[np.arange(mdp.S), policy, :]  # (S, d)
        W_pi = resolvent(mdp, policy) @ Phi_pi  # (S, d)
    # M(s, a) = phi(s, a) + gamma * P(s, a, :) @ W_pi
    M = mdp.phi + mdp.gamma * np.einsum("sat,td->sad", mdp.P, W_pi)  # (S, A, d)
    return M


def c_k(mdp: TabularMDP, policy: np.ndarray, theta_bar: np.ndarray,
        M_pi: np.ndarray | None = None, solver=cp.CLARABEL) -> float:
    """Fast path: SOCP over theta alone (size d), using the precomputed linear
    Q-operator M_pi. This is the accelerated computation -- see module docstring.
    """
    if M_pi is None:
        M_pi = q_linear_operator(mdp, policy)

    S, A, d = mdp.S, mdp.A, mdp.d
    theta = cp.Variable(d)
    constraints = [cp.norm(theta, 2) <= 1]
    for s in range(S):
        a_star = policy[s]
        q_star = M_pi[s, a_star] @ theta
        for a in range(A):
            if a == a_star:
                continue
            constraints.append(M_pi[s, a] @ theta <= q_star)

    prob = cp.Problem(cp.Maximize(theta @ theta_bar), constraints)
    prob.solve(solver=solver)
    if prob.value is None:
        raise RuntimeError(f"c_k (fast) infeasible: status={prob.status}")
    return float(prob.value)


def resolvent_update(mdp: TabularMDP, base_policy: np.ndarray, base_W: np.ndarray,
                      new_policy: np.ndarray) -> np.ndarray:
    """Compute the resolvent (I - gamma P_{new_policy})^{-1} from an already-computed
    base_W = (I - gamma P_{base_policy})^{-1}, via a Sherman-Morrison-Woodbury update,
    when `new_policy` differs from `base_policy` on only a few states.

    Motivation (plan.md Sec 2.5 / notes/phase2_resolvent.md): in the population-of-
    demonstrators setting, many demonstrators plausibly share the same or very similar
    policies (e.g. most rideshare drivers take a small number of common routes). Given
    an already-computed resolvent for one policy, computing another differing on k
    states in O(k^2 S + k^3) via Woodbury is far cheaper than a fresh O(S^3) inverse
    when k << S.

    Derivation: let A = I - gamma P_base (base_W = A^{-1}). P_new differs from P_base
    only on the rows in `diff_states = {s : new_policy[s] != base_policy[s]}`,
    so P_new = P_base + E @ D^T where E (S x k) has columns = standard basis vectors
    e_s for s in diff_states, and D (S x k) has column j = P[s_j, new_policy[s_j], :]
    - P[s_j, base_policy[s_j], :]. Then I - gamma P_new = A - gamma E D^T, and by the
    Woodbury identity:
        (A - gamma E D^T)^{-1}
            = A^{-1} + gamma A^{-1} E (I_k - gamma D^T A^{-1} E)^{-1} D^T A^{-1}
    requiring only a k x k inverse, not S x S.
    """
    diff_states = np.where(new_policy != base_policy)[0]
    k = len(diff_states)
    if k == 0:
        return base_W.copy()

    S = mdp.S
    E = np.zeros((S, k))
    E[diff_states, np.arange(k)] = 1.0
    D = (mdp.P[diff_states, new_policy[diff_states], :]
         - mdp.P[diff_states, base_policy[diff_states], :]).T  # (S, k)

    W = base_W
    WE = W @ E  # (S, k)
    inner = np.eye(k) - mdp.gamma * (D.T @ WE)  # (k, k)
    correction = mdp.gamma * WE @ np.linalg.solve(inner, D.T @ W)  # (S, S)
    return W + correction


def c_k_reference(mdp: TabularMDP, policy: np.ndarray, theta_bar: np.ndarray,
                   solver=cp.CLARABEL) -> float:
    """Reference (slow) path: joint SOCP over (theta, V), size d + S. Directly mirrors
    cio/io_pipeline.py's `_c_k` (dual potentials -> value function, DAG -> general MDP).
    Used only to verify the fast path is exactly correct, not for production use.
    """
    S, A, d = mdp.S, mdp.A, mdp.d
    theta = cp.Variable(d)
    V = cp.Variable(S)
    constraints = [cp.norm(theta, 2) <= 1]
    for s in range(S):
        a_star = policy[s]
        for a in range(A):
            lhs = mdp.phi[s, a] @ theta + mdp.gamma * (mdp.P[s, a, :] @ V)
            if a == a_star:
                constraints.append(lhs == V[s])
            else:
                constraints.append(lhs <= V[s])

    prob = cp.Problem(cp.Maximize(theta @ theta_bar), constraints)
    prob.solve(solver=solver)
    if prob.value is None:
        raise RuntimeError(f"c_k (reference) infeasible: status={prob.status}")
    return float(prob.value)


def c_k_partial(mdp: TabularMDP, visited_actions: dict[int, int], theta_bar: np.ndarray,
                 solver=cp.CLARABEL) -> float:
    """Calibration score under PARTIAL observation: only a subset of states (a
    finite-trajectory sample, not the full policy) were observed. This is the
    setting plan.md Sec 4 P2 and notes/T2_proof.md Sec 5 flag as needing care: the
    resolvent-based fast path (`c_k`) requires the FULL policy (the resolvent
    (I-gamma P_pi)^-1 is undefined if pi is unknown at some states), so partial
    observation falls back to the joint (theta, V) formulation (`c_k_reference`'s
    structure), but with Bellman-optimality constraints written ONLY for the
    VISITED (s, a_observed) pairs -- V is otherwise unconstrained at unvisited
    states (it does not appear in the objective, so this costs nothing but does
    relax the feasible set, exactly as T2_proof.md predicts: Theta-hat_empirical
    (visited-only) is a SUPERSET of the true Theta-hat (full-policy), so c_k here is
    an UPPER BOUND on what the full-policy c_k would have been. Because arccos is
    decreasing, this typically gives a NARROWER cap. Exchangeability still gives
    intersection coverage for the relaxed empirical feasible set, but that does
    not imply coverage for the unobserved full-policy feasible set.

    `visited_actions`: dict mapping visited state index -> the action observed
    there (majority-vote action if visited multiple times with different actions).
    """
    S, A, d = mdp.S, mdp.A, mdp.d
    theta = cp.Variable(d)
    V = cp.Variable(S)
    constraints = [cp.norm(theta, 2) <= 1]
    for s, a_star in visited_actions.items():
        for a in range(A):
            lhs = mdp.phi[s, a] @ theta + mdp.gamma * (mdp.P[s, a, :] @ V)
            if a == a_star:
                constraints.append(lhs == V[s])
            else:
                constraints.append(lhs <= V[s])

    prob = cp.Problem(cp.Maximize(theta @ theta_bar), constraints)
    prob.solve(solver=solver)
    if prob.value is None:
        raise RuntimeError(f"c_k (partial) infeasible: status={prob.status}")
    return float(prob.value)

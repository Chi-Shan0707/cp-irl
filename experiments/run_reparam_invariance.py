"""E1 (CPIRL_FIRST_PRINCIPLES.md Sec 1.3, Sec 9): does the angular cap score
depend on an arbitrary choice of feature basis, while the value-gap and
decision-aware scores do not?

Construction: reparametrize phi -> phi @ A^T, theta -> A^{-T} theta for a
symmetric positive-definite A = Q diag(s) Q^T with log-spaced singular values
1..kappa (a random rotated rescaling of feature "units"). This leaves the
reward function R(s,a) = phi(s,a)^T theta and hence the MDP's behavior EXACTLY
unchanged (checked below) -- so any score whose value changes under this
transform is picking up an artifact of the coordinate choice, not a property
of the underlying decision problem.

conformal/tests/test_decision_metric.py::test_feature_reparametrization_invariance
already proves invariance of ||.||_D to machine precision on a single random
GL_d matrix; this script instead sweeps the CONDITION NUMBER of the
transformation and shows the angular score's sensitivity grows with it while
value-gap and decision-aware stay flat -- the shape of evidence a reviewer
asking "how bad is this in practice" wants to see.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_reparam_invariance.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from envs.mdp import TabularMDP, random_mdp, value_iteration
from irl.feasible_set import c_k as angular_c_k
from conformal.calibrate import conformal_calibrate
from conformal.value_gap import calibrate_value_gap, value_gap_score
from conformal.decision_metric import calibrate_decision_aware, distance_to_cone

KAPPAS = [1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
SEEDS = list(range(10))
S, A, D, GAMMA = 6, 3, 4, 0.9
N_VAL = 16
GAMMA_TARGET = 0.8


def rotated_rescaling(d: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """A = Q diag(s) Q^T, s log-spaced in [1, kappa], Q random orthogonal.
    Symmetric positive-definite, so A = A^T and A^{-T} = A^{-1} -- a
    representative, easily-invertible member of the reparametrization group
    GL_d (full generality is already covered by the unit test's random GL_d
    matrix; this family varies the transform only through its conditioning. Note that
    run_one reseeds per (seed, kappa), so the MDP, center and demonstrators are also
    redrawn at each kappa; the before/after comparison within a run is what is exact).
    """
    if kappa == 1.0:
        return np.eye(d)
    Qraw, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.exp(np.linspace(0.0, np.log(kappa), d))
    return Qraw @ np.diag(s) @ Qraw.T


def run_one(seed: int, kappa: float) -> dict:
    rng = np.random.default_rng(seed * 1000 + int(kappa * 7))
    mdp = random_mdp(S, A, D, gamma=GAMMA, rng=rng)
    theta_bar = rng.normal(size=D)
    theta_bar /= np.linalg.norm(theta_bar)

    pols_val = []
    for _ in range(N_VAL):
        tk = theta_bar + 0.35 * rng.normal(size=D)
        _, _, pol = value_iteration(mdp, tk)
        pols_val.append(pol)
    probe_policy = pols_val[0]

    A_mat = rotated_rescaling(D, kappa, rng)
    A_invT = np.linalg.inv(A_mat).T
    phi_tilde = mdp.phi @ A_mat.T
    mdp_tilde = TabularMDP(mdp.P, phi_tilde, mdp.gamma, mdp.mu0)
    theta_bar_tilde = A_invT @ theta_bar

    # sanity: reward function itself must be pointwise unchanged
    reward_diff = float(np.max(np.abs(mdp.reward(theta_bar) - mdp_tilde.reward(theta_bar_tilde))))

    # --- angular score (needs unit-norm theta_bar in EACH coordinate system,
    # which is itself part of why this score is not coordinate-free) ---
    theta_bar_unit = theta_bar / np.linalg.norm(theta_bar)
    theta_bar_tilde_unit = theta_bar_tilde / np.linalg.norm(theta_bar_tilde)
    ang_before = angular_c_k(mdp, probe_policy, theta_bar_unit)
    ang_after = angular_c_k(mdp_tilde, probe_policy, theta_bar_tilde_unit)
    alpha_before, _ = conformal_calibrate(mdp, pols_val, theta_bar_unit, GAMMA_TARGET)
    alpha_after, _ = conformal_calibrate(mdp_tilde, pols_val, theta_bar_tilde_unit, GAMMA_TARGET)

    # --- value-gap score ---
    vg_before = value_gap_score(mdp, probe_policy, theta_bar)
    vg_after = value_gap_score(mdp_tilde, probe_policy, theta_bar_tilde)
    q_vg_before, _ = calibrate_value_gap(mdp, pols_val, theta_bar, GAMMA_TARGET)
    q_vg_after, _ = calibrate_value_gap(mdp_tilde, pols_val, theta_bar_tilde, GAMMA_TARGET)

    # --- decision-aware score ---
    da_before, _ = distance_to_cone(mdp, probe_policy, theta_bar)
    da_after, _ = distance_to_cone(mdp_tilde, probe_policy, theta_bar_tilde)
    q_da_before, _ = calibrate_decision_aware(mdp, pols_val, theta_bar, GAMMA_TARGET)
    q_da_after, _ = calibrate_decision_aware(mdp_tilde, pols_val, theta_bar_tilde, GAMMA_TARGET)

    return dict(
        seed=seed, kappa=kappa, reward_diff=reward_diff,
        angular_score_diff=abs(ang_after - ang_before),
        angular_alpha_before=alpha_before, angular_alpha_after=alpha_after,
        angular_alpha_diff=abs(alpha_after - alpha_before),
        value_gap_score_diff=abs(vg_after - vg_before),
        value_gap_q_diff=abs(q_vg_after - q_vg_before) if np.isfinite(q_vg_before + q_vg_after) else 0.0,
        decision_aware_score_diff=abs(da_after - da_before),
        decision_aware_q_diff=abs(q_da_after - q_da_before) if np.isfinite(q_da_before + q_da_after) else 0.0,
    )


def main():
    t0 = time.time()
    rows = []
    for kappa in KAPPAS:
        for seed in SEEDS:
            rows.append(run_one(seed, kappa))
    elapsed = time.time() - t0

    with open("experiments/reparam_invariance_results.json", "w") as f:
        json.dump(rows, f, indent=1)

    print(f"Total: {elapsed:.1f}s for {len(KAPPAS)}x{len(SEEDS)} runs\n")
    print("Sanity check: max |reward difference| across ALL runs "
          f"= {max(r['reward_diff'] for r in rows):.2e} (must be ~0)\n")

    print(f"{'kappa':>7} {'angular c_k diff':>17} {'angular alpha diff':>19} "
          f"{'value-gap diff':>15} {'decision-aware diff':>20}")
    for kappa in KAPPAS:
        subset = [r for r in rows if r["kappa"] == kappa]
        ang = np.mean([r["angular_score_diff"] for r in subset])
        alpha = np.mean([r["angular_alpha_diff"] for r in subset])
        vg = np.mean([r["value_gap_score_diff"] for r in subset])
        da = np.mean([r["decision_aware_score_diff"] for r in subset])
        print(f"{kappa:>7.0f} {ang:>17.5f} {alpha:>19.5f} {vg:>15.2e} {da:>20.2e}")

    print("\ncalibrated-set sensitivity (q_gamma / alpha_gamma difference before vs after):")
    print(f"{'kappa':>7} {'angular alpha diff':>19} {'value-gap q diff':>17} "
          f"{'decision-aware q diff':>22}")
    for kappa in KAPPAS:
        subset = [r for r in rows if r["kappa"] == kappa]
        alpha = np.mean([r["angular_alpha_diff"] for r in subset])
        vgq = np.mean([r["value_gap_q_diff"] for r in subset])
        daq = np.mean([r["decision_aware_q_diff"] for r in subset])
        print(f"{kappa:>7.0f} {alpha:>19.5f} {vgq:>17.2e} {daq:>22.2e}")


if __name__ == "__main__":
    main()

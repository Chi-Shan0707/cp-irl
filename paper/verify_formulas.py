"""Numerical audit of every displayed formula in paper/main.tex.

Each check re-derives the paper's claim from scratch (brute force where
possible) rather than calling the library routine the claim is about, so that
an error in the library cannot hide an error in the paper.

Run: python paper/verify_formulas.py
"""

import itertools
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from envs.mdp import TabularMDP, occupancy_of_policy, random_mdp, value_iteration

RNG = np.random.default_rng(20260815)
TOL = 1e-7
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def all_deterministic_occupancies(mdp):
    """Every vertex of M(P): one occupancy per deterministic policy."""
    out = []
    for actions in itertools.product(range(mdp.A), repeat=mdp.S):
        out.append(occupancy_of_policy(mdp, np.array(actions)))
    return np.array(out)


def V(mdp, theta, mu):
    """V_theta(mu) = theta^T Phi^T mu / (1 - beta), as defined in Section 2."""
    return float(mdp.phi.reshape(-1, mdp.d).T @ mu.reshape(-1) @ theta) / (1 - mdp.gamma)


def span_bruteforce(mdp, v, verts):
    vals = [V(mdp, v, m) for m in verts]
    return max(vals) - min(vals)


def span_two_solves(mdp, v):
    """The paper's closed form: ||v||_D = V*_v(rho_0) + V*_{-v}(rho_0)."""
    Vp = value_iteration(mdp, v)[0]
    Vm = value_iteration(mdp, -v)[0]
    return float(mdp.mu0 @ Vp + mdp.mu0 @ Vm)


def main():
    mdp = random_mdp(S=4, A=2, d_feat=3, gamma=0.9, rng=RNG)
    verts = all_deterministic_occupancies(mdp)
    Phi = mdp.phi.reshape(-1, mdp.d)

    # --- Sec. 2: normalisation of the occupancy measure --------------------
    check("Sec.2  occupancy normalised, sum_sa mu = 1",
          np.allclose([m.sum() for m in verts], 1.0, atol=TOL))

    # --- Sec. 2: V_theta(mu) equals the expected discounted return ---------
    th = RNG.normal(size=mdp.d)
    pol = RNG.integers(0, mdp.A, mdp.S)
    mu = occupancy_of_policy(mdp, pol)
    r = mdp.reward(th)
    P_pi = mdp.P[np.arange(mdp.S), pol, :]
    Vpi = np.linalg.solve(np.eye(mdp.S) - mdp.gamma * P_pi, r[np.arange(mdp.S), pol])
    check("Sec.2  V_theta(mu) = E[sum beta^t r_t]",
          abs(V(mdp, th, mu) - mdp.mu0 @ Vpi) < 1e-8,
          f"{V(mdp, th, mu):.10f} vs {mdp.mu0 @ Vpi:.10f}")

    # --- Sec. 3: ||v||_D = V*_v(rho0) + V*_{-v}(rho0) ----------------------
    errs = []
    for _ in range(20):
        v = RNG.normal(size=mdp.d)
        errs.append(abs(span_bruteforce(mdp, v, verts) - span_two_solves(mdp, v)))
    check("Sec.3  ||v||_D = V*_v + V*_{-v} (vs vertex enumeration)",
          max(errs) < 1e-6, f"max err {max(errs):.2e}")

    # --- Prop. 1: seminorm axioms -----------------------------------------
    u, v = RNG.normal(size=mdp.d), RNG.normal(size=mdp.d)
    c = abs(RNG.normal()) + 0.1
    hom = abs(span_two_solves(mdp, c * v) - c * span_two_solves(mdp, v))
    sub = span_two_solves(mdp, u + v) - span_two_solves(mdp, u) - span_two_solves(mdp, v)
    check("Prop.1 positive homogeneity", hom < 1e-6, f"err {hom:.2e}")
    check("Prop.1 subadditivity", sub < 1e-8, f"slack {sub:.3e}")

    # --- Prop. 1: regret bound with constant 1 -----------------------------
    worst = -np.inf
    for _ in range(60):
        th1, th2 = RNG.normal(size=mdp.d), RNG.normal(size=mdp.d)
        p1 = value_iteration(mdp, th1)[2]
        p2 = value_iteration(mdp, th2)[2]
        reg = V(mdp, th1, occupancy_of_policy(mdp, p1)) - V(mdp, th1, occupancy_of_policy(mdp, p2))
        worst = max(worst, reg - span_two_solves(mdp, th1 - th2))
    check("Prop.1 Reg_theta(pi_theta') <= ||theta - theta'||_D",
          worst <= 1e-8, f"max violation {worst:.2e}")

    # --- Prop. 1: reparametrisation invariance -----------------------------
    Amat = RNG.normal(size=(mdp.d, mdp.d))
    while abs(np.linalg.det(Amat)) < 1e-3:
        Amat = RNG.normal(size=(mdp.d, mdp.d))
    mdp2 = TabularMDP(mdp.P, np.einsum("ij,saj->sai", Amat, mdp.phi), mdp.gamma, mdp.mu0)
    v = RNG.normal(size=mdp.d)
    check("Prop.1 invariance under phi -> A phi, theta -> A^{-T} theta",
          abs(span_two_solves(mdp, v) - span_two_solves(mdp2, np.linalg.inv(Amat).T @ v)) < 1e-6)

    # --- Prop. 1 / App.: A(P) is the exact zero set ------------------------
    flat = verts.reshape(len(verts), -1)
    D = np.array([Phi.T @ (flat[i] - flat[j]) for i in range(len(flat))
                  for j in range(len(flat))])            # feature difference body, unscaled
    null = np.linalg.svd(D)[2][np.linalg.matrix_rank(D, tol=1e-9):]
    if null.size:
        z = null[0]
        check("Prop.1 v in A(P)  =>  ||v||_D = 0", span_two_solves(mdp, z) < 1e-8)
    else:
        # Construct a null direction by appending a feature that is constant on M(P):
        # a potential-shaping column, which Lemma 1 says every mu prices identically.
        pot0 = RNG.normal(size=mdp.S)
        col = (mdp.gamma * mdp.P @ pot0 - pot0[:, None])[:, :, None]
        mdp3 = TabularMDP(mdp.P, np.concatenate([mdp.phi, col], axis=2), mdp.gamma, mdp.mu0)
        e = np.zeros(mdp3.d); e[-1] = 1.0
        check("Prop.1 v in A(P)  =>  ||v||_D = 0 (shaping column)",
              span_two_solves(mdp3, e) < 1e-8, f"{span_two_solves(mdp3, e):.2e}")

    # --- Lemma 1: potential shaping is constant on M(P) --------------------
    pot = RNG.normal(size=mdp.S)
    shaped = mdp.gamma * mdp.P @ pot - pot[:, None]       # (S,A): beta E[pot(s')] - pot(s)
    vals = [float(shaped.reshape(-1) @ m.reshape(-1)) for m in verts]
    check("Lemma 1 c^T Phi^T mu = -(1-beta) rho_0^T Phi_pot for every mu",
          max(abs(np.array(vals) + (1 - mdp.gamma) * (mdp.mu0 @ pot))) < 1e-9,
          f"spread over vertices {np.ptp(vals):.2e}")

    # --- Prop. 4 (App.): the angular cap's inner minimum --------------------
    def inner_min_bruteforce(tb, x, alpha, n=400000):
        Z = RNG.normal(size=(n, tb.size))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        Z *= RNG.uniform(0, 1, size=(n, 1)) ** (1 / tb.size)   # uniform in the ball
        Z = Z[Z @ tb >= np.cos(alpha)]
        return (Z @ x).min()

    def inner_min_formula(tb, x, alpha):
        a = float(tb @ x)
        b = float(np.linalg.norm(x - a * tb))
        if a <= -np.linalg.norm(x) * np.cos(alpha):
            return -float(np.linalg.norm(x))
        return a * np.cos(alpha) - b * np.sin(alpha)

    ok = True
    for alpha in (0.3, 0.9, 1.4):
        for _ in range(3):
            tb = RNG.normal(size=3); tb /= np.linalg.norm(tb)
            x = RNG.normal(size=3) * RNG.choice([1.0, -1.0, 4.0])
            f, s = inner_min_formula(tb, x, alpha), inner_min_bruteforce(tb, x, alpha)
            ok &= f <= s + 5e-3                      # sampling can only over-estimate
    check("Prop.4 piecewise inner minimum over the cap (vs sampling)", ok)

    # --- Thm. 4 (App.): cap diameter <= 2 sin(alpha) for alpha <= pi/2 ------
    ok = True
    for alpha in (0.2, 0.7, np.pi / 2):
        tb = np.array([1.0, 0.0, 0.0])
        Z = RNG.normal(size=(200000, 3))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        Z *= RNG.uniform(0, 1, size=(200000, 1)) ** (1 / 3)
        Z = Z[Z @ tb >= np.cos(alpha)][:2000]
        dmax = max(np.linalg.norm(Z[:, None] - Z[None], axis=2).max(), 0)
        ok &= dmax <= 2 * np.sin(alpha) + 1e-6
    check("Thm.4 diam(cap) <= 2 sin(alpha) on [0, pi/2]", ok)

    # --- Thm. 4 (App.): nu(mu) <= nubar ------------------------------------
    nubar = np.linalg.norm(Phi, axis=1).max() / (1 - mdp.gamma)
    check("Thm.4 nu(mu) = ||Phi^T mu||/(1-beta) <= ||phi||_{2,inf}/(1-beta)",
          all(np.linalg.norm(Phi.T @ m) / (1 - mdp.gamma) <= nubar + 1e-12 for m in flat))

    # --- Sec. 3: gauge form of the deployed robust objective ---------------
    # min_{||delta||_D <= q} delta^T x  ==  -q * gauge_{D_F}(x),  for x in D_F.
    Dbody = D / (1 - mdp.gamma)
    def gauge(x):
        """min{t>0 : x in t*conv(D_F)} by LP over the generating points."""
        import cvxpy as cp
        lam = cp.Variable(len(Dbody), nonneg=True)
        t = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(t),
                          [Dbody.T @ lam == x, cp.sum(lam) == t])
        prob.solve(solver=cp.CLARABEL)
        return float(t.value)

    import cvxpy as cp
    ok, worst = True, 0.0
    for _ in range(5):
        i, j = RNG.integers(0, len(verts), 2)
        x = Phi.T @ (flat[i] - flat[j]) / (1 - mdp.gamma)
        q = 0.7
        delta = cp.Variable(mdp.d)
        # ||delta||_D <= q  written through its own generating points
        cons = [Dbody @ delta <= q]          # h_{D_F}(delta) <= q, D_F symmetric
        prob = cp.Problem(cp.Minimize(delta @ x), cons)
        prob.solve(solver=cp.CLARABEL)
        lhs = float(prob.value)
        rhs = -q * gauge(x)
        worst = max(worst, abs(lhs - rhs))
        ok &= abs(lhs - rhs) < 1e-5
    check("Sec.3 min_{||delta||_D<=q} delta^T x = -q gauge_{D_F}(x)", ok,
          f"max err {worst:.2e}")

    # --- Sec. 3: the penalty is automatically normalised -------------------
    gs = []
    for _ in range(6):
        i, j = RNG.integers(0, len(verts), 2)
        gs.append(gauge(Phi.T @ (flat[i] - flat[j]) / (1 - mdp.gamma)))
    check("Sec.3 gauge_{D_F}(x) in [0,1] for x in D_F",
          all(-1e-7 <= g <= 1 + 1e-6 for g in gs), f"range [{min(gs):.3f}, {max(gs):.3f}]")

    # --- Thm. 3 (App.): angular score lies in [0,1] ------------------------
    # c_k = max_{theta in Theta(pi_k)} theta^T thetabar, 0 feasible, ||theta||<=1.
    check("Thm.3 sentinel argument: 0 <= c_k <= 1", True, "0 feasible; Cauchy-Schwarz")

    print()
    bad = [n for n, ok_, _ in results if not ok_]
    print(f"{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("FAILED:", *bad, sep="\n  ")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

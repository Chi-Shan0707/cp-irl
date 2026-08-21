"""Figure 2: the identification budget required for a nontrivial certificate.

The tractable Stage-I score yields the containment radius R=2q+eta whenever
q<1.  Proposition 5 says that safe robust prescription can differ from the
reference only if R<1.  Hence each fitted/calibrated run has an observable
identification budget

    eta_max = max(1 - 2*q, 0),

and an externally certified fiber diameter must be strictly below this budget.
This figure does not estimate eta from test rewards.  It reports how strong an
independent certificate would have to be, which is the auditable diagnostic
implied by the theory.

Data: experiments/containment_audit_results_v2.json
Run:  python paper/make_fig2_identification_frontier.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "containment_audit_results_v2.json"
OUT = Path(__file__).resolve().parent / "figures"

# Okabe--Ito: colorblind safe, with line style redundancy for grayscale.
COLORS = {
    "random_mdp": "#0072B2",
    "gridworld": "#D55E00",
    "objectworld": "#009E73",
    "pooled": "#111111",
}
LABELS = {
    "random_mdp": "random MDP",
    "gridworld": "gridworld",
    "objectworld": "Objectworld",
    "pooled": "pooled",
}
STYLES = {"random_mdp": "-", "gridworld": "--", "objectworld": "-.", "pooled": ":"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def _load_rows():
    rows = json.loads(DATA.read_text())
    if not rows or any(r.get("schema_version") != "projective-v2" for r in rows):
        raise RuntimeError("expected only projective-v2 audit rows")
    if any(not 0 <= r["q_decision_aware"] <= 1 + 1e-6 for r in rows):
        raise RuntimeError("normalized q must lie in [0,1]")
    return rows


def _block_values(rows, env, fn):
    """Average estimators before treating environment--seed as a replicate."""
    blocks = defaultdict(list)
    for r in rows:
        if env == "pooled" or r["env"] == env:
            blocks[(r["env"], r["seed"])].append(fn(r))
    return np.array([st.mean(v) for v in blocks.values()], dtype=float)


def main():
    rows = _load_rows()
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

    # Panel (a): the observable calibration term, before any eta is supplied.
    ax = axes[0]
    envs = ["random_mdp", "gridworld", "objectworld"]
    positions = np.arange(1, len(envs) + 1)
    q_groups = [np.array([r["q_decision_aware"] for r in rows if r["env"] == e])
                for e in envs]
    bp = ax.boxplot(q_groups, positions=positions, widths=0.52, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": "0.25"}, capprops={"color": "0.25"})
    for box, env in zip(bp["boxes"], envs):
        box.set(facecolor=COLORS[env], alpha=0.35, edgecolor=COLORS[env])
    for i, (env, vals) in enumerate(zip(envs, q_groups), start=1):
        # Deterministic jitter: visible individual runs without stochastic plotting.
        jitter = np.linspace(-0.16, 0.16, len(vals))
        ax.scatter(i + jitter, np.sort(vals), s=8, color=COLORS[env], alpha=0.7,
                   edgecolors="none", zorder=3)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.text(3.38, 0.505, r"$q=0.5$", va="bottom", ha="right", fontsize=6.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[e] for e in envs])
    ax.set_ylim(-0.02, 1.03)
    ax.set_ylabel(r"calibrated intersection radius $q_\gamma$")
    ax.set_title("(a) Calibration spends the identification budget", loc="left")

    # Panel (b): exact certificate-strength frontier implied by R=2q+eta<1.
    ax = axes[1]
    eta_grid = np.linspace(0.0, 1.0, 101)
    for env in [*envs, "pooled"]:
        means, lows, highs = [], [], []
        for eta in eta_grid:
            vals = _block_values(
                rows, env,
                lambda r, e=eta: float(2.0 * r["q_decision_aware"] + e < 1.0),
            )
            mean = float(vals.mean())
            # 95% t interval over independent environment--seed blocks.  Clamp
            # only for display; the plotted mean remains the observed fraction.
            tcrit = 2.262 if len(vals) == 10 else 2.045
            half = 0.0 if len(vals) < 2 else tcrit * float(vals.std(ddof=1)) / np.sqrt(len(vals))
            means.append(mean)
            lows.append(max(0.0, mean - half))
            highs.append(min(1.0, mean + half))
        ax.plot(eta_grid, means, color=COLORS[env], linestyle=STYLES[env],
                linewidth=1.6 if env == "pooled" else 1.25, label=LABELS[env])
        if env == "pooled":
            ax.fill_between(eta_grid, lows, highs, color="0.5", alpha=0.16, linewidth=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"externally certified fiber width $\eta$")
    ax.set_ylabel(r"fraction with $R=2q_\gamma+\eta<1$")
    ax.set_title("(b) Required strength of extra identification", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.text(0.98, 0.25, r"universal $\eta=2$: 0%", ha="right", va="bottom",
            fontsize=6.5, color="0.3")

    fig.tight_layout(pad=0.6, w_pad=1.7)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_identification_frontier.{ext}", bbox_inches="tight")
    print("wrote", OUT / "fig2_identification_frontier.pdf")


if __name__ == "__main__":
    main()

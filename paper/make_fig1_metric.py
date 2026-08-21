"""Figure 1: the metric choice, measured.

(a) A reward-preserving feature reparametrization moves the angular cap's
    calibrated radius but leaves the span radius at solver tolerance.
(b) Intersection coverage is at target for every construction while containment
    of the fresh demonstrator's latent reward ray is not. The event is
    feasible-set intersection for the angular and span scores, but only score
    coverage for the value-gap baseline.

Data: experiments/reparam_invariance_results.json,
      experiments/containment_audit_results_v2.json
Run:  python paper/make_fig1_metric.py
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = Path(__file__).resolve().parent / "figures"

# Okabe-Ito, colorblind safe.
ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
GREY = "#999999"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def panel_a(ax):
    rows = json.load(open(EXP / "reparam_invariance_results.json"))
    by_kappa = defaultdict(list)
    for r in rows:
        by_kappa[r["kappa"]].append(r)
    kappas = sorted(by_kappa)

    def series(key):
        return [st.mean(abs(r[key]) for r in by_kappa[k]) for k in kappas]

    ang = series("angular_alpha_diff")
    ang_hi = [max(abs(r["angular_alpha_diff"]) for r in by_kappa[k]) for k in kappas]
    dec = series("decision_aware_q_diff")
    vg = series("value_gap_q_diff")

    floor = 1e-17  # so that the exact zeros at kappa=1 remain plottable on a log axis
    clip = lambda xs: [max(x, floor) for x in xs]

    ax.fill_between(kappas, clip(ang), clip(ang_hi), color=ORANGE, alpha=0.18, lw=0)
    ax.plot(kappas, clip(ang), "o-", color=ORANGE, lw=1.4, ms=3.5,
            label=r"angular cap $\alpha_\gamma$ (rad)")
    ax.plot(kappas, clip(dec), "s-", color=BLUE, lw=1.4, ms=3.5,
            label=r"span score $q_\gamma$")
    ax.plot(kappas, clip(vg), "^--", color=GREEN, lw=1.2, ms=3.5,
            label=r"value-gap ball $q_\gamma$")
    ax.axhspan(floor, 1e-9, color=GREY, alpha=0.15, lw=0)
    ax.text(180, 2e-17, "solver tolerance", fontsize=6.5, color="0.35", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor, 30.0)
    ax.set_xlabel(r"reparametrization condition number $\kappa$")
    ax.set_ylabel("change in calibrated radius")
    ax.set_title("(a) An intervention that changes nothing", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.6)


def panel_b(ax):
    aud = json.load(open(EXP / "containment_audit_results_v2.json"))
    if any(r.get("schema_version") != "projective-v2" for r in aud):
        raise RuntimeError("panel (b) requires projective-v2 audit rows")
    mean = lambda rows, k: st.mean(r[k] for r in rows)
    # 95% t interval over 30 environment-seed blocks. Estimators within a block
    # reuse the same generated population, so average them before computing the
    # interval; test draws sharing one fit are likewise not independent replicates.
    def ci95(rows, key):
        blocks = defaultdict(list)
        for r in rows:
            blocks[(r["env"], r["seed"])].append(r[key])
        vals = [st.mean(v) for v in blocks.values()]
        return 2.045 * st.stdev(vals) / len(vals) ** 0.5

    labels = ["angular\ncap", "value-gap\nball", "span\nball"]
    event_keys = ["angular_intersect_cov", "value_gap_score_event_cov",
                  "decision_aware_intersect_cov"]
    contain_keys = ["angular_containment", "value_gap_containment",
                    "decision_aware_containment"]
    events = [mean(aud, k) for k in event_keys]
    contain = [mean(aud, k) for k in contain_keys]

    x = range(len(labels))
    w = 0.38
    dx = 0.12  # uniform rightward offset from each bar's centerline
    for i, (v, k) in enumerate(zip(events, event_keys)):
        ax.bar(i - w / 2, v, w, yerr=ci95(aud, k), capsize=2,
               color=GREY, edgecolor="black", lw=0.5,
               label="intersection" if i == 0 else None)
        ax.text(i - w / 2 + dx, v + 0.04, f"{v:.2f}", ha="center", fontsize=6.5)
    for i, (v, k) in enumerate(zip(contain, contain_keys)):
        ax.bar(i + w / 2, v, w, yerr=ci95(aud, k), capsize=2,
               color=ORANGE, edgecolor="black", lw=0.5,
               label="containment" if i == 0 else None)
        ax.text(i + w / 2 + dx, v + 0.04, f"{v:.2f}", ha="center", fontsize=6.5)

    ax.axhline(0.8, color="black", ls=":", lw=1)
    ax.text(1.0, 0.68, r"target $\gamma=0.8$", fontsize=6.5, ha="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("realized rate")
    ax.set_title("(b) Intersection is not containment", loc="left")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(-0.02, 1.03))


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout(pad=0.6, w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_metric.{ext}", bbox_inches="tight")
    print("wrote", OUT / "fig1_metric.pdf")


if __name__ == "__main__":
    main()

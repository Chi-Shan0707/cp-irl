"""Web overview figures; exact geometry, no experiment rerun. Run `make overview`.

Use the Appendix D 16-action regular polygon. In coordinates w=2*theta/(1-beta),
the occupancy-span norm is N(w)=max_j p_j^T w. Thus balls are polygons, not
Euclidean circles. Figure 1 displays action 1's full cone and normalized fiber.
Figure 2's curve is specific to this MDP and the fixed uniform reference:
G(R)=max(0,(1-R)/2). Since conv(p_j) is centrally symmetric, its gauge g gives
worst-case advantage (z_1-R*g(z))/2. z_1<=g(z)<=1; z=p_0 attains the positive
branch, and z=0 attains zero. This is not a universal performance curve.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from scipy.optimize import linprog

OUT = Path(__file__).resolve().parent / "figures"
BLUE, GREEN, ORANGE, INK = "#0072B2", "#009E73", "#D55E00", "#253442"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                    "mathtext.fontset": "dejavusans", "svg.fonttype": "path",
                    "pdf.fonttype": 42, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.labelcolor": INK,
                    "text.color": INK, "axes.edgecolor": "#AAB3BC"})
m = 16
angles = np.arange(m) * 2 * np.pi / m
p = np.column_stack((np.cos(angles), np.sin(angles)))
h = np.pi / m
vertices = np.column_stack((np.cos(angles+h), np.sin(angles+h))) / np.cos(h)
center, latent = np.array([1., 0.]), p[1]
fiber = np.array([[np.cos(h), np.sin(h)], [np.cos(3*h), np.sin(3*h)]]) / np.cos(h)
eta = 2 * np.tan(h)
cone_constraints = p - p[1]
res = linprog([0, 0, 1], A_ub=np.vstack((np.column_stack((p, -np.ones(m))),
                  np.column_stack((cone_constraints, np.zeros(m))))),
              b_ub=np.r_[p @ center, np.zeros(m)],
              bounds=[(None, None), (None, None), (0, None)], method="highs")
assert res.success
q = res.fun
R = min(2, 2*q+eta)
norm = lambda w: np.max(p @ w)
assert np.isclose(norm(latent), 1) and norm(latent-center) > q
assert np.isclose(norm(fiber[1]-fiber[0]), eta)
assert all(norm(u-center) <= R for u in fiber)


def save(fig, name, labels):
    """Check text bounds, including axes, legends and annotation text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(labels)
    for ax in fig.axes:
        labels.extend(ax.texts)
        labels.extend([ax.xaxis.label, ax.yaxis.label])
        labels.extend(ax.get_xticklabels() + ax.get_yticklabels())
    for legend in fig.legends:
        labels.extend(legend.get_texts())
    labels = [t for t in labels if t.get_visible() and t.get_text()]
    # Text's method excludes the arrow extent of Annotation objects.
    boxes = [Text.get_window_extent(t, renderer) for t in labels]
    for i, box in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        for j in range(i):
            assert not box.overlaps(boxes[j]), (labels[i].get_text(), labels[j].get_text())
    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        temporary = OUT / f".{name}.tmp.{ext}"
        fig.savefig(temporary, dpi=220, facecolor="white",
                    metadata={"Creator": "CP-IRL: paper/make_core_figures.py"})
        temporary.replace(OUT / f"{name}.{ext}")
    plt.close(fig)


fig = plt.figure(figsize=(12, 6.5))
labels = []
for left, title, enlarged in [(0.07, "1  Calibrate intersection", False),
                              (0.55, "2  Certify reward containment", True)]:
    labels.append(fig.text(left, .945, title, fontsize=17, weight="bold"))
    labels.append(fig.text(left, .885, "Nearest-cone distance gives an LP score." if not enlarged
                           else "A certified fiber width supplies the bridge.", fontsize=12))
    ax = fig.add_axes([left+.035, .32, .355, .49])
    ax.add_patch(Polygon([[0, 0], 3*fiber[0], 3*fiber[1]], facecolor="#ECEFF2",
                         edgecolor="#9AA5AF", linewidth=1.3, zorder=0))
    if enlarged:
        ax.add_patch(Polygon(center+R*vertices, facecolor="#DDF3EA", edgecolor=GREEN,
                             linewidth=2, zorder=1))
    ax.add_patch(Polygon(center+q*vertices, facecolor="#DDECF7", edgecolor=BLUE,
                         linewidth=2, linestyle="--" if enlarged else "-", zorder=2))
    ax.plot(fiber[:, 0], fiber[:, 1], color=ORANGE, linewidth=4, solid_capstyle="round", zorder=3)
    ax.scatter(*center, color=INK, marker="+", s=130, linewidths=2, zorder=4)
    ax.scatter(*latent, color=ORANGE, marker="*", s=150, edgecolor="white", linewidth=.7, zorder=4)
    ax.set(xlim=(0, 1.9), ylim=(-.85, 1.0), xlabel=r"$w_1$", ylabel=r"$w_2$")
    ax.set_aspect("equal")
    ax.set_xticks([0, .5, 1, 1.5]); ax.set_yticks([-.5, 0, .5, 1])
    ax.tick_params(labelsize=10)
    ax.grid(alpha=.15, zorder=-1)
    labels.append(fig.text(left+.035, .20,
        r"$B_q\cap K(\pi)\ne\varnothing,\qquad\theta^\circ\notin B_q$" if not enlarged
        else r"$R=\min\{2,\,2q+\eta\},\qquad\theta^\circ\in B_R$", fontsize=16))
handles = [Patch(facecolor="#ECEFF2", edgecolor="#9AA5AF", label=r"Feasible cone $K(\pi)$"),
           Line2D([], [], color=ORANGE, lw=4, label=r"Unit-span fiber $F^\circ(\pi)$"),
           Line2D([], [], color=INK, marker="+", ls="", markersize=9, label="Fitted center"),
           Line2D([], [], color=ORANGE, marker="*", ls="", markersize=11, label="Latent reward")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.095), ncol=4,
           frameon=False, fontsize=11, columnspacing=1.5)
labels.append(fig.text(.5, .045,
    "Exact 16-action geometry  ·  span balls, not Euclidean circles  ·  q = 0.191, η = 0.398, R = 0.781",
    ha="center", fontsize=11))
save(fig, "core_geometry", labels)

fig = plt.figure(figsize=(12, 5.0))
labels = [fig.text(.065,.93,"3  Prescribe relative to a reference",fontsize=18,weight="bold")]
ax = fig.add_axes([.085,.20,.40,.62])
radii = np.linspace(0,2,401)
ax.axvspan(1,2,color="#ECEFF2",zorder=0)
ax.plot(radii, np.maximum(0,(1-radii)/2), color=GREEN, lw=3)
ax.axvline(1, color="#87929C", ls="--", lw=1.3)
ax.scatter([R],[(1-R)/2],color=ORANGE,s=45,zorder=3)
ax.annotate("Certified toy example\nR = 0.781, G = 0.110", xy=(R,(1-R)/2),
            xytext=(1.10,.46),fontsize=10,arrowprops={"arrowstyle":"-","color":ORANGE},
            ha="left",va="center")
ax.text(1.5,.28,"Zero optimal\nworst-case advantage",ha="center",fontsize=11,color=INK)
ax.set(xlim=(0,2),ylim=(-.025,.53),xlabel="Containment radius R",ylabel="Optimal worst-case advantage G(R)")
ax.set_xticks([0,.5,1,1.5,2]); ax.set_yticks([0,.1,.2,.3,.4,.5]); ax.grid(alpha=.15)
for y, text, size, weight in [
    (.77, "Containment → reference-relative safety", 13, "bold"),
    (.665, r"$\Pr[V_{\theta^\circ}(\mu_R)\geq V_{\theta^\circ}(\mu_{\rm ref})]\geq\gamma$", 17, "normal"),
    (.545, "With exchangeability and a valid width certificate.", 11, "normal"),
    (.415, "For this translated span-ball prescription:", 12, "bold"),
    (.315, "R < 1: necessary, not sufficient, for positive advantage.", 10.5, "normal"),
    (.235, "R ≥ 1 gives G(R) = 0; the reference is optimal.", 11, "normal"),
    (.155, "This is not an impossibility for other reward sets.", 11, "normal"),
]:
    labels.append(fig.text(.555,y,text,fontsize=size,weight=weight))
labels.append(fig.text(.5,.045,"Curve: exact 16-action toy with a uniform reference; G(R) = max{0, (1 − R)/2}. Not a general performance curve.",
                       ha="center",fontsize=10.5))
save(fig, "core_prescription", labels)
print(f"Saved two verified geometric figures (SVG, PDF, PNG) in {OUT}")

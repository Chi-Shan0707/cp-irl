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
        if ext == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        temporary.replace(OUT / f"{name}.{ext}")
    plt.close(fig)


fig = plt.figure(figsize=(12, 6.5))
labels = []
for left, title, enlarged in [(0.07, "1  Many rewards fit the same behavior", False),
                              (0.55, "2  Bound the remaining ambiguity", True)]:
    labels.append(fig.text(left, .945, title, fontsize=17, weight="bold"))
    labels.append(fig.text(left, .885, "The blue set can still miss the actual reward." if not enlarged
                           else "Use that bound to enlarge the reward set.", fontsize=12))
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
    ax.set(xlim=(0, 1.9), ylim=(-.85, 1.0), xlabel="Reward weight 1", ylabel="Reward weight 2")
    ax.set_aspect("equal")
    ax.set_xticks([0, .5, 1, 1.5]); ax.set_yticks([-.5, 0, .5, 1])
    ax.tick_params(labelsize=10)
    ax.grid(alpha=.15, zorder=-1)
    labels.append(fig.text(left+.035, .20,
        "Touches a possible explanation; misses the star." if not enlarged
        else "The larger set covers the actual reward.", fontsize=12))
handles = [Patch(facecolor="#ECEFF2", edgecolor="#9AA5AF", label="Compatible rewards"),
           Line2D([], [], color=ORANGE, lw=4, label="At a common scale"),
           Line2D([], [], color=INK, marker="+", ls="", markersize=9, label="Reward estimate"),
           Line2D([], [], color=ORANGE, marker="*", ls="", markersize=11, label="Actual reward")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.095), ncol=4,
           frameon=False, fontsize=11, columnspacing=1.5)
labels.append(fig.text(.5, .045,
    "Exact example  ·  Blue: calibrated set  ·  Green: expanded set  ·  Orange: rewards at a common scale",
    ha="center", fontsize=11))
save(fig, "core_geometry", labels)

fig = plt.figure(figsize=(12, 5.0))
labels = [fig.text(.065,.93,"3  Choose a policy that protects a baseline",fontsize=18,weight="bold")]
ax = fig.add_axes([.085,.20,.40,.62])
radii = np.linspace(0,2,401)
ax.axvspan(1,2,color="#ECEFF2",zorder=0)
ax.plot(radii, np.maximum(0,(1-radii)/2), color=GREEN, lw=3)
ax.axvline(1, color="#87929C", ls="--", lw=1.3)
ax.scatter([R],[(1-R)/2],color=ORANGE,s=45,zorder=3)
ax.annotate("Small-example result\nR = 0.781; gain = 0.110", xy=(R,(1-R)/2),
            xytext=(1.10,.46),fontsize=10,arrowprops={"arrowstyle":"-","color":ORANGE},
            ha="left",va="center")
ax.text(1.5,.28,"No improvement\ncan be certified",ha="center",fontsize=11,color=INK)
ax.set(xlim=(0,2),ylim=(-.025,.53),xlabel="Size of the reward set (R)",ylabel="Best worst-case gain over baseline")
ax.set_xticks([0,.5,1,1.5,2]); ax.set_yticks([0,.1,.2,.3,.4,.5]); ax.grid(alpha=.15)
for y, text, size, weight in [
    (.77, "What does the guarantee mean?", 13, "bold"),
    (.66, "If the set covers the actual reward,", 12, "normal"),
    (.58, "the policy is no worse than a chosen baseline.", 12, "normal"),
    (.43, "Why not make the set very large?", 13, "bold"),
    (.33, "More possible rewards make improvement harder.", 11, "normal"),
    (.24, "At R ≥ 1, this ball can certify no gain.", 11, "normal"),
    (.15, "R < 1 allows a gain, but does not guarantee one.", 11, "normal"),
]:
    labels.append(fig.text(.555,y,text,fontsize=size,weight=weight))
labels.append(fig.text(.5,.045,"Safety has the same probability guarantee as reward coverage. Curve: one exactly solved example.",
                       ha="center",fontsize=10.5))
save(fig, "core_prescription", labels)
print(f"Saved two verified geometric figures (SVG, PDF, PNG) in {OUT}")

<div align="center">

# CP-IRL

**Calibrating What Behavior Can Identify**

*Conformal Reward Sets for Inverse Reinforcement Learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-8a3b2b.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#environment)
[![Paper](https://img.shields.io/badge/paper-PDF-8a3b2b.svg)](paper/main.pdf)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-555.svg)](https://chi-shan0707.github.io/cp-irl/)

[Paper (PDF)](paper/main.pdf) &nbsp;·&nbsp;
[Math walkthrough](math_summary.html) &nbsp;·&nbsp;
[Project page](https://chi-shan0707.github.io/cp-irl/) &nbsp;·&nbsp;
[Citation](#citation) &nbsp;·&nbsp;
[License](#license)

</div>

---

**Yuhan Chi** · School of Mathematical Sciences, Fudan University
<yhchi25@m.fudan.edu.cn>

Accepted as a **poster** at the NeurIPS 2026 workshop
[PUDM](https://sites.google.com/view/neurips-2026-workshop-pudm) —
*Physical Understanding for Decision-Making: Bridging Foundation Models and
Reliable Agents*, Sydney, 11–12 December 2026.

## The question

Conformal inverse optimization (CIO) handles one-shot inverse problems: fit a
point estimate, calibrate a radius on held-out decisions, optimize robustly
over the resulting set. The recipe looks portable to inverse reinforcement
learning. This paper reports what happens when it is carried over, component
by component — and the answer is that the conformal step is the one part that
transfers untouched.

**The score does not survive.** CIO measures a demonstration by an angle
between reward vectors. That angle depends on coordinates the problem never
fixes: a feature reparameterization that changes no reward and no demonstrator's
policy still moves the calibrated angular radius by 24° on average at condition
number κ = 200. Scoring instead by the spread a reward difference induces over
achievable policy values restores invariance, keeps the score a linear
program, and is the pointwise smallest seminorm that bounds worst-case regret.

**The event does not survive either.** Which event a quantile certifies is
decided by which distance to the cone of rewards explaining new behavior gets
scored. Scoring the *nearest* point certifies intersection with that cone;
scoring the *farthest* point of the normalized cone certifies containment of
the demonstrator's own latent reward. Only the first is a linear program. At a
0.80 target we measure the two events at **0.82** and **0.44** on the same
calibration run.

**So we price the relaxation.** Given a certified cone width η, the
calibrated radius q converts into a containment radius `R = min{2, 2q + η}`,
and `R ≥ 1` is provably degenerate: the robust policy collapses to the
reference. Without side information η = 2 and that is always the case. The
paper states this rather than hiding it — the certified policy is worse than
the point estimate in every cell of Table 2, and the constructive part of the
contribution is left for the reader to build.

## What's here

- tabular MDP, gridworld, and Objectworld environments
- classical, maximum-entropy, and Bayesian IRL estimators
- Bellman-resolvent feasible-reward sets and the span-seminorm conformal score
- intrinsic reward-ray normalization and the two-stage `q, η → R` bridge
- an occupancy-measure solver for robust MDPs with reward ambiguity
- the three experiment scripts behind the paper's numbers, with their recorded
  JSON outputs; superseded probes and ablations are in `experiments/legacy/`
- the camera-ready paper, its figure sources, and the submitted version

## Paper

| File | What it is |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | camera-ready, built by `make paper` |
| `paper/main.tex` | camera-ready source: `[sglblindworkshop, final]`, named author |
| `paper/main_full9_revised.tex` | the submitted double-blind version, unchanged; its PDF is byte-identical to the one on OpenReview |
| [`paper/ERRATA.md`](paper/ERRATA.md) | every change between the reviewed version and the camera-ready |
| [`paper/PROVENANCE.md`](paper/PROVENANCE.md) | which script and recorded file each number in the paper comes from |
| `paper/archive/` | the 19 August draft (source and PDF), plus the figures and figure scripts of an earlier draft, kept for provenance |
| [`math_summary.html`](math_summary.html) | a guided walk through the geometry, with proofs |

A math walkthrough of the core theory — the seminorm, the two events, the
`q → R` bridge, and the degeneracy threshold — is at
[`math_summary.html`](math_summary.html).

## Environment

The implementation uses Python 3.10 with NumPy, SciPy, and CVXPY. Install the
tested dependency ranges with:

```bash
python -m pip install -r requirements.txt
```

Common tasks, via the Makefile (`make help` lists them all):

```bash
make paper        # build paper/main.pdf (4-pass, runs bibtex)
make test         # 436 unit tests
make figures      # redraw both figures from the recorded JSONs
make verify       # re-check the identities listed in App. C numerically
make experiments  # rerun the three scripts behind the paper's numbers (hours)
```

## Repository layout

| Path            | Contents                                             |
|-----------------|-------------------------------------------------------|
| `envs/`         | tabular MDP, gridworld, Objectworld environments       |
| `cio/`          | conformal inverse optimization reference implementation |
| `irl/`          | inverse reinforcement learning estimators and feasible sets |
| `conformal/`    | calibration and concentration baselines                |
| `robust/`       | robust MDP optimization                                |
| `experiments/`  | experiment and ablation scripts                        |
| `tests/`        | environment tests                                       |
| `paper/`        | manuscript, bibliography, and figures                  |

## Scope

The implementation targets tabular MDPs with rewards linear in known features.
A useful safe prescription requires an application-specific certificate
satisfying `2q + η < 1`; the universal `η = 2` bound is valid but forces the
robust policy back to the reference. The framework is complete, while learning
or certifying a small cone width is left to the intended application's
additional information.

## Citation

```bibtex
@inproceedings{chi2026cpirl,
  title     = {Calibrating What Behavior Can Identify:
               Conformal Reward Sets for Inverse Reinforcement Learning},
  author    = {Chi, Yuhan},
  booktitle = {Physical Understanding for Decision-Making:
               Bridging Foundation Models and Reliable Agents},
  series    = {NeurIPS 2026 Workshop},
  year      = {2026},
  note      = {Poster}
}
```

A machine-readable version is in
[`CITATION.cff`](CITATION.cff), which GitHub renders as a "Cite this
repository" button. Add a DOI once the workshop proceedings are posted.

## License

Code is released under the [MIT License](LICENSE). The paper text and figures
in `paper/` are © Yuhan Chi and are made available for reading and citation;
if you'd like to reuse figures or text beyond fair use/citation, please reach
out (yhchi25@m.fudan.edu.cn).

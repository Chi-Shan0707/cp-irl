# Errata and revision history

The version reviewed on OpenReview is `paper/main_full9_revised.tex`. Its PDF,
`paper/main_full9_revised.pdf`, is byte-identical to the PDF on OpenReview
(SHA-256 `79fd2a882937df7396c370db13e06e1a58c7737903f09de4bd8fdae5e44f5e45`).

## 2026-10-06: post-acceptance author revision

`paper/main.tex` / `paper/main.pdf` now contain a visibly dated author revision.
It corrects attribution to CIO, limits the degeneracy interpretation, repairs
two proof details without changing the theorems, and adds a separately labelled
exact-cone toy experiment in Appendix D. The new appendix was **not part of the
reviewed submission**. See [POST_ACCEPTANCE_CHANGES.md](POST_ACCEPTANCE_CHANGES.md)
for the precise changes, version hashes, validation and reproduction commands.
The accepted submission above and original E1–E3 data remain unchanged.

## 2026-09-30: camera-ready corrections

The camera-ready at Git commit `4553b426876222a6b16cd0029bf954425a806bd0`,
before the October revision, differed from the reviewed version only by its
author/workshop notice and the corrections below. Those September corrections
did not change a theorem, a proof, or a reported experimental result.

## Mathematical correction

- **App. A, inherited angular baseline.** The reviewed version states the
  closed form of the cap's inner minimum, a(x)cos α − b(x)sin α, "under the sign
  condition a(x) ≤ −‖x‖₂cos α". The inequality was reversed: the closed form
  holds when a(x) ≥ −‖x‖₂cos α (otherwise the minimum is −‖x‖₂).
  `paper/verify_formulas.py` always used the correct branch.

## Statements corrected to match the code and recorded data

- **E1 design (§5).** The reviewed version says "we draw one random invertible
  A at each κ". In fact, for each κ and each of 10 seeds,
  `experiments/run_reparam_invariance.py` draws a random MDP (|S|=6, |A|=3, d=4),
  a center, 16 calibration demonstrators, and a symmetric positive-definite A of
  condition number κ. The span score in E1 is taken around θ̄ rather than its
  unit-span representative; both are invariant. The setup paragraph is now
  scoped to E2 and E3, which is where the 20/20/20 split applies.
- **E1 reward discrepancy.** 2.2×10⁻¹⁴ → 2.1×10⁻¹⁴ (recorded maximum
  2.13×10⁻¹⁴).
- **Feature dimension.** "d ≤ 5" → "d ≤ 4", the largest dimension actually used.
- **Concentration baseline (App. B).** The reviewed version attributes the
  4–460× regret inflation to DKW's additive slack moving a raw-value threshold.
  In the recorded runs the DKW-corrected level γ+ε exceeds 1 (γ=0.8, δ=0.05,
  N=30), so the DKW radius is infinite in all 270 runs. The script then
  substitutes a fixed cap of 10‖θ̄‖₂, and the 4–460× reflects that cap. The
  camera-ready says so.
- **Resolvent speed-up (App. A).** "2–5.5× the speed at a few hundred states" →
  "for |S| from 20 to 100", the range that was measured.
- **Seeding (App. B).** "All scripts set NumPy random generators from the
  recorded integer seed" → "are deterministic given the recorded integer seed".
  The Bayesian IRL sampler uses a fixed seed of 0 in every run.

## Wording and citations

- **§4.1.** The sentence "It is also the only such choice, which is what
  licenses calling any other geometry wrong rather than merely different" was
  removed. Proposition 2 proves minimality, not uniqueness: any c‖·‖_D with c ≥ 1
  also bounds regret. It now reads "No seminorm that bounds regret can be
  smaller."
- **§2 Related work.**
  - STARC is now described as generalizing the canonicalize-then-normalize
    recipe rather than "unifying" EPIC and DARD, which the STARC paper shows
    are not STARC metrics.
  - Gadot et al. is now described as handling ball-shaped non-rectangular
    reward ambiguity.
  - Kim et al. (2026) now sits with the work on suboptimal demonstrators.
  - "No finite-sample probability" now reads "no predictive coverage guarantee
    for a new demonstrator", because that line of work does give PAC guarantees.
  - Zhou and Zhu is now listed under "RL and decision making".
- **Missing citations added:** MaxEnt IRL (Ziebart et al., 2008), Bayesian IRL
  (Ramachandran and Amir, 2007), and Objectworld (Levine et al., 2011).
- **Bibliography metadata:** corrected author order for DARD (Wulfe et al.)
  and DPO, and brace-protected acronyms that the style would lowercase. See
  `paper/CITATION_AUDIT.md`.

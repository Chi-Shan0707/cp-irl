# Where each number in the paper comes from

Every value below was recomputed from the recorded output on 2026-09-30 and
matches the camera-ready. The recorded JSONs are committed, so `make figures`
and these checks work from a fresh clone. Rerunning the scripts (`make
experiments`) regenerates them.

| Paper location | Number(s) | Script | Recorded output |
|---|---|---|---|
| Abstract, §1, §5 E1, Fig. 1a | angular radius moves 0.424 rad (24.3°) on average and 1.071 worst at κ=200; span radius ≤ 2.0×10⁻⁸ at κ=200; reward discrepancy ≤ 2.1×10⁻¹⁴ | `experiments/run_reparam_invariance.py` | `experiments/reparam_invariance_results.json` |
| Abstract, §1, §5 E2, Fig. 1b | intersection 0.812 / 0.838 / 0.819; containment 0.368 / 0.004 / 0.438 (95% t over 30 environment–seed blocks) | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| §5 E3, Fig. 2 | q_γ mean 0.316, median 0.289, range [0.019, 0.871]; nontrivial 85.0 / 76.7 / 36.7 % at η = 0, 0.2, 0.5; 95 / 75 / 60 % by environment at η=0.2; η=2 gives R=2, containment 1.000, dominance 1.000, nontriviality 0 | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| App. B, Table 2 | all 30 regret cells (`aog_*` fields, mean over 10 seeds) | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| §5 setup | exact-fit LP baseline: span-zero center in 1 of 30 fits (gridworld, seed 6) | `experiments/run_center_validity.py` | `experiments/center_validity_results.json` |
| App. B, concentration baseline (value gap) | DKW radius infinite in 270/270 runs; 4–460× regret at the 10‖θ̄‖₂ cap | `experiments/legacy/run_value_gap_full_suite.py` | `experiments/value_gap_full_suite_results.json` (not committed; regenerate with the script) |
| App. B, concentration baseline (angular) | DKW cap never narrower than conformal | `experiments/legacy/run_phase5_gridworld.py` (prints to stdout); `conformal/tests/test_concentration_baseline.py` | — |
| App. A, after Prop. 5 | resolvent score 2–5.5× faster than the joint-(θ, V) reference for \|S\| = 20–100; agreement to solver precision | agreement: `irl/tests/test_feasible_set.py`; timing was measured once during development and no timing script was kept | — |
| App. C, all displayed identities | numerical checks | `paper/verify_formulas.py` | stdout |

The regenerated JSONs contain every run, not only the summaries reported in
the paper. Confidence intervals follow the figure scripts: 95% Student-t over
environment–seed blocks, after averaging the two estimators within each block.

# Legacy experiments

Scripts that back the author's working notes (`notes/`, kept locally and not
part of this repository) but are **not** cited by the camera-ready paper. They
are kept, not deleted, because several of them record
the negative results the paper's honesty argument depends on — in particular that
the value-gap ball and the uncertified policies are not dominated, and that the
Phase-1/2/5 lines of attack did not pan out.

They are excluded from the main experiment directory so that
`experiments/run_*.py` is exactly the set of scripts behind the paper's
numbers. Nothing here is imported by the live scripts. The paper cites only
`run_value_gap_full_suite.py` and `run_phase5_gridworld.py`, for the
concentration-baseline paragraph in App. B (see `paper/PROVENANCE.md`).

## Running one

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
PYTHONPATH=. python experiments/legacy/<script>.py
```

Four of them (`run_cone_width_probe.py`, `run_containment_score.py`,
`run_ipm_critic_probe.py`, `run_value_gap_full_suite.py`) write
`experiments/*.json` (gitignored); `run_cio_sweep.py` writes
`experiments/runs/cio_sweep/`; the rest print to stdout. Most docstrings still
give the pre-move command line `experiments/run_*.py`; run them from
`experiments/legacy/` as above.

## What is here, and why

| Group | Scripts | Why it is no longer in the paper |
|---|---|---|
| Phase-5 ablations | `run_phase5_gridworld.py`, `run_phase5_{gamma,noise,rationality,split,trajectory}_ablation.py` | The intersection/containment rework replaced the framing these ablations tested. See `notes/phase5_*.md`. |
| Value gap | `run_value_gap_comparison.py`, `run_value_gap_full_suite.py` | Superseded by the span-seminorm audit. The value gap survives in the paper only as a baseline; `notes/2026-08-09_value_gap_redesign.md` records why. |
| CIO reproduction | `run_cio_repro.py`, `run_cio_sweep.py` | The CIO reference implementation in `cio/` is still live and tested; only these driver scripts are stale. |
| Probes | `run_cone_width_probe.py`, `run_containment_score.py`, `run_ipm_critic_probe.py`, `run_separation_estimator_scope.py`, `run_reward_transfer_probe.py` | Exploratory. `farthest_in_cone` in `conformal/decision_metric.py` survives from `run_containment_score.py` as a superseded lower bound. |
| Separation (deleted result) | `run_t5_separation.py`, `run_irl_demo.py`, `run_scale_gridworld.py` | Back the unbounded-regret separation result that the rework removed from the paper. `notes/T5_proof.md`. |
| RLHF relabeling | `run_rlhf_relabeling_demo.py`, `run_rlhf_relabeling_robustness.py` | The paper's Discussion mentions preference-based reward modelling as untested future work; these probes predate that decision. `notes/rlhf_relabeling_result.md`. |

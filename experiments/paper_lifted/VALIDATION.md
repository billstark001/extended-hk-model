# Local validation and frozen plan — 2026-10-27

Historical initial snapshot. See [RECHECK.md](RECHECK.md) for the current source
version, recovery checks and refreshed frozen plan.

Validated on the local Apple Silicon host, Go 1.27.3, Accelerate, four workers.
Times below are elapsed wall time, including builds and offline output.
Formal execution has **not** started.

| Check | Result |
| --- | --- |
| Ruff lint | Passed across common, ehk, theory, lifted, both paper studies and tests |
| Ruff format | 29 changed Python files already formatted |
| Mypy | Passed, 56 source files |
| Python suite | 186 passed, 5 subtests passed; 113.95 seconds |
| Lifted Go continuation tests | Passed with the calibrated Accelerate module |
| Smoke | 14/14 tasks; 60.32 seconds |
| Timing pilot | 74/74 tasks; 204.86 seconds |
| Random naive/base pilot pairs | Three exact physical-path comparisons passed |
| Calibrated executable staging | Hashes, repeated staging and source bindings passed; no model executed |
| Frozen formal status | 0/35,870 tasks completed; no execution files |

Local artifacts live under `outputs/paper_lifted/`:

- `smoke-20261027-final/`: full smoke records, checkpoints and analysis.
- `pilot-20261027-final/`: timing records and analysis, including six fixed-window
  channel controls. Every control completed its planned tau=30 window.
- `formal-20261027/`: only `plan.json` and `budget.json`.
- `calibration-check-20261027/`: authenticated executable staging, without simulation.
- `verification-20261027.json`: machine-readable completion and source checks.

The pilot fingerprint is
`43004b6a11f7a53a987debbee128778ff3b1a69e414796c0441678c426f544e1`.
Only budget selection and executable staging changed after the final numerical
pilot. The freeze receipt records their before/after source hashes; every
generation, mathematical-analysis and scientific-design source still matches.

The frozen formal fingerprint is
`576ecb266aa064d47fd07a90b2eab687b4ad7ecd4b3d5e5d4f8ff64403e50a50`.
The selected configuration has 32 independent initial states per core ratio,
two mothers per initial state, 256 microscopic continuations per queried state,
and 256 coarse continuations per selected state on the main B=161 grid.
Side grids and mechanism cohorts have their own smaller allocations in the plan.
Core ratios are .25, 2 and 10. Physical settings remain HK + Random, N=500,
epsilon=.4, alpha=.02, dt=1, fixed out-degree 15, ten recommendations and Q=33.

The pilot-based estimate is **6.6507 hours**, including 15% scheduling overhead,
45 minutes for follow-up validation/tails and 40 minutes for offline analysis.
The persistent execution budget is eight hours. It stops admitting core work
before those reserves; unfinished tasks remain missing. The estimate extrapolates
path lengths and is not a guarantee of complete coverage. Sampling prioritizes
independent initial states and mother paths before inner replication.

The plan also retains B=81/241 resolution checks, the eleven-ratio mechanism
sweep, N=250/1000/2000 comparisons, bridge counts 1/4/16, joint moments, the
theory measure reference and fixed-window channel controls. Adaptive tail,
independent validation and local-window work is separately frozen after the
core and constrained by the remaining budget.

To start this frozen plan later, explicitly run from the repository:

```sh
.venv/bin/python -m experiments.paper_lifted run --work-dir outputs/paper_lifted/formal-20261027 --workers 4
```

This command was **not executed** during validation. Formal startup verifies the
frozen sources and stages the original measured executables from the pilot;
preserve the pilot directory alongside the formal plan.

# Recovery, partial exports and phase checks — 2026-10-27

This is the current validation snapshot. The earlier `VALIDATION.md` and its
frozen plan remain historical records. No formal experiment has been started.

The phases are **run / analyze / plot**, consistent with `paper_theory`.
`run` and `resume` default to all three phases; `--phase run` selects execution
alone. Standalone `analyze` and `plot` never start a model. Multiple `--phase`
options run in the fixed run/analyze/plot order.

| Check | Result |
| --- | --- |
| Ruff lint / format / mypy | Passed; 31 formatted files, 56 typed source files |
| Full Python suite | 194 passed, 5 subtests passed, 115.25 seconds |
| Native Go continuation tests | Passed with Accelerate |
| Standard smoke execution | 14/14, 51.06 seconds; analysis/plot independently rerun |
| Real in-flight SIGTERM | Native child exited; 1/16 committed before interruption |
| Resume after SIGTERM | 16/16; every numerical result exactly matches uninterrupted baseline |
| Second resume | All raw records unchanged; default three-phase workflow passed |
| Native tail-plan recovery | 2 core tasks + 2 tail tasks; first committed tail skipped after interruption |
| Tail sample count | Remains two original samples; extensions are never new samples |
| Expired deadline | No new tail task dispatched |
| Partial / empty exports | 1/16 and 0/16 both analyze and plot independently |
| Portable plotting | Four CSV files and optional manifest; no raw/runtime required |
| Latest timing pilot | 74/74; execution alone 187.41 seconds; independent analysis/plot passed |
| Calibrated executable staging | Updated binding and repeat staging passed; no solver executed |

Partial analysis retains all planned initial states and inner samples. Its report
separates pending work from physically unresolved paths. All tables use one
authenticated input snapshot even if execution commits additional records.
Follow-up plan creation during snapshot capture is detected. New raw records
invalidate `analysis_current` and `plots_current`; separate manifests prevent old
plots from entering the analysis identity. Completely pending unclassified
queries are not plotted as zero observed probabilities.

The simultaneous correction reserves 42 potential adaptive queries before
execution. Adding a follow-up plan cannot change the core's family size. Same-
stream tail results replace the original path's unknown label, without increasing
its sample count. Focused validation/window cohorts remain separate from population
timing, including the shared mother-zero initial-time estimate.

All local receipts, drivers, logs and partial exports are under
`outputs/paper_lifted/recheck-20261027/`. See `verification.json`,
`restart_report.json`, `followup_report.json`, `partial_report.json`,
`partial-review/analysis/`, `empty-review/analysis/` and `pytest.log`.
The native interruption and tail rehearsals use **smoke** profiles.

The current pilot is `outputs/paper_lifted/pilot-20261027-phases/`.
Only offline snapshot handling and calibration bookkeeping changed after its
numerical generation; the freeze receipt preserves those exact before/after
hashes. Simulator adapters, runtime sources, task design and physical settings
match the numerical calibration.

The current frozen plan is `outputs/paper_lifted/formal-20261027-phases/`, fingerprint
`e27e4d7c002b8e80c81b6128ce8f47cdf5a64a2dcbf3e041af59d92bc19b27d9`.
It selects 32 initial states per core ratio, two mothers per initial state, 256
microscopic continuations per queried state and **512** main-grid mesoscopic
continuations per selected state. Estimated total time is **6.8635 hours**,
including 45 minutes for follow-ups and 40 minutes for analysis; the cumulative
execution budget remains eight hours. The directory contains only the plan and
budget, with **0/37,186 tasks completed**. Preserve its pilot directory.

Example commands from the model repository for a later explicit start:

```sh
.venv/bin/python -m experiments.paper_lifted run --work-dir outputs/paper_lifted/formal-20261027-phases --workers 4 --phase run
.venv/bin/python -m experiments.paper_lifted analyze --work-dir outputs/paper_lifted/formal-20261027-phases
.venv/bin/python -m experiments.paper_lifted plot --work-dir outputs/paper_lifted/formal-20261027-phases
```

Use `resume` in place of `run` after an interruption. These formal commands have
**not** been executed during this check.

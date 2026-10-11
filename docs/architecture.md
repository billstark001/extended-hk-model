# Repository architecture

`experiments/` owns study design and orchestration: select cases, freeze plans, call libraries, assemble analyses/tables, and choose/layout plots. Algorithms, system operations, mathematical models and statistical estimators belong in `src/`. Analysis may select/group records and call estimators; an estimator or optimizer must not depend on experiment modules, output paths or historical artifacts.

`src/common` owns model-independent settings, I/O and hashes, numerical/statistical tools, plotting primitives and process/build/environment services. `src/ehk` owns microscopic observables, geometry and model-specific visualizations. The existing `experiments/paper` workflows are retained; any remaining embedded mathematical or system helpers there are recorded as legacy boundary debt, outside this theory-focused change.

`src/theory` owns parameters, energies, finite exposure probabilities, recommendation/rewiring operators, transport, distribution/trajectory comparisons and release observables. `release/` includes restricted means, T50/other first-crossing quantiles, plateau diagnostics, conditional statistics and empirical survival/mixture fits. T50 remains available as a sensitivity diagnostic. `runtime/` owns public request schemas, JSONL policy/transport, result identity and execution services. It does not implement a kinetic/PDE solver.

```text
src/common/runtime/          explicit resource environment, Go builds, subprocesses
src/theory/runtime/          request/result contracts and bounded execution
src/theory/release/          mathematical clocks, statistical estimators, empirical fits
experiments/paper_theory/
  __main__.py                one command interface
  designs.py, protocols.py   historical study choices and analysis protocols
  runtime/                   study build selection, runner, routes, sweep orchestration
    engine/                  Go adapter to the two public runtime libraries
  evidence/                  fresh revised-paper plan, analysis/report/figure assembly
  micro_validation/          matched micro/pair, conditional ensembles, finite population
  release_clocks/            release study selection and historical fit audits
  operator_validation/       measure/Fokker–Planck and numerical comparisons
  kernel_validation/         finite exposure and mixed-increment audits
  scaling/                   compatible small-update comparison assembly
  energy_scan/               baseline/expanded study orchestration
  manuscript/                historical evidence reports and fixed-data claim audits
```

The Go adapter can remain under `experiments`: it constructs explicit initial states, invokes public solvers and exports inspection/audit results. Model stepping and recommendation factories remain in the two Go libraries. Existing adapter projection and diagnostic instrumentation are retained; new mathematical/statistical estimators go into `src`. No solver is duplicated in the experiments.

`evidence` is the active fresh workflow for the proposed revision. One parameter-only plan generates measure and matched micro trajectories, then restricted means, conditional/hierarchical intervals, recommendation contrasts, fixed-r alpha collapse, state comparisons, numerical controls and finite-N robustness figures. Its paper profile includes both zeta=1/4 and a common tau=480 window with shorter prefixes. Smoke traverses the same paths with short horizons; it cannot establish a plateau or a paper result. Own endpoint and fixed pair endpoint normalization are explicit alternatives. Numerical diagnostics do not automatically justify extrapolating an infinite mean.

Old suites and T50 analyses stay available for historical reproducibility. The fixed historical manuscript figure assembler is not the fresh result entry point. Before removing unused experiments, classify their scientific role: main result, supplementary diagnostic, historical reproduction, or redundant workflow. Do not extend an unused historical scan to produce revised-paper evidence when the shared fresh plan covers it.

Dependency direction is `ehk → common`, `theory → common`, and `paper_theory → common/theory/public runtimes`. Neither source package imports experiments. Importing analysis creates no outputs and does not load a solver. CLI startup configures caches and subprocess limits explicitly. Builds bind actual source/toolchain/backend/binary hashes; plans and raw protocols reject incompatible resumes. Offline analysis needs frozen records, not a runtime executable.

Large inputs stay outside Git; generated builds, raw records, tables and figures stay ignored. Historical paper artifacts are optional provenance/data inputs, never code dependencies. `evidence` does not read them. CAS 2026 remains external empirical motivation; this repository does not recreate a CAS dataset. See the [study guide](../experiments/paper_theory/README.md) and [three-repository installation](runtime-dependencies.md).

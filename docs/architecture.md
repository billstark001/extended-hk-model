# Repository architecture

`experiments/` owns study design and orchestration: select cases, freeze plans, call libraries, assemble analyses/tables, and choose/layout plots. Algorithms, system operations, mathematical models and statistical estimators belong in `src/`. Analysis may select/group records and call estimators; an estimator or optimizer must not depend on experiment modules, output paths or historical artifacts.

`src/common` owns model-independent settings, I/O and hashes, numerical/statistical tools, plotting primitives and process/build/environment services. `src/ehk` owns microscopic observables, geometry and model-specific visualizations. The existing `experiments/paper` workflows are retained; any remaining embedded mathematical or system helpers there are recorded as legacy boundary debt, outside this theory-focused change.

`src/theory` owns parameters, energies, finite exposure probabilities, recommendation/rewiring operators, transport, distribution/trajectory comparisons and release observables. `release/` includes restricted means, T50/other first-crossing quantiles, plateau diagnostics, conditional statistics and empirical survival/mixture fits. T50 remains available as a sensitivity diagnostic. `runtime/` owns public request schemas, JSONL/binary execution, result identity/completion checks and verified micro/pair references. It does not implement a kinetic/PDE solver. Model-independent binary codecs, array conversion/comparison, subprocess cancellation, retry and local storage staging belong in `src/common`.

```text
src/common/types.py          generic record, transport, storage and provenance contracts
src/common/runtime/          resource environment, Go builds, subprocesses, storage staging
src/common/io/               generic codecs, typed arrays, streaming hashes and CSV I/O
src/theory/types.py          physical parameter, task, request and result contracts
src/theory/runtime/          request/result contracts, pair identity and completion validation
src/theory/release/          mathematical clocks, statistical estimators, empirical fits
experiments/paper_theory/
  __main__.py                one command interface
  types.py                   frozen plans, study protocols and run metadata
  designs.py, protocols.py   historical study choices and analysis protocols
  runtime/                   study build selection and measure-sweep orchestration
    engine/                  Go adapter to the two public runtime libraries
  evidence/                  named formal/smoke topic plans, analysis and figures
  micro_validation/          matched micro/pair, conditional ensembles, finite population
  release_clocks/            completed historical release and tail audits
  operator_validation/       measure/Fokker–Planck and numerical comparisons
  kernel_validation/         finite exposure and mixed-increment audits
  scaling/                   compatible small-update comparison assembly
  energy_scan/               offline node/edge fits and single-slot NPZ figures
  manuscript/                historical supplementary evidence tables
```

The Go adapter can remain under `experiments`: it constructs explicit initial states, invokes public solvers and exports inspection/audit results. Model stepping and recommendation factories remain in the two Go libraries. Existing adapter projection and diagnostic instrumentation are retained; new mathematical/statistical estimators go into `src`. No solver is duplicated in the experiments.

`evidence` is the active workflow, with five independently selectable research topics and separate formal/smoke tasks. `main` and `main_smoke` aggregate their respective five topics. See the named-task table in the study guide. One parameter-only plan generates measure and matched micro trajectories, then restricted means, conditional/hierarchical intervals, recommendation contrasts, fixed-r alpha collapse, state comparisons, numerical controls and finite-N robustness figures. Main includes both zeta=1/4 and a common tau=480 matched window with shorter prefixes. Main smoke traverses the same paths with short horizons; it cannot establish a plateau or a paper result. Own endpoint and fixed pair endpoint normalization are explicit alternatives. Numerical diagnostics do not automatically justify extrapolating an infinite mean.

Old suite/scan/clocks CLI routes and the superseded evidence paper/smoke generators are removed. Uncalled historical script modules and their private helpers are removed as well, including the fixed manuscript figure/claim/budget assemblers. Historical readers, T50 estimators and table analyses cited by the supplement remain importable libraries. Scan generators, whole-grid plot menus and historical smoke/minimum/extended grids have been removed. Energy-scan node/edge fits and single-slot figures remain available through offline `fit-slot` / `plot-slot` actions; their archive readers use saved coordinates and protocols without importing scan designs or solvers. Do not extend a historical scan to produce revised-paper evidence when the shared main plan covers it.

The only Python `main()` in `paper_theory` is its canonical `__main__.py`. Launch orchestration receives parsed arguments directly; it does not maintain another CLI or translate arguments into command strings. Runtime calls use their owning module/function rather than old function names or forwarding wrappers. Resume requires the original authenticated run manifest and canonical named profile names. Offline record readers preserve the original data formats independently of generation entry points.

Figure assemblers share study colors/selection through `evidence/figure_style.py`; they do not import each other for helpers. Generic panel decoration/saving lives in `common.plotting`. CSV-only plotting reads frozen exports and writes a separate directory. The retired migration script is archived with the Cache original-data backup. Existing frozen datasets remain readable. The pruned 1794-task main design requires a new run directory; retained task IDs and scientific parameters are unchanged.

Dependency direction is `ehk → common`, `theory → common`, and `paper_theory → common/theory/public runtimes`. Neither source package imports experiments. Importing analysis creates no outputs and does not load a solver. CLI startup configures caches and subprocess limits explicitly. Builds bind actual source/toolchain/backend/binary hashes; plans and raw protocols reject incompatible resumes. Offline analysis needs frozen records, not a runtime executable.

TypedDict definitions are concentrated in the three owning layers above; consumers import them directly rather than re-exporting aliases or introducing local duplicate schemas. `Parameters` retains its validated dataclass behavior, while `ParametersData` describes its serialized fields. Measure and microscopic tasks are distinguished by their literal `kind`; microscopic seeds and identities are required. Stored branches allow a shared pair reference, and resolved records require the pair trace. Exact microscopic energies remain separate from optional projected/binned fields. Array annotations specify dtype, while existing validators enforce shape, conservation and scientific identity.

Type declarations do not coerce data or add defaults. Dynamic decoding, dataclass serialization and public binding adapters use explicit casts at their boundaries. The public binding retains ownership of the nested kinetic request schema. Generic codecs keep unknown payloads generic, with scientific readers supplying the precise task/result view. Main/main_smoke frozen plans, task IDs, record bytes and offline compatibility are verified independently of static types.

Large inputs stay outside Git; generated builds, raw records, tables and figures stay ignored. Historical paper artifacts are optional provenance/data inputs, never code dependencies. `evidence` does not read them. CAS 2026 remains external empirical motivation; this repository does not recreate a CAS dataset. See the [study guide](../experiments/paper_theory/README.md) and [three-repository installation](runtime-dependencies.md).

The remaining duplicated lifecycles and proposed shared `src/common/experiments` boundary are discussed in the [study guide](../experiments/paper_theory/README.md#infrastructure-consolidation-discussion-only). No shared-infrastructure extraction or changes to `experiments/paper` are included in the 2026-10-17 pruning.

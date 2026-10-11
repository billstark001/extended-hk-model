# Theory paper workflows

Run commands from the model repository root after following the [three-repository installation guide](../../docs/runtime-dependencies.md). `src/common` provides reusable tools, `src/theory` provides mathematical definitions, and this package owns scientific designs, execution, fitting, analysis and figures. CAS 2026 supplies the literature motivation; none of these commands replays it.

## One interface and one custom engine

```sh
python -m experiments.paper_theory --help
python -m experiments.paper_theory build --suite validation
python -m experiments.paper_theory plan --suite validation --profile core
python -m experiments.paper_theory run --suite validation --profile core
python -m experiments.paper_theory analyze --suite validation
```

The `validation` preset selects micro/pair comparisons, release clocks, operator comparisons and finite-list audits. `ensembles` adds conditional process ensembles, mesh/quadrature controls, population checks, extended horizons and local crossing scans. The scientific designs live in their corresponding subpackages; these presets use one runner and analyzer. Historical dates and study labels remain in frozen task metadata, not code entry points.

`build` links both external Go source repositories into the one `runtime/engine` binary. It generates module replacements in ignored output directories and does not modify either runtime checkout. Pure Go is the default; `--backend accelerate` or `openblas` is optional. `--build-dir` chooses an immutable build snapshot. Changed source, backend or toolchain requires a new build directory. `--micro-root` and `--meso-root` override configured source paths.

A run is bound to its frozen plan, working-tree source fingerprint and binary hash. Definition-file changes are tolerated only when generated jobs and analysis protocols remain identical. Changed scientific parameters require a new label. Old raw schemas can be read as data; old code and command names have no compatibility wrappers.

## Foreground pipelines and smoke checks

```sh
python -m experiments.paper_theory pipeline --suite validation --profile core --label validation
python -m experiments.paper_theory pipeline --suite ensembles --profile core --label ensembles --baseline-label validation

python -m experiments.paper_theory pipeline --suite validation --profile smoke --label smoke_validation --output-root outputs/check
python -m experiments.paper_theory pipeline --suite ensembles --profile smoke --label smoke_ensembles --baseline-label smoke_validation --output-root outputs/check
python -m experiments.paper_theory status --suite validation --profile smoke --label smoke_validation --output-root outputs/check
```

A pipeline builds, freezes the plan, optionally estimates cost from `--baseline-label`, runs and analyzes. Ordinary failures still permit analysis of completed tasks and preserve the run exit status. An interrupt stops the pipeline and returns 130. Runs use at most four single-thread worker processes, with an exclusive run lock and resumable caches.

Validation defaults to a nine-hour global budget and a 1800-second task timeout. Ensembles defaults to no wall budget and no task timeout. Explicit `--hours`, `--no-wall-budget` and `--job-timeout` control these choices. Smoke designs contain 26 validation or 15 ensemble jobs; they are interface checks, not scientific evidence from a production grid. Use `--families` and `--tiers` to select a declared subset.

## Matched population and measure sweeps

```sh
python -m experiments.paper_theory build --suite population --output-root outputs/population
python -m experiments.paper_theory pipeline --suite population --profile smoke --output-root outputs/population

python -m experiments.paper_theory build --suite scaling --output-root outputs/sweeps
python -m experiments.paper_theory plan --suite scaling --profile paper --output-root outputs/sweeps
python -m experiments.paper_theory run --suite scaling --profile smoke --output-root outputs/sweeps
python -m experiments.paper_theory analyze --suite scaling --profile smoke --output-root outputs/sweeps
python -m experiments.paper_theory plot --suite scaling --profile smoke --output-root outputs/sweeps
```

The matched population study fixes uniform initial law, k=15 and R=10, uses separate opinion/graph RNG streams and starts the pair model from the actual fixed-degree graph projection without renormalization. Exact continuous microscopic energies, variance and discordant-edge fraction are compared with pair results. Process branches are averaged within each independent initial state before aggregating initial states. Its recording policy uses the same engine as the richer conditional audits.

Measure sweeps use the public `smp-kinetic` executable (`--binary` can override it). `scaling` keeps q/alpha fixed while decreasing alpha; `release` varies relative rates and confidence; `population_closure` varies nominal N in the implemented pair closure. The last experiment alone does not establish a microscopic large-N limit. All clocks and comparisons retain physical t versus scaled tau=alpha*t explicitly.

## Existing data, fits and plots

```sh
python -m experiments.paper_theory analyze --suite existing_release --scan-input /path/to/expanded --clock-input /path/to/clocks --output-root outputs/reanalysis
python -m experiments.paper_theory plot --suite existing_release --output-root outputs/reanalysis
python -m experiments.paper_theory plot --suite existing_scaling --scan-input /path/to/expanded --output-root outputs/reanalysis

python -m experiments.paper_theory scan --workflow expanded --scan-input /path/to/expanded --output-root outputs/refit --analysis-only --no-plots
python -m experiments.paper_theory clocks --workflow fit --scan-input /path/to/expanded --output-root outputs/clocks --pilot
python -m experiments.paper_theory clocks --workflow parameter_clocks --scan-input /path/to/expanded --output-root outputs/clocks
python -m experiments.paper_theory clocks --workflow parameter_clocks --clock-input /path/to/clocks --output-root outputs/redraw --plots-only
python -m experiments.paper_theory clocks --workflow plot_parameter_summary --clock-input /path/to/clocks --output-root outputs/redraw
```

The release fitting, validation, reduced-mixture, branch-audit and refinement workflows are selected by `clocks --workflow`; use the selected workflow's `--help`. `scan --workflow` supports `baseline`, `expanded` and `scaling`. Expanded scans keep the original triangular 720-case grid and resumable per-case/shard storage. Passing `--analysis-only` never simulates. Generation uses the current public adapter and requires a separate output location for changed provenance.

The main mean clock is the finite-window integral of nonnegative normalized remaining energy. Rebounds are retained, and removed negative area and plateau status are reported. Infinite empirical-fit means are separate extrapolated observables. First-crossing T10/T50/T90 remain diagnostics with their own historical plateau/amplitude protocol. Energy-only confirmation is checked independently at every required stopping checkpoint. Mean-curve T50 and mean individual T50 are distinct; conditional bootstrap units are explicit. Refactoring does not substitute one definition for another.

## Manuscript evidence

```sh
python -m experiments.paper_theory review --suite validation --label validation
python -m experiments.paper_theory review --suite ensembles --label ensembles --baseline-label validation
python -m experiments.paper_theory figures --run-root /path/to/frozen/runs --review-root /path/to/reviews --output-root outputs/manuscript
python -m experiments.paper_theory audit --workflow claims --run-root /path/to/frozen/runs --review-root /path/to/reviews --figure-input /path/to/manuscript --output-root outputs/audit
python -m experiments.paper_theory audit --workflow budgets --run-root /path/to/frozen/runs --output-root outputs/audit
```

Study-owned review functions compute scientific summaries; manuscript modules assemble figures and check historical claims. The frozen paper figure selection still needs its original plans, bindings, status and per-task raw trajectories. Small CSV summaries cannot replace those inputs. Historical numerical assertions are tied to their recorded datasets and are not general runner invariants.

All generated files and external-data links remain ignored. See [migration provenance](MIGRATION.md) for the source-to-package map. Neither an archival upload nor a full production rerun is part of a code smoke check.

## Fresh evidence for the proposed revision

This entry point starts with parameters and seeds in an empty output directory. It builds the two public-runtime adapters, freezes the combined design, generates all records, analyzes them and saves five figure groups plus `report.md`, `report.json`, coverage and SHA256 provenance tables. No `existing_*` input, previous run label, frozen historical job ID or manuscript artifact is required.

```sh
PYTHONPATH=src:. python -m experiments.paper_theory pipeline --suite evidence --profile smoke --output-root /tmp/ehk-evidence-smoke
# Plan the formal design without starting it:
PYTHONPATH=src:. python -m experiments.paper_theory plan --suite evidence --profile paper --output-root /path/to/new-evidence
# When ready for the full design:
PYTHONPATH=src:. python -m experiments.paper_theory pipeline --suite evidence --profile paper --output-root /path/to/new-evidence
```

Use `--micro-root`/`--meso-root` for source checkouts outside the default sibling locations. Install the dependencies described in `docs/runtime-dependencies.md` first. The paper profile is a large study; planning does not estimate its cost or authorize launching it. `--workers` controls concurrent independent jobs, with one Go thread per job; `--limit` limits each family and reports incomplete coverage.

The mean clock is a finite-window survival integral in tau=alpha*t. Long-window endpoint references are reused for prefix areas; plateau checks, tail20 sensitivity and weak-release status are retained. Conditional micro intervals resample process branches at one fixed initial state. Population intervals resample independent initial states and their branches, giving each initial state equal weight. Recommendation contrasts use the shared initial state/process-seed design. Fixed pair endpoint normalization is a separately labeled sensitivity control; these data do not imply an infinite-mean theorem.

The formal design covers HK/Deffuant, Random and Opinion/Structure zeta=1/4; parameter strips, fixed-r decreasing-alpha collapse; tau=480 with 30/120/240/480 prefixes; grid/quadrature/dt controls; identical-law nominal-N closure checks; matched micro mesh/quadrature controls; and N=500/1000/2000 with fixed k=15/R=10 and four independent initial states × four conditional branches. Short smoke uses two initial states × two branches and N=40/80, solely to exercise generation, analysis and plotting.

Figures `mean_release`, `scaling_collapse`, `matched_micro_means` and `population_robustness` provide the proposed primary evidence. `numerical_controls`, full state comparison tables, prefix and plateau diagnostics can support supplementary material. Existing T50 workflows remain available for sensitivity and historical reproduction; the historical `figures` command still addresses its frozen dataset and is not used by `evidence`.

Use `--horizon` on the evidence paper profile to predeclare a longer common window in a new output directory when the tail diagnostics require it. The default is 480; stopping remains fixed-window so scaling and prefix comparisons have matching time domains.

## Current main figures, with historical data reads prohibited

`evidence.design.design()` is a facade. `paper_design.paper_design()` preserves
the original evidence plan; `main_design.design()` defines the three remaining
main figures after removal of the CAS overview figure. The `main` profile uses
epsilon=.2/.4/.8, alpha=.02, all five weighted recommenders and both dynamics.
It adds the edge/node mean-area log ratio, representative raw tails, two initial
graphs with eight conditional branches each at N=500/B=161/Q=33, and matched
N=250/500/1000/2000 distributions, variance, marginal-referenced edge pairing,
restricted areas and closure errors for Random/Opinion4 at r=2 through tau=120.
Population runs use B=161/Q=33 and four independent
initial states with four branches each. Analytic-initial mesoscopic runs have
a tau=480 cap, with joint state/energy stopping after tau=120 and 40 tau of
stationarity. Microscopic/pair ensembles retain the common tau=480 window.
Observed endpoint areas, unavailable later prefixes and stopping diagnostics
are explicit; curves are not extended after stopping. Alpha collapse compares
raw curves on their common observed interval. Prefix, grid, quadrature and
physical-step controls remain.

The dedicated entry point below refuses existing scientific/build caches by default,
builds the solvers from the two source checkouts, and exports only five PDF/PNG
figure pairs per confidence radius plus small tables and manifests. The baseline
epsilon=.4 figures are at the root; epsilon=.2/.8 versions are in separate
subdirectories named in the figure manifest. Raw JSON/NPZ, build files and
runtime caches stay under the new work directory. Small executables link and
run from a temporary APFS directory to avoid macOS exFAT mapping failures;
verified provenance copies remain in Cache. On macOS, Accelerate is the default
BLAS backend; select `--backend purego` for the portable fallback.
It disables dotenv and uses
a Python read audit hook that permits code, dependencies and generated files;
old inputs and resolved symlink targets are rejected. The Go scientific
executables receive generated requests on stdin; the audit records subprocess
commands but does not claim to trace all Go kernel-level reads.

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. python -m experiments.paper_theory.evidence.run_main \
  --profile paper --workers 4 \
  --work-dir /Volumes/Cache/extended-hk-model-paper/NEW_EMPTY_RUN \
  --publish-dir outputs/experiments/paper_theory/main_figures/NEW_RESULT \
  --model-root "$PWD" \
  --micro-root ../social-media-models --meso-root ../social-media-mesoscopic-models
```

Use `--profile smoke` for the complete short pipeline, with N=40/80 and tau=.4.
It checks all plotting paths and matching controls; these figures are not
scientific results. Formal output names are `mean_recommendation_clocks`,
`mean_micro_validation`, `population_robustness`, `scaling_collapse` and
`numerical_controls`. Means are unclipped integrals; all figures retain the
finite-window/end-reference qualification. Tail stability alone does not prove
an infinite-time mean. The manuscript launcher is
`artifacts/theory_main_figures_20261012/run_full.sh` in the paper repository.

Explicit `--resume` continues the same work and publication directories. It
verifies the frozen design, source/request-schema hashes, backend and exact
original executables, restores executable copies to APFS, skips valid completed
tasks and reruns incomplete tasks from their initial state. Analysis, plotting
and partial exports can also be resumed. Only worker count may change without
starting a new scientific run. Whole-run OS locks prevent competing launchers;
stale legacy task locks are recovered only after their owner PID has exited.
Ctrl-C/SIGTERM stop new task submissions and kill active solver subprocesses.
Each `attempt-*` directory records status and a separate read audit. Retain the
original `build/` and `bin/` copies and unchanged source checkouts for resume.
The paper launcher exposes this as `bash run_full.sh resume RUN_DIRECTORY` and
appends output to the same `stdout.log`.

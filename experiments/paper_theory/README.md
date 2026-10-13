# Current theory main figures

`src/theory` owns mathematical definitions; the external Go repositories own
model stepping. This package owns the frozen main design, execution, offline
analysis and figures. The active profiles are **main** and **main_smoke** only.
The old evidence paper/smoke designs and old suite/scan/clocks CLI routes were
removed. Reusable historical readers, fits and internal validation functions
remain libraries. Uncalled historical script modules and private helpers were
removed, including old fit-audit/clock plotting scripts and fixed manuscript
figure/claim/budget assemblers. Only the canonical CLI defines `main()`.

```sh
python -m experiments.paper_theory --help
python -m experiments.paper_theory plan --profile main --work-dir /path/to/new-run
python -m experiments.paper_theory run --profile main --work-dir /path/to/new-run --publish-dir /path/to/export
python -m experiments.paper_theory resume --run-dir /path/to/run --publish-dir /path/to/export
python -m experiments.paper_theory status --run-dir /path/to/run
python -m experiments.paper_theory analyze --run-dir /path/to/run
python -m experiments.paper_theory plot --run-dir /path/to/export --tables-only
```

Run/resume build or restore verified executables. Plan, status, offline analyze
and plot do not launch scientific solvers. Use `--micro-root`, `--meso-root` and
`--backend` for generation. Analysis reads the saved plan rather than generating
one from current sources. Changed scientific source/backend/parameters still
require a new run. `main_smoke` is a short software exercise, never paper evidence.
Resume requires the original `run_manifest.json` and canonical main/main_smoke
profile names. Old profile aliases and inferred manifests are removed. Legacy
JSON/NPZ data remain readable by the offline analysis/migration functions.

The existing main plan remains exactly unchanged: 828 analytic-initial measure
tasks and 1056 microscopic tasks across epsilon=.2/.4/.8. Primary B=161/Q=33,
alpha=.02, k=15/R=10; matched N=500 ensembles have two initial graphs and eight
branches through tau=480. Population robustness uses N=250/500/1000/2000,
four independent states and four branches through tau=120. Only analytic
measure tasks use the original joint state/energy stopping rule. The original
plan fingerprint and all scientific task IDs are retained.

## Storage and recovery

Keep Python, local control logs, dependency caches and executable mappings on
local APFS. `--control-dir` chooses a persistent directory outside the bulk run;
default is under model `outputs/runtime`. The paper launcher uses this local
Python environment and logs to that control directory. Never delete it while
pending commits remain. `--spool-mib` defaults to 4096: each of at most four
workers caps its binary response at half its share of this allowance, leaving
room for its staged record. Oversized responses fail explicitly rather than
consume unbounded local storage. Committed results are removed from the spool.

Plan/binding/status metadata remain small JSON. Measure arrays remain NPZ. New
micro records are frame-streamed `.msgpack.lz4` with typed binary arrays,
float64 precision, shape/dtype metadata, a content hash and LZ4 checksum. Python
still reads legacy `.json` records. Go sends binary vectors directly, avoiding
large JSON strings and boxed numeric lists. Frequent micro samples retain exact
continuous energies/variance/discordance; full binned states are computed only
at declared snapshots/endpoints. Missing binned energies are explicitly absent,
never represented by fabricated zeros.

Deterministic pair traces are computed once per initial-state/physical-setting
cohort and referenced by SHA256 from process branches. In the main plan there
are 228 such dependencies; they do not add or change scientific task IDs.
`micro/pairs/` must remain with `micro/raw/`. Source, request schema and executable
provenance remain bound to the frozen run.

An external I/O error waits with cancellable backoff. It is not classified as a
truncated record. A completed result is staged locally, copied to a same-volume
temporary file, flushed, checksum-verified and renamed. Interrupted pending
commits are recovered before scheduling tasks; a cable disturbance during
publication does not require recomputing the task. Ownership markers prevent
writing into a different run appearing at the same path.

Independent scientific failures are recorded and the other tasks continue.
Only subprocess transport/timeouts are retried (three attempts, same IDs/seeds).
Protocol/source identity errors and Ctrl-C/SIGTERM cancel all active children.
Partial coverage may produce diagnostic tables, but publication plots and final
export require the complete plan. Each attempt and task transition is logged
locally. Recovery remains at task boundaries; solver timestep state is not
checkpointed.

## Figures from local exports

`plot --tables-only` generates `figures_csv/` without a build or raw-data read.
Original figures and CSVs are retained. The derived manifest records input
hashes and the frozen plan fingerprint. Each epsilon has separate plots:

- Recommendation relative clocks and shifts from Random, with q=0 isolated and
  unconfirmed terminal points marked without supported connecting lines.
- Absolute observed release areas, shown on logarithmic axes.
- Matched node/edge area intervals and pair predictions.
- Population structure, separate closure errors and release areas. Near-zero
  variance/pairing is summarized explicitly rather than magnifying roundoff.
- Fixed-r scaling errors and numerical-control heatmaps.

CSV exports cannot recover raw energy bands or rho/E snapshots. Their original
exports remain available; those panels need raw trajectories for redesign.
Areas retain rebounds/negative excursions, own endpoint references and actual
observed horizons. Terminal stability alone does not establish an infinite mean.

## Offline migration

```sh
python -m experiments.paper_theory migrate --run-dir /path/to/raw-dataset --output-dir /separate/new-dataset
python -m experiments.paper_theory analyze --run-dir /separate/new-dataset
```

Migration is explicit, task-resumable and never executes a solver. It validates
source records, copies measure NPZ byte-for-byte, converts micro vectors without
precision loss, verifies every scientific field and deduplicates matching pair
traces. Original files, task IDs, frozen plan and binary/source bindings remain
unchanged. The migration manifest maps original and converted SHA256 values.
Derived tables are regenerated with the original analysis/bootstrap protocol.
A figures/CSV-only export cannot be migrated into full raw data. No migration
of the 20261012 formal run has been performed while its external disk is absent.

The CAS study is literature motivation, not a dataset supplied to these runs.
See [installation](../../docs/runtime-dependencies.md) for the three checkouts.

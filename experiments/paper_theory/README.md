# Theory manuscript experiments

The design follows the 2026-10-17 theory structure proposal: matched microscopic
and mesoscopic comparisons, relative control, recommendation, and collective
organization. `src/theory` owns mathematical definitions; the external Go
repositories own model stepping. This package selects experiments, freezes
plans, executes them, and assembles offline tables and figures.

## Named tasks

Use `--profile NAME` for `plan` or `run`. Formal and smoke tasks have separate
entry modules, `evidence/main_design.py` and `evidence/main_smoke_design.py`.
`main` selects all five formal topics; `main_smoke` selects all five short
software exercises. The latter never supplies manuscript evidence.

| Research topic | Formal task | Smoke task | Formal trajectories | Manuscript use |
| --- | --- | --- | ---: | --- |
| Recommendation response | `recommendation` | `smoke_recommendation` | 330 measure | RQ2 response; proposed Fig. 4 and supplementary confidence/tail comparisons |
| Relative control | `scaling` | `smoke_scaling` | 450 measure | RQ1; proposed Fig. 3 and supplementary errors |
| Numerical controls | `numerics` | `smoke_numerics` | 60 measure | Supplementary grid, quadrature and physical-update controls |
| Microscopic / mesoscopic matching | `matched` | `smoke_matched` | 288 micro | Proposed Fig. 2 and supplementary distributions/errors |
| Population structure and variation | `population` | `smoke_population` | 768 micro | Proposed Figs. 1/5 and supplementary population comparisons |

Shared trajectories are deduplicated in the aggregate. The new `main` has
**1794 tasks: 738 measure and 1056 micro**. The new `main_smoke` has **142 tasks:
62 measure and 80 micro**. Topic selection also controls analysis and plotting;
a measure-only topic does not need micro data, and vice versa. Population
records supply the snapshots and repeated processes needed by the proposed
structure and W/I analyses. The five revision figures now use these records through
a portable CSV analysis bundle; the estimators belong to `src/theory/variation.py`.

```sh
python -m experiments.paper_theory --help
python -m experiments.paper_theory plan --profile scaling --work-dir /path/to/new-run
python -m experiments.paper_theory run --profile scaling --work-dir /path/to/new-run --publish-dir /path/to/export
python -m experiments.paper_theory resume --run-dir /path/to/new-run --publish-dir /path/to/export
python -m experiments.paper_theory status --run-dir /path/to/new-run
python -m experiments.paper_theory analyze --run-dir /path/to/dataset
python -m experiments.paper_theory plot --run-dir /path/to/dataset --output-dir /path/to/figures
python -m experiments.paper_theory plot --run-dir /path/to/export --tables-only
```

Theme names omit the former `main_` prefix; smoke theme names keep `smoke_`.
The aggregate names `main` and `main_smoke` are unchanged. Existing frozen
`main_TOPIC` / `main_smoke_TOPIC` datasets remain readable and resumable with
their saved names; new CLI selections use the shorter names.

A run directory freezes its selected task name. Resume restores that name from
`run_manifest.json`; an explicit different name is rejected. If a bulk root
contains several datasets, pass the desired `evidence/NAME` directory to offline
commands. Run/resume build or restore verified executables. Plan, status,
analyze and plot never launch scientific solvers. Generation accepts
`--micro-root`, `--meso-root` and `--backend`.

The formal physical parameters are unchanged: epsilon=.2/.4/.8, alpha=.02,
B=161/Q=33, degree=15 and recommendation-list length=10. Matched N=500 ensembles
have two initial states and eight conditional processes through tau=480;
population comparisons have N=250/500/1000/2000, four initial states and four
processes through tau=120. Only analytic measure tasks use joint state/energy
stopping. Process replicates remain conditional on their initial state.

## Retained and removed designs

The proposal retains historical evidence in `revision_validation.tex` and
`channel_diagnostics.tex`. The completed 762-job validation and 346-job follow-up
batches are defined by their own authenticated frozen plans. Historical generation
modules and their separate executor have been removed. `theory.runtime.historical`
validates their saved job/build identities and reads gzip records without rewriting
them; `paper_theory.historical` and the manuscript report modules assemble the
retained supplementary tables. Their saved analysis protocols remain distinct from
`evidence/protocol.py`, which defines the current manuscript study.

Removed code includes `energy_scan` baseline/expanded/channel-scaling
generation and whole-grid plot menus, the old alternative scaling/release/
nominal-population sweeps, the duplicate microscopic population design, old
parameter-clock fitting and plotting, historical smoke/minimum/extended
variants, unrun robustness/alpha/confidence extensions, and superseded historical
figure assemblers. The old analysis libraries retain the tables used by the
supplement; they no longer emit those obsolete plot menus.

The nominal-N analytic closure scan had no retained figure or supplementary
evidence role. Its 90 exclusive formal tasks (20 smoke) are removed. Overlapping
recommendation/scaling/numerical tasks remain. Retained scientific task IDs and
settings are unchanged; the selected roles and aggregate plan fingerprints
change. **Use a new run directory for the revised design.** Offline analysis
and plotting still read the original frozen 1884-task dataset and its saved
protocol, without regenerating a plan or rewriting raw records.

## Retained energy-curve fitting and single-slot plots

Energy-scan fitting remains available as an offline capability. A slot is one
`(dynamics, epsilon, alpha, q/alpha)` combination. `fit-slot` exports descriptors,
node/edge low-parameter fits and the edge mixture fit; `plot-slot` additionally
renders one PDF/PNG showing raw energies, fitted curves and release quantiles.
Neither command needs a full grid, current solver, build or simulation.

```sh
python -m experiments.paper_theory fit-slot --run-dir /path/to/raw_shards/hk_alpha_0.01.npz --dynamics hk --epsilon 0.4 --alpha 0.01 --ratio 3.125 --output-dir /path/to/slot-output
python -m experiments.paper_theory plot-slot --run-dir /path/to/raw_shards/hk_alpha_0.01.npz --dynamics hk --epsilon 0.4 --alpha 0.01 --ratio 3.125 --output-dir /path/to/slot-output
```

`--run-dir` accepts an expanded shard, its `raw_shards` directory or parent,
or a baseline `energy_curve_parameterization.npz`. Coordinates must match the
saved archive and protocol. Baseline physical time is converted to tau; coarse
recording restores the saved first-step energies using the saved physical dt.
The output contains `descriptors.csv`, `low_parameter_fits.csv`,
`edge_timescale_fits.csv` (when identifiable), and `slot_manifest.json` with the
source hash and selected coordinates. Use a separate output directory per slot.

The retained `expanded/fit/node.py` supports Gompertz, exponentiated Weibull and
rational quadratic Bezier quantile fits for both channels.
`expanded/fit/edge.py` supports the fast KWW component plus observed node
survival, including the unidentifiable fast shape at q=0. Their public
`low_parameter_fits` and `fit_edge_timescales` functions remain importable;
`parameterize` accepts the supplied saved slots rather than requiring the old
complete scan. Empirical fits retain the historical finite-window tail
normalization and do not supply new manuscript evidence by themselves.
`expanded/config.py` now contains fit conventions only; scan designs, generation,
full-grid summaries and parameter-clock workflows remain retired.

## Storage and recovery

Keep Python, control logs, dependency caches and executable mappings on local
APFS. `--control-dir` chooses a persistent directory outside the bulk run;
default is under model `outputs/runtime`. Never delete it while pending commits
remain. `--spool-mib` defaults to 4096 and limits aggregate staging. Committed
results are removed from the local spool.

Plan/binding/status metadata use small JSON; measure arrays use NPZ. Micro
records use frame-streamed `.msgpack.lz4` with float64 arrays, shape/dtype,
content hashes and LZ4 checksums. Legacy JSON/gzip readers remain. Shared
predictions live in `micro/pairs/` and must accompany `micro/raw/`; the formal
microscopic design still has 228 pair dependencies, outside the 1056 process
IDs. Scientific source/backend/request changes require a separate run.

External I/O errors wait with cancellable backoff. Completed results are staged
locally, copied to a same-volume temporary file, flushed, verified and renamed.
Resume recovers pending commits before scheduling. Independent scientific
failures are recorded while other tasks continue; subprocess transport/timeouts
retry with the same IDs and seeds. Protocol/source failures and interruption
cancel active children. Partial analysis is diagnostic; publication plots and
final export require every task in the selected profile.

## Existing Cache data

On 2026-10-17, `/Volumes/Cache/extended-hk-model-paper/20261012_main_figures/`
contains the migrated formal dataset under `migrated/20261012_040418_paper_8arWRU`:
828 NPZ measure records, 1056 binary micro records and 228 shared predictions.
The migration receipt records all 1884 records verified, unchanged task IDs,
identical scientific fields/tables and matching rendered figures.

The old 1056 JSON records remain under
`/Volumes/Cache/extended-hk-model-paper-original-data/20261012_040418_paper_8arWRU/`.
At the user's request, the retired `migrate.py` was copied to that backup root
and removed from this repository, along with the CLI command and migration-only
types/tests. The archived script is a historical copy, dependent on the original
package context; it is not a standalone supported command. No raw data was
removed or reconverted.

All publication plots read the portable CSV bundle and use one implementation in
`evidence/figures.py`. The old `main_figures.py` / `table_figures.py` split is removed.
`--tables-only` remains accepted and selects this same complete implementation.
Areas retain rebounds/negative excursions, own endpoint references and observed
horizons. Terminal stability alone does not establish an infinite mean.

See [installation](../../docs/runtime-dependencies.md) and
[architecture](../../docs/architecture.md). The CAS study is literature
motivation, not a required raw dataset for the theory workflow.

## Shared experiment lifecycle

Both `paper_theory` and `paper` now use `src/common/experiments` for named-task
selection, immutable plans/manifests, coverage, exclusive sessions, recovery locks,
stage dispatch and task completion journals. Execution reuses `common.runtime`
for bounded threads/processes, cancellation, transport retries and storage staging.
`common.pipeline.Context` continues to cache observable computations inside a
process; it does not schedule durable experiment tasks.

The study modules register parameter designs and scientific analysis/figure stages.
Model adapters own the request/result contracts and persistence: mesoscopic NPZ,
microscopic binary/shared-pair records, original-paper SMP simulation directories
and SQLite statistics. Historical audit data uses a read-only adapter. The old
executors, workflow dispatchers, historical design generators and forwarding
modules have been removed; there are no code aliases for those modules.

Existing CLI arguments, binary wire protocols, raw record layouts, SQLite columns,
NumPy blob encodings, seeds and frozen task identities remain unchanged. Existing
frozen plans are authenticated and read on resume rather than regenerated from
current designs. Previously saved long topic names are interpreted only at this
metadata boundary; new CLI task selections use the shorter names.

Task completion displays the count, per-task elapsed time, total elapsed time and
a throughput-based ETA. Cache hits are identified separately and do not provide
computational timing samples. Failed/deferred tasks remain missing from coverage;
partial statistics or simulation batches do not become complete status reports.

`--binding-batch` lets the owning task adapter opt into its public runtime's native
batch launcher: mesoscopic measure tasks in `paper_theory`, and SMP scenarios in
`paper.run`. The default uses the common task scheduler. The shared experiments
package knows only a generic batch callback, never either runtime API. Adapters
validate the returned identities and completion markers, persist the same record
formats, and isolate native launchers in cancellable process groups. Native batch
completion timing includes the enclosing batch latency; it is not solver-only
per-request CPU time. Micro/shared-pair theory tasks retain their binary adapter.

No scientific protocol is merged merely because it has similar settings. Current
bootstrap intervals, historical resampling protocols, original-paper observable
selection and SQLite estimators stay with their studies.


## Run, analyze, plot

There are three computational stages. `run` owns simulation/binary execution;
`analyze` owns extraction, comparisons, bootstrap, fitting and variance components;
`plot` only renders local CSV observations and statistics. Plan/build/resume remain
run lifecycle operations. Binary requests, source manifests and raw codecs are
unchanged. Original run/resume CLI invocations still perform all three stages.
Use repeated `--phase run|analyze|plot` options to select stages explicitly.

```sh
# Same full workflow as before; derived data and figures are placed locally.
python -m experiments.paper_theory run --work-dir /Volumes/Cache/new-run --publish-dir /local/results
# Run simulations only.
python -m experiments.paper_theory run --work-dir /Volumes/Cache/new-run --publish-dir /local/results --phase run
# Export existing frozen data while its disk is connected.
python -m experiments.paper_theory analyze --run-dir /Volumes/Cache/dataset --output-dir /local/results
# Recompute statistics from the local CSVs with the raw volume disconnected.
python -m experiments.paper_theory analyze --run-dir /local/results
python -m experiments.paper_theory analyze --run-dir /local/results --force
# Full figures, including curves/bands and field snapshots, from local CSV only.
python -m experiments.paper_theory plot --run-dir /local/results --output-dir /local/figures
```

A bundle contains `plan.json`, `data_manifest.json`, `data/`, `tables/`,
`report.json`, and `analysis_manifest.json`. Keep this bundle on local storage.
`data/grids.csv` describes each native bin's bounds and center.
`data/tasks.csv` preserves parameters, initial/branch IDs, exact string seeds and
shared-pair relationships. Per-record CSVs preserve every recorded scalar, the
separate energy/observable timestamp grids and all native field arrays. Matrix
columns are native grid coordinates; only exactly zero rows are omitted, with
explicit shapes to restore them. CSV gzip compression is deterministic. No
thresholding, extra interpolation or tail extension is introduced by extraction.

Incremental extraction reuses unchanged record CSVs after checking raw and local
CSV hashes. Derived blocks are cached by physical comparison group and artifact
kind (measure summaries, micro summaries, conditional curves, representative fields,
population variation). Keys include selected task identities, relevant CSV hashes,
analysis settings, record metadata and estimator/reader/library source hashes. A changed figure style does
not invalidate analysis; a changed estimator invalidates its affected blocks.
`--force` recomputes extraction and derived blocks when raw data is present, or all
derived blocks from CSV when the raw volume is absent. Complete cache blocks survive
later analysis failures; partial blocks are never reused. Cache hit/compute counts
are recorded in `analysis_manifest.json`. `analysis_cache/` is optional for plotting.

The unified renderer in `evidence/figures.py` owns the five main figures; previous
main-figure layouts and the competing paper-repository renderer are removed.
Figure 1 uses five square panels and the EHK network-snapshot utility for an
explicit event schematic, followed by matched node/edge fields and remaining
energies from one four-process population group. Figure 2 uses a 2-by-4 outer
grid, with two spanning area panels; its legend explains filled HK and hollow
Deffuant markers. Figure 3 uses a 2-by-4 grid across all three confidence radii:
reference curves, hollow largest-update markers and calculated update envelopes
replace overlapping line-style-only comparisons. Envelopes interpolate only on
the common observed interval and are not confidence bands. Figure 4 uses a 2-by-3
grid and retains both original log ratios and a second Random-baseline version:
the original equal-area line is transformed to minus the Random log ratio.
Figure 5 contains structure means and the node/joint-edge CDF W/I comparisons.

All newly rendered panels use a manuscript-width canvas (7.1 inches) with
consistent 8-point axis/panel titles, 7-point ticks and 7-point legends. Other
confidence radii, absolute areas, numerical controls, terminal W/I/B² and full
population CDF times remain supplementary outputs, at the same typography. Each figure has PDF, PNG and a plain-text
scientific caption; the manifest identifies the exact CSV and renderer inputs.

Population CDF norms use dx for node CDF and dx² for the joint directed-edge CDF.
W/I/B² use balanced initial/process groups; negative corrected I/B² estimates are
retained, MSE=W+B² and I=V_P+H+2C are distinct identities. Their shaded ranges are
leave-one-initial-out sensitivities, not 95% confidence intervals. Structure panels
use exploratory bootstrap of whole initial groups, retaining every process branch.
Only saved observations/times are available offline; new unrecorded observables
require the source data. Old summary-only exports must be upgraded by `analyze`
once while their raw volume is available; plot never falls back to reading raw.

The final display pass uses 60% data/error/reference strokes, 55% filled-marker
radii, 65% hollow-marker radii and 75% frame/tick strokes; Figure 1(a) is excluded.
Legend samples receive the same scaling. Fonts and manuscript-size canvases stay
fixed. Frozen supplementary vector assets can be restyled with the independent
`evidence/vector_style.py` helper (optional `pypdf`), preserving text, page bounds,
open data paths and marker centers. It does not restore retired plot generators.

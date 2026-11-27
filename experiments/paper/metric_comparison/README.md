# Canonical indices on matched intervention trajectories

This offline workflow evaluates the paper's single `I_p/I_s/I_h/I_w` protocol on saved trajectories and compares it with the independent P/E observations from the intervention studies. It never launches a simulation. All analysis inputs, tables and per-path receipts use SQLite; CSV files are exports for inspection.

| Responsibility | Entry and output |
| --- | --- |
| Run | Recommendation/staged/variant studies write native histories and authenticated arm observations in their `analysis.sqlite`. |
| Analyze | `collect-raw` uses the canonical `MeasurementEvaluator`, B=4081, at the prepared states and on each full adaptive trajectory. |
| Collect figure data | `prepare` authenticates one SQLite portable bundle layout, inventories raw paths and commits its tables; `compare` joins collected indices to those identities. The report commits matched states and paired-effect tables. |
| Render | `report` renders its scientific figures from those tables. CSV is never a fallback input. |

## Prepare

```sh
python -m experiments.paper.metric_comparison prepare \
  --bundle /path/to/staged-bundle --bundle /path/to/timing-bundle \
  --output /path/to/new-preparation
```

Only schema-2 portable bundles are supported: `data/manifest.json`, `data/study/plan.json`, `data/study/analysis.sqlite`, bindings and table SQLite databases. Plans, task fingerprints, binary provenance, database hashes and unique observation steps are checked. Preparation copies the recorded P/E observations without reinterpreting them as `I_p/I_s`.

The preparation's `analysis.sqlite` holds `points.csv.gz`, `paths.csv` and `tasks.csv` as named tables; corresponding CSV exports remain available. Raw locations come from the bundle's recorded raw root and each arm's `raw_record`. Explicit `--raw-root PROFILE=DATASET` overrides a moved mount. Same-name tasks from different generations are rejected. Missing raw inputs remain pending.

## Analyze and resume

```sh
python -m experiments.paper.metric_comparison collect-raw \
  --preparation /path/to/preparation --output /path/to/collection --workers 4
```

The formal path uses the same formula, 4081-bin density evaluation and adaptive sampler as the original-paper analysis. `I_w` integrates the complete saved prefix from step zero and includes `(1-I_p(T))*I_h(T)` by default. `--no-endpoint-complement` creates an explicitly different configuration. A 16-step observation export is never substituted for the adaptive integral.

`--max-paths N` limits a diagnostic run; its status remains partial when prepared paths remain unmeasured. Process workers compute individual paths, reuse bounded canonical scalar caches, and return receipts to a single parent-process SQLite writer. Each completed path commits before it is reported complete. Scientific source hashes include the metric implementation and actual trajectory, event and snapshot readers. Receipts also bind metadata, graph snapshots, trajectory manifests and event data. Empty read-side WAL files and SHM lock bytes are excluded; nonempty WAL contents remain bound.

Native stdout observations are read from the authenticated study database. An `observations.json` raw sidecar is not required. Saved RNG seeds are compared as integer identities, preserving their original string provenance. Runtime defaults may add recommendation parameters; all explicitly declared parameters must match.

The collection stores `indices.csv.gz` and `pathways.csv` in `analysis.sqlite`, with CSV exports and a configuration/coverage manifest. Arbitrary historical metric-table import and old bundle layouts are unsupported. Cached receipts require the same scientific protocol and raw identity.

## Collect tables and plot

```sh
python -m experiments.paper.metric_comparison compare \
  --preparation /path/to/preparation --collection /path/to/collection \
  --output /path/to/comparison
python -m experiments.paper.metric_comparison.report \
  --preparation /path/to/preparation --comparison /path/to/comparison \
  --output /path/to/report
```

The report retains matched endpoints, actual horizons, shared-prefix first observed passages and paired intervention units. P/E thresholds, confidence-component outcomes and opinion peak counts remain independent diagnostics. Missing values are not zero-filled. Reports reject incomplete coverage unless an interim report is explicitly requested. SQLite table digests and exported-file hashes are checked before downstream analysis.

One command performs collection, alignment and reporting:

```sh
python -m experiments.paper.metric_comparison.run \
  --preparation /path/to/preparation --output /path/to/pipeline --workers 4
```

Use `--collection /path/to/existing-collection` to resume authenticated receipts. Preparation, alignment and report destinations must be fresh. Collection can reuse its authenticated path receipts. Changed frozen inputs are rejected. For CPU reproducibility, limit BLAS/OpenMP threads to one while parallelizing paths. The independent `kde_approximation` study computes explicit exact references with the same kernel; it does not change this formal 4081-bin route.

Preparation, collection jobs, results, receipts and manifests use declared dataclasses internally. JSON and SQLite decoding create those objects at the boundary; scientific tables remain DataFrames. Preparation exports its complete scientific tables to SQLite. Collection reports committed-path coverage.

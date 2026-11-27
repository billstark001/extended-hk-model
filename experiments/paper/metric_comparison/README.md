# Comparing two metric families on matched intervention records

This package first prepares a verifiable comparison cohort, then collects existing metrics or imports metric tables. Preparation, collection and exact alignment have separate commands. They read frozen experiments and existing outputs without starting simulations or changing metric definitions.

## Prepare a comparison

Run from the model repository and choose a new preparation directory:

```sh
python -m experiments.paper.metric_comparison prepare \
  --bundle /path/to/staged-bundle \
  --output /path/to/new-preparation
```

Repeat `--bundle` to include timing or implementation-comparison bundles. Select one generation per study: mixing duplicate tasks from older six-arm and newer nine-arm bundles is rejected. Bundles contain `data/manifest.json`, `data/study/plan.json`, records and runtime bindings.

The authenticated older timing-export layout is also supported. Its `source_manifest.json` provides hashes for plans, records and bindings under `source/`; preparation can read it without the raw volume or another export. Combine staged and timing bundles in one preparation:

```sh
python -m experiments.paper.metric_comparison prepare \
  --bundle /path/to/staged-bundle --bundle /path/to/timing-bundle \
  --output /path/to/new-staged-timing-preparation
```

Preparation checks all manifest SHAs, frozen-plan and task fingerprints, per-arm executable provenance and unique observation steps. It expands the recorded `P`, `E`, `C_edge` and `C_pair` values without recomputing them. Outputs include:

- `points.csv.gz`: observations keyed by `profile,task_id,stage,arm,step`, retaining values, relative steps and plan/executable/raw-record provenance.
- `paths.csv`: each arm's start/end steps, recorded endpoint classification, parameters, initialization seed, required raw directory and basic file availability. Staged reference paths use `stage=reference,arm=reference`.
- `tasks.csv`: frozen task fingerprints and complete metadata; two `legacy_*_template.csv` files provide import headers.
- `preparation.json`: source bundles, coverage, output hashes and the metrics awaiting collection.

Raw directories come from each bundle's `source_dataset` and each arm's `raw_record`. Missing volumes leave `pending_legacy_metrics` and the missing filenames visible. If a mount has moved, use `--raw-root staged=/new/path/to/staged` in a new preparation directory; other profiles work similarly. Preparation checks basic file availability. Collection additionally validates readability, history coverage and state consistency.

## Collect from raw histories

```sh
python -m experiments.paper.metric_comparison collect-raw \
  --preparation /path/to/preparation \
  --output /path/to/collection --max-paths 2
```

Omit `--max-paths` to process every prepared path. Collection uses persistent `RawSimulationRecord` histories and evaluates `observables.the canonical `MeasurementEvaluator` for `I_p,I_s,I_h` at the exact prepared states. Full-path `I_w` uses the shared evaluator and `pathway_index`, retaining their adaptive sampling. Chronological signed trapezoids include `(1-I_p(T))*I_h(T)` by default. `collect-raw`, `collect` and the pipeline `run` accept `--no-endpoint-complement` for the observed integral alone. Their configuration, path tables and receipts retain the boolean, and changing it requires a new destination.

Collection resumes by path and binds its configuration and metric-source hashes. `--raw-root PROFILE=DATASET` supplies relocated roots. A changed bound source, parameter or root override requires a new collection directory. Per-path receipts hash raw metadata, observations, trajectory manifests, initial graphs and event databases. Cached metrics recheck these hashes when raw sources are available. The trajectory reader also verifies chunk checksums.

Prepared staged branches list intervention-window observations, while their raw records must retain the full replay prefix from step 0. An intervention-window or 16-step exported-grid integral is not the existing full-path `I_w`. References and early/middle/late branches retain their own endpoints and path identities.

## Import existing metric tables

Populate the supplied templates and import them:

```sh
python -m experiments.paper.metric_comparison collect \
  --preparation /path/to/preparation --output /path/to/collection \
  --points /path/to/legacy_points.csv.gz --paths /path/to/legacy_paths.csv
```

Point tables require the five observation-key fields plus `plan_fingerprint,binary_sha256,raw_record,I_p,I_s,I_h`. Path tables require four path-key fields, the same three provenance fields, and `I_w,sampling_method,prefix_start,trajectory_end,endpoint_complement`. `prefix_start` must be 0 and the endpoint must match the prepared path. The supported `sampling_method` is `existing_get_indices_adaptive_0.01_max_interval_512`; an integral on the exported 16-step grid cannot use that label. Historical metric-comparison path tables without the completion field are interpreted as uncompleted and require explicit `collect --no-endpoint-complement`; the default import rejects that mismatch. Duplicate or unprepared keys, inconsistent provenance and nonfinite metric values are rejected. Uncalculated states remain pending and are omitted from import tables. `--collection /path/to/collection` can import an existing collection.

```sh
python -m experiments.paper.metric_comparison compare \
  --preparation /path/to/preparation --collection /path/to/collection \
  --output /path/to/comparison
```

`compare` validates preparation/collection hashes and provenance, then performs an exact join. It exports `points_aligned.csv.gz`, `paths_aligned.csv.gz` and coverage, carrying the source collection's completion convention into alignment and the report. Missing values remain empty and pending. It does not classify paths, fit another scalar or enforce agreement between metric families.

## Interpretation

`legacy` denotes the original metric family. Collection runs the current implementation and records its source hashes. It does not automatically reproduce different baselines, counting objects or endpoint complements in older archives.

`I_h` weights positive-outdegree users equally; `C_edge` weights directed edges equally. `I_p,I_s` and `P` have different definitions. `I_w` summarizes a full trajectory, whereas P/E first-passage classification uses 16-step observations and its own thresholds. Comparisons should examine states, trajectory patterns and paired intervention effects under those definitions. Confidence-component endpoints and KDE opinion-peak counts retain their separate meanings.

## Parallel collection and the comparison report

`collect-raw --workers N` uses bounded process workers (1–8), grouped by original initial state. A bounded exact-state cache retains absolute step, opinion precision and ordering, directed-edge ordering and confidence radius. Each path still runs the existing adaptive calculation independently. Completed path receipts survive interruptions; `progress.json` reports collection counts and elapsed time. Raw graph inventory accepts both `graph-0` and zero-padded names, hashes every stored graph and WAL data, and excludes SQLite's volatile SHM lock state.

For reproducible CPU use, keep numerical-library thread counts at one while parallelizing initial states:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.paper.metric_comparison collect-raw \
  --preparation /path/to/preparation --output /path/to/collection --workers 6
```

After collection and exact alignment complete, generate tables, a Markdown report and PNG/PDF/SVG scientific figures:

```sh
python -m experiments.paper.metric_comparison.report \
  --preparation /path/to/preparation --comparison /path/to/comparison \
  --output /path/to/new-report
```

The report includes matched endpoints, one hold checkpoint per task/stage, existing first-passage labels with shared prefixes, and within-task intervention contrasts. Main figures use core conditions; sensitivity conditions remain separate in tables. Old/new effect estimates use the same matched pairs. `--allow-partial` is only for an explicitly labeled interim report; the default rejects pending input. Small legacy-index roundoff at shared checkpoints is retained in the paired table and accepted only within a float64 validation tolerance, without altering metric values.

Run the complete offline pipeline in one command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.paper.metric_comparison.run \
  --preparation /path/to/preparation --output /path/to/pipeline --workers 8
```

The pipeline writes `collection/`, `alignment/` and `report/`, with phase/status in `run_status.json` and path-level progress in `collection/progress.json`. Use `--collection /path/to/existing-collection` to reuse authenticated receipts from an interrupted collection. The same frozen collection configuration is required. Existing alignment/report destinations are never overwritten. Only the identified first-read source-change failure may retry under the same configuration; missing records or other failures stop the pipeline before analysis. A complete report also verifies the existing KDE opinion-peak counts against authenticated saved endpoint opinions and retains their comparison with confidence-component outcomes.

Formal collections use B=4081 and the clustered-reference JS-distance similarity. The numerical KDE study computes exact references internally using that same formula.

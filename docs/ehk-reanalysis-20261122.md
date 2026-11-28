# EHK primary analysis routes

## Measurement contract

Production four-index routes share `MeasurementEvaluator`: clustered-reference
JS-distance similarity for I_p/I_s, user-weighted current-edge I_h, and full-path
signed I_w with endpoint completion enabled by default. The numerical method is
4081-node linear-binned Gaussian KDE. Existing exact/B parameters are used only
by explicit numerical studies through the same metric formula core.

The former random-reference indices, post-based I_h and historical multi-formula
audit are removed. Supplementary P/E, exposure opportunities and terminal outcomes
remain distinct scientific measurements. Saved values are never relabeled under
a different formula or numerical protocol.

## Coordinated source revisions

The typed pipeline uses these coordinated code revisions. This guide follows the
EHK code revision and changes no measurement implementation.

| Repository | Code revision |
|---|---|
| `social-media-models` | `902fada29716a6c7235ed3b8863e96735a357bf7` |
| `extended-hk-model` | `7a9688a1914231a7eae8290bea03e13b798771a2` |

The EHK typed-interface cutover is
`844832b713da2c048bf0adcf4f3bd3d1654ea946`, paired with the SMP revision above.
When inspecting the earlier metric-only intermediate
`b0568cf8df0d34830374891abc610a33bde84043`, pair it with SMP platform revision
`e4b1ccf27ba92a10b3fcc70e5866557d16be1840`, which retains the earlier SDK interface.
Tests and generated verification records stay separate from the external raw
histories, observation databases and JSON archives.

## Original scans

From this repository in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m experiments.paper.analysis.run gradation `
  --raw-root F:/pathway-index-scan/data2/run `
  --metadata-source original-design --origin original-data2 `
  --database outputs/paper/reanalysis_20261127/original/stats.sqlite --concurrency 4
```

This selects all 12,288 planned tasks. `--limit N` or `--tasks NAME ...` selects a
subset; repeating the full command reuses matching complete documents. Other
modes are `epsilon` and `replicate`, each requiring its own actual raw root. The
inspected F:/pathway-index-scan/data1/2/3 roots contain gradation records only.
Different data roots remain different origins even when task names match.

The default `--metadata-source record` reads stored metadata.json. The explicit
`original-design` declaration is for the historical datasets that lack it. Output
records disclose this parameter provenance and missing RNG seeds. A missing file
never silently switches parameter sources. Date-stamped accumulated states and
chunked histories are real supported input formats, not alternate metric routes.

The default database location can be configured with SIMULATION_STAT_DIR.
Explicit `--database` is recommended for a new metric generation. All documents
bind the protocol, package/source identities and original inputs. Missing,
failed and stale tasks remain visible. Incomplete commands return nonzero.

For the main original-design descriptive summary (St0; Op0/2/6):

```powershell
.\.venv\Scripts\python.exe -m experiments.paper.pathway_index.grouped_summary `
  --database outputs/paper/reanalysis_20261127/original/stats.sqlite `
  --origin original-data2 --history-scope original-design --cutoff 0.6 `
  --output outputs/paper/reanalysis_20261127/groups
```

Use the returned analysis_id when a database contains multiple revisions. The
historical 0.6 split must be reassessed; it does not establish PbS/SbP first passage.
Use `--history-scope zero` in a separate output directory for the auxiliary
k_h=0 subset. Pooling data2/data3 requires both named `--origin` arguments;
the full current commands and output conventions are in the
[formal figure guide](../experiments/paper/pathway_index/README.md).

## Recommendation observations and analysis

The migrated local batch is already available at
`E:/ResearchData/ehk/recommendation/20261127/<profile>/analysis.sqlite`, with its
plan and binding. Its original JSON observations have verified ZIP archives in
the same directories. Source JSON cleanup status is recorded in the receipts;
complete raw histories remain on F:. Use this SQLite copy directly for analysis.

The canonical study directory contains plan.json, executable bindings and
analysis.sqlite. Existing JSON datasets are imported explicitly once:

```powershell
.\.venv\Scripts\python.exe -m experiments.paper.recommendation_shared.import_json `
  --source F:/ehk_cache/run/staged `
  --raw-root F:/ehk_cache/run/staged `
  --output outputs/paper/reanalysis_20261127/staged
```

Repeat for staged_timing and recommender_variants. Import authenticates task and
arm identities, recorded binaries, observation grids and update denominators.
Available raw summaries are cross-checked. Missing raw files remain missing in
the receipt; saved observations can still be analyzed. Normal study readers use
SQLite only. Existing scientific raw files and complete recovery checkpoints are
unchanged by storage migration.

New native runs stream observations to Python and a single SQLite writer; they
produce neither per-arm observations.json nor task-level JSON result files.
Online control and non-reconstructible opportunity counters remain recorded.
Changing an actual intervention trigger changes the experiment, whereas offline
state metrics and statistical summaries can be recomputed from saved inputs.

Export authenticated SQLite observations and analysis tables:

```powershell
.\.venv\Scripts\python.exe -m experiments.paper.recommendation_shared.bundle export `
  --dataset outputs/paper/reanalysis_20261127/staged `
  --output outputs/paper/reanalysis_20261127/staged-bundle --require-complete
```

Use each study's plot entry for its analysis/figures. The bundle records both the
observation database location and original raw root. Its SQLite snapshot uses
the backup API, including committed WAL content. CSVs are explicit exports.

## Recompute the four indices on paired-study histories

Prepare one or more authenticated bundles, then collect, align and report:

```powershell
.\.venv\Scripts\python.exe -m experiments.paper.metric_comparison prepare `
  --bundle outputs/paper/reanalysis_20261127/staged-bundle `
  --output outputs/paper/reanalysis_20261127/paired-preparation
.\.venv\Scripts\python.exe -m experiments.paper.metric_comparison.run `
  --preparation outputs/paper/reanalysis_20261127/paired-preparation `
  --output outputs/paper/reanalysis_20261127/paired-metrics --workers 4
```

Repeat --bundle to include timing and implementation-comparison studies. These
commands start no simulations. Exact joins match recorded P/E observations to
the adopted four indices; full-path I_w uses each branch's complete shared prefix
and its independent adaptive grid, not just the 16-step exported points. Numerical
receipts and tables live in SQLite, with explicit report exports.

## Process controls, numerical studies and rendering

Pathway controls use the same four-index evaluator and SQLite storage. Checkpoint
exports retain physical graph/opinion/post data independently of the metrics.
Mechanism plots use authenticated SQLite derived records rather than an
unversioned task-name pickle. KDE approximation studies explicitly evaluate exact
and candidate densities with the shared metric core; they do not change formal
analysis defaults or retain another formula implementation.

Original figure entry points remain main_scan.plot, pathway_index.plot,
pathway_index.plot_raw, pathway_index.plot_trajectories,
mechanism_cases.plot_cluster_and_repost, epsilon_scan.plot and replicate.plot
under experiments.paper. Set SIMULATION_STAT_DIR to the selected database's
containing directory and select EHK_STATS_ORIGIN/EHK_ANALYSIS_ID explicitly,
or EHK_STATS_ORIGINS for a declared compatible batch union. The main-scan,
pathway and cluster/repost entries also accept explicit --database, repeated
--origin, --analysis-id and --output arguments. Their main scope is the original
design; --history-scope zero keeps auxiliary figures in a separate subdirectory.
The registered original-figure filename is stats-v3.sqlite; an explicitly named
database used elsewhere does not change it. Grouped plots additionally require
EHK_IW_CUTOFF. Missing or obsolete protocols cause a clear error, never a fallback.

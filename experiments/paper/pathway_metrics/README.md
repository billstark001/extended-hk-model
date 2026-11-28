# Pathway metrics: continuous tendencies and P/E observations

This experiment compares the recorded continuous `I_w` pathway tendency with
independent P/E observations on the same saved trajectories. PbS/SbP are
tendency labels; `I_w = 0.6` is a working split. P/E first-observed threshold
order is a separate diagnostic, not classification ground truth.

| Module | Responsibility |
| --- | --- |
| `collection.py` | Authenticate saved histories, replay graphs, evaluate the existing native P/E core, and commit one result per task. |
| `study.py` | Freeze the design, authenticate relocation, inspect actual receipts, and explicitly resume all or a selected batch. |
| `report.py` | Read a consistent SQLite snapshot, commit aggregate tables to a separate SQLite database, then export CSV, vector PDF, PNG and a caption. |

No command reruns the original simulations or recalculates primary `I_w`.
Preparation, migration, status, plans and reporting never start measurement.

## Existing design and saved results

The fixed design contains 512 runs: `8 alpha × 8 q × 4 p × 2 algorithms`,
`k_h=0`, historical repeat ID 2. A parameter-cell checkerboard assigns data2 or
data3, with both algorithms in a cell using the same source. The source batches
contribute 256 runs each. Equal repeat IDs do not establish shared seeds or
paired initialization.

Collection is stopped at **120 complete, 392 missing, zero failed/invalid**,
with 81,416 P/E observations. The first 100 were a sequential parameter prefix;
the additional authorized batch contains ten `q<alpha` and ten `q=alpha` runs.
The original 100 receipt bodies were retained byte-for-byte. These partial
counts are not full-design pathway frequencies; use the report's coverage table.
The historical 32-run pilot remains separate.

Data remain outside Git:

- Primary statistics: `E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite`.
- Existing P/E observations: `E:/ResearchData/ehk/original/20261128/iw-pe-comparison/analysis.sqlite`.
- Original stopped study: `E:/ResearchData/ehk/original/20261128/iw-pe-comparison/study`.
- Current experiment directory: `E:/ResearchData/ehk/original/20261128/pathway_metrics`.

## Authenticate the directory relocation

The collector was moved byte-for-byte from `pathway_index/cutoff_validation.py`
to `pathway_metrics/collection.py`. Its SHA256 remains
`cb3d139edb25e08b8497ce7ff699f16f3d9f9def7a2d6de4e02dbbf473f3bd2c`.
The source-map path changes the implementation protocol identity. The explicit
`migrate` command proves this is the only protocol difference and records that
certificate. Completed tasks retain their original `f98de1...` execution
identities and exact result keys; unfinished tasks bind to the relocated
collector. They are completed and unfinished portions of the same scientific
design. Existing observation documents are neither copied nor rewritten.

The old Python entry points have been removed. There are no automatic fallback
readers or compatibility aliases. A local `.gitattributes` keeps experiment
Python sources in LF form so Git checkout does not change their byte hashes.

## Complete commands for the current experiment

Run in PowerShell. Repeating the migration authenticates an already prepared
destination; it does not measure any task.

```powershell
Set-Location D:/Research/extended-hk-model
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$PathwayPython = 'D:/Research/extended-hk-model/.venv/Scripts/python.exe'
$PathwayRoot = 'E:/ResearchData/ehk/original/20261128/pathway_metrics'
$PathwayStudy = "$PathwayRoot/study"
$PathwayOldStudy = 'E:/ResearchData/ehk/original/20261128/iw-pe-comparison/study'

& $PathwayPython -m experiments.paper.pathway_metrics.study migrate `
  --from-study $PathwayOldStudy --study $PathwayStudy

& $PathwayPython -m experiments.paper.pathway_metrics.study status `
  --study $PathwayStudy

& $PathwayPython -m experiments.paper.pathway_metrics.study collect `
  --study $PathwayStudy --plan-only --workers 4

# Analyze the current 120-run snapshot without starting measurement.
& $PathwayPython -m experiments.paper.pathway_metrics.report analyze `
  --study $PathwayStudy --output "$PathwayRoot/report-preliminary" --preliminary
```

The plan should show `complete=120`, `missing=392`, `would_measure=392`.
Only the following explicit command starts the remaining measurement:

```powershell
& $PathwayPython -m experiments.paper.pathway_metrics.study collect `
  --study $PathwayStudy --resume --workers 4

# Run after all 512 tasks have completed; incomplete coverage is rejected.
& $PathwayPython -m experiments.paper.pathway_metrics.report analyze `
  --study $PathwayStudy --output "$PathwayRoot/report-complete"
```

After any interruption, inspect status and repeat the same resume command.
Authenticated completions are skipped. `--limit N` bounds newly measured tasks.
A study session lock prevents concurrent workflow executions, and only the
parent process writes scientific receipts.

### Execute a named subset

A schema-1 selection JSON contains unique `tasks: [{"origin": "...", "name":
"..."}, ...]`. Every identity must belong to the frozen design. Planning does
not create execution files. Actual collection freezes the batch and hashes
under `study/runtime/batches/`; the 512-task manifest stays fixed.

```powershell
$PathwayBatch = "$PathwayOldStudy/supplemental20.json"
& $PathwayPython -m experiments.paper.pathway_metrics.study collect `
  --study $PathwayStudy --batch-selection $PathwayBatch --plan-only --workers 4

# This particular batch is already complete and should require zero new tasks.
# Use --resume instead of --plan-only only when executing an authorized batch.
```

## A new independent study

Use `prepare` for a new design. Use `migrate`, above, to retain the existing
120 measurements after this directory relocation.

```powershell
& $PathwayPython -m experiments.paper.pathway_metrics.study prepare `
  --study E:/ResearchData/ehk/original/NEW_STUDY `
  --database E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite `
  --analysis-id e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e `
  --selection E:/path/to/frozen-selection.json `
  --comparison-database E:/ResearchData/ehk/original/NEW_STUDY/observations.sqlite `
  --probe D:/Research/extended-hk-model/outputs/runtime/smp-probe-typed-20261127.exe
```

Study/selection JSON files are small frozen design inputs. Observations and
analysis tables use the existing SQLite `DocumentStore`; CSVs are explicit
exports, never normal-reader inputs or per-run caches. A committed receipt
includes the exact primary reference, source identity, complete P/E observation
grid, actual execution protocol and graph-replay verification. `status` counts
these records, not stale aggregate run-status documents. Changed scientific
inputs require a new authenticated study/protocol.

An existing derived analysis can be rendered without the source observations,
raw histories or study directory. For example, re-export the retained 120-run
analysis using the current figure style:

```powershell
& $PathwayPython -m experiments.paper.pathway_metrics.report export `
  --database E:/ResearchData/ehk/original/20261128/iw-pe-comparison/report-stopped120/analysis.sqlite `
  --report-id f485dab7376755ba73d820b41131e0020b1249c9c256a3e080b4be9573bdbf93 `
  --output "$PathwayRoot/figures-120"
```

The export keeps the original analysis receipt and independently records the
renderer source, shared plotting source and physical rendering parameters.

## Measurement and rendering

- `I_w` and its endpoint convention come from authenticated revision-6 primary
  statistics. P/E are measured at completed steps `0,16,...,T`, including `T`.
- Existing Python event-header graph replay matches primary-statistics input.
  Every saved graph must match the replay. Historical body actor discrepancies
  are recorded; original event databases are not modified.
- An event-database backup and performance index exist only in memory. At most
  16 detached states are sent to one native JSONL probe per run. Process workers
  avoid Python graph-replay/encoding contention.
- Reports commit aggregate tables and provenance to their own `analysis.sqlite`
  before export. `comparison.pdf` is vector output; `comparison.png` uses 300 dpi.
  `comparison_caption.txt` carries sample size, incomplete coverage and the
  min/quartile/median/max definition outside the figure.
- The two-panel figure is about 8 inches wide, using shared paper serif/CM
  fonts, 12-point axis labels and 10-point ticks/annotations. It is intended for
  approximately `0.9\textwidth` across the manuscript's two columns.

Run the experiment's focused tests:

```powershell
& $PathwayPython -m pytest tests/experiments/pathway_metrics -q
```

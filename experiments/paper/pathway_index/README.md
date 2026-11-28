# Formal main-scan figures

Run these commands from the model repository in PowerShell. They read existing
statistics only; they do not start simulations or recompute state indices.

The main scope is `original-design`: StructureM9 at `k_h=0`, and OpinionM9 at
`k_h=0,2,6`. This is the original asymmetric history design. Adding a historical
post-selection rule to StructureM9 would define another mechanism. The
auxiliary `zero` scope restricts both policies to `k_h=0`; it does not replace
the main scope. Opinion history controls and trajectory facets remain separate.

## Reproduce the current data2/data3 figures

```powershell
Set-Location D:/Research/extended-hk-model
$FigurePython = 'D:/Research/extended-hk-model/.venv/Scripts/python.exe'
$FigureDatabase = 'E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite'
$FigureAnalysis = 'e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e'
$FigureOutput = 'E:/ResearchData/ehk/original/20261128/formal-figures'
$env:MPLBACKEND = 'Agg'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$FigureSelection = @('--database', $FigureDatabase, '--analysis-id', $FigureAnalysis,
                     '--origin', 'original-data2', '--origin', 'original-data3')

# Main heatmaps and activity time do not depend on an I_w split.
& $FigurePython -m experiments.paper.main_scan.plot @FigureSelection `
  --history-scope original-design --output "$FigureOutput/main_scan"
if ($LASTEXITCODE -ne 0) { throw 'main_scan plotting failed' }

# Figure 5, algorithm/repost bars with history controls, and cluster/repost plot.
foreach ($FigureCutoff in @('0.6', '0.7')) {
  & $FigurePython -m experiments.paper.pathway_index.plot @FigureSelection `
    --history-scope original-design --cutoff $FigureCutoff `
    --output "$FigureOutput/iw-$FigureCutoff/pathway_index"
  if ($LASTEXITCODE -ne 0) { throw 'Figure 5 plotting failed' }
  & $FigurePython -m experiments.paper.pathway_index.plot_raw @FigureSelection `
    --history-scope original-design --cutoff $FigureCutoff `
    --output "$FigureOutput/iw-$FigureCutoff/pathway_index"
  if ($LASTEXITCODE -ne 0) { throw 'history-control plotting failed' }
  & $FigurePython -m experiments.paper.mechanism_cases.plot_cluster_and_repost @FigureSelection `
    --history-scope original-design --cutoff $FigureCutoff `
    --output "$FigureOutput/iw-$FigureCutoff/mechanism_cases"
  if ($LASTEXITCODE -ne 0) { throw 'cluster/repost plotting failed' }
}

# This figure already separates St0/Op0/Op2/Op6 and repost probabilities.
& $FigurePython -m experiments.paper.pathway_index.plot_trajectories @FigureSelection `
  --output "$FigureOutput/pathway_index"
if ($LASTEXITCODE -ne 0) { throw 'trajectory plotting failed' }

# Optional auxiliary subset: separate zero/figures, without overwriting main figures.
& $FigurePython -m experiments.paper.pathway_index.plot @FigureSelection `
  --history-scope zero --cutoff 0.6 --output "$FigureOutput/iw-0.6/pathway_index"
if ($LASTEXITCODE -ne 0) { throw 'zero-history plotting failed' }
```

Every formal `--output` above is a root directory. Outputs go to `figures/`, or
`zero/figures/` for the auxiliary scope. Original basenames such as
`f_grad_index_interpret` are retained. The raw bar figure's first four panels
follow `--history-scope`; its fifth panel always shows Opinion 0/2/6 separately.
The main-scan entry writes both heatmaps, activity-time plots and its terminal
summary with the same scope. No command copies figures into the paper repository.

The selected revision has 24,566 complete runs: data2 12,279 and data3 12,287;
Structure 6,144; Opinion 18,422 (history 0/2/6: 6,141/6,142/6,139).
The `zero` subset has 12,285 runs. Figure 5 reads finite I_w and records finite
observation counts for each metric. Each run has equal weight; histories and
repost levels are pooled by observed counts. Comparisons are descriptive
design-set means, not paired intervention effects. Low/high I_w is a working
split, not an exact PbS/SbP event-order boundary.

Source selection is explicit: repeat `--origin` or set
`EHK_STATS_ORIGINS=original-data2,original-data3`. A single source can still use
`EHK_STATS_ORIGIN`. Set only one environment selector; CLI selection takes
precedence. Unselected batches are not automatically merged. Missing requested
sources, mixed analysis IDs/protocols, obsolete metrics and incomplete/stale
records cause an error. Captions retain each source and its count. A default
database uses `SIMULATION_STAT_DIR/stats-v3.sqlite`; explicit `--database` is
recommended. `--cutoff` takes precedence over `EHK_IW_CUTOFF`.

## Computation, display and audit

`grouped_summary.summarize` is the single four-metric Figure 5 calculation.
Timing divides stored whole-trajectory event time by positive activity time;
AUC uses the existing clipped piecewise-linear integral of saved I_s on
normalized time [0,1]. Missing/nonpositive activity time leaves these metrics
missing. The formal and audit Figure 5 share `bar_plot.render_figure5` and its
mean +/- population-SD limits. Signed endpoints remain visible; a logarithmic
bar axis is used only when every endpoint is positive. SD is not a confidence
interval. Main heatmap color scales and trajectory facets retain the paper's
existing display conventions, with saturation and out-of-axis values disclosed
in their captions.

The cluster/repost moving curve retains its existing adaptive smoother. Its
mean and SD band now use the same support and evaluation grid; the former
repost band combined arrays evaluated at different x coordinates. This fixes
the displayed band without changing saved measurements or Figure 5 summaries.

The dedicated two-cutoff audit additionally exports shared axes across cutoffs,
history weights, group migration and direction changes. Its explicit `--output`
is an artifact directory, unlike the formal figure roots above:

```powershell
& $FigurePython -m experiments.paper.pathway_metrics.figure5 `
  --database $FigureDatabase --history-scope original-design `
  --output "$FigureOutput/figure5-audit"
if ($LASTEXITCODE -ne 0) { throw 'Figure 5 audit failed' }
```

That audit authenticates the fixed primary cohort itself. Its CSVs are exhibits,
not input caches. Keep SQLite, full per-run exports and generated verification
figures outside Git (or under ignored `outputs/`). Frozen paper artifacts retain
the provenance of the source version that generated them; updating the current
renderer does not rewrite earlier receipts.

## Verification on 2026-11-28

The current data2/data3 database was rendered through all five formal entries.
Figure 5 original-design .6/.7 and zero .6 matched all 32 summary rows and 32
contrasts in the corresponding frozen paper exhibits (rtol=2e-13,
atol=5e-12); source counts matched exactly. The two-cutoff audit was also rerun
separately. Main heatmaps, raw/history bars, trajectory facets and both
cluster/repost cutoffs rendered successfully. The source database SHA256 stayed
`eca4c9a7b82bd6201f70edc17220ba145fa42bba5cd9bb803c9833e11bc95733`.

The following 72 regression tests passed, including source compatibility,
history scopes, retained history controls, missing/signed metrics, complete
error ranges and the moving-band grid:

```powershell
& $FigurePython -m pytest tests/experiments/test_paper_document_figures.py tests/experiments/test_paper_grouped_summary.py tests/experiments/test_pathway_bar_plots.py tests/experiments/test_main_scan_plot_rendering.py tests/experiments/test_pathway_trajectory_bounds.py tests/experiments/test_iw_endpoint_exports.py tests/experiments/test_mechanism_scope.py tests/experiments/test_pathway_raw_scope.py tests/experiments/pathway_metrics -q
```

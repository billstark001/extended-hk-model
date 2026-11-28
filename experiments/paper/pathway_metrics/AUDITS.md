# Read-only main-scan and historical-pilot audits

The paper repository keeps compact exhibits and interpretation. The three
analysis entry points now live in this experiment package; their paper-side
copies have been removed. These commands read existing SQLite results and do
not start simulations or P/E measurement. Explicit output directories prevent
exports from being written beside experiment source files.

Run in PowerShell from the model repository:

```powershell
Set-Location D:/Research/extended-hk-model
$PathwayPython = 'D:/Research/extended-hk-model/.venv/Scripts/python.exe'
$PathwayPrimary = 'E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite'
$PathwayPaper = 'D:/Research/extended-hk-model-paper'

& $PathwayPython -m experiments.paper.pathway_metrics.threshold_audit `
  --database $PathwayPrimary --model-root D:/Research/extended-hk-model `
  --output "$PathwayPaper/artifacts/ehk_iw_threshold_20261128"

& $PathwayPython -m experiments.paper.pathway_metrics.recovered_pilot `
  --database $PathwayPrimary --model-root D:/Research/extended-hk-model `
  --pilot-database D:/Research/extended-hk-model/outputs/metric_revision_20261127/pilot/stats-v3.sqlite `
  --diagnostics "$PathwayPaper/artifacts/ehk_metric_revision_20261127/pilot/run_diagnostics.csv" `
  --output "$PathwayPaper/artifacts/ehk_iw_threshold_20261128" --figure

& $PathwayPython -m experiments.paper.pathway_metrics.figure5 `
  --database $PathwayPrimary `
  --history-scope zero `
  --output "$PathwayPaper/artifacts/ehk_figure5_threshold_20261128"
```

`threshold_audit` selects the exact formal primary analysis with 24,566 records
and uses its 12,285 zero-history records for matched comparisons. It retains the
recorded state arrays and verifies their scalar signed `I_w` integration using
the bound metric core. Density smoothing across runs is separate from state
metric KDE. Every analysis and metric source binding is checked. The historical
`src/ehk/domain.py` declaration binding is recorded separately: that file is not
imported by this saved-array audit, and no state is regenerated. A difference in
that declaration does not establish equality of the entire current checkout to
the primary frozen tree. Actual hashes and this scope are exported in provenance.

`recovered_pilot` authenticates the separate 32-run revision-5 historical pilot,
its saved P/E observations and source metadata, then compares with revision-6
primary `I_w`. It neither reuses old `I_w` as current values nor merges this
pilot with the fixed 512-run diagnostic. Its diagnostic CSV is a frozen input,
so it must be supplied explicitly.

`figure5` draws both `.6` and `.7` variants from the formal primary cohort.
The command above explicitly reproduces the auxiliary zero-history exhibits.
Its metric core is `pathway_index.grouped_summary.summarize`; its renderer is
shared with the formal `pathway_index.plot` entry. The audit adds authenticated
combined-batch selection, shared display ranges across cutoffs, history weights
and an explicit direction-change report.

The relocation was verified outside Git data directories: six main-scan audit
CSVs, four pilot numeric CSVs and six Figure 5 numeric CSVs were byte-identical
to their paper-side counterparts. Scientific formulas and cohorts were not
changed. Current source hashes and predecessor identities are recorded in the
paper artifact `ehk_iw_threshold_20261128/code_relocation.json`; earlier source
copies for the first two audits are retained at its documented external archive.

The complete 512-run workflow remains `study` / `report`, documented in
[README.md](README.md). It is a different analysis from these main-scan audits.

## Compare the archived statistics with the formal reanalysis

`legacy_stats_audit` reads the archived ORM SQLite table with pickle disabled,
matches its `data2` / `ssd_tmp` runs to the two formal primary batches using
parameters and invariant outcomes, and separates history pooling, batch
coverage, grouping, activity clocks, saved index curves and old NULL values.
It executes only the archived AUC helper from an identified Git object. It
does not execute the historical plotting driver or rerun raw simulations.

```powershell
Set-Location D:/Research/extended-hk-model
& .venv/Scripts/python.exe -m experiments.paper.pathway_metrics.legacy_stats_audit analyze `
  --legacy-database E:/ResearchData/ehk/original/archived/stats.db `
  --database E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite `
  --output E:/ResearchData/ehk/original/20261128/pathway_metrics/legacy-stats-audit/report `
  --export-directory D:/Research/extended-hk-model-paper/artifacts/ehk_legacy_stats_audit_20261128
```

The derived `analysis.sqlite` stores immutable report/table identities; its
CSV/JSON/PDF/PNG exports are compact exhibits. The `export` subcommand accepts
that database and a report ID without reopening either source database.
Baseline-only counterfactual bounds preserve the unknown pre-clipping values
at historical zeros, signed chronological integration and endpoint completion.
The sequential comparisons describe reaggregation, without identifying causal
contribution shares or inventing a frozen execution protocol for the archive.

## Figure 5: explicit history scope

Figure 5 has two separately named scopes. `original-design` is the original
main comparison: StructureM9 at `k_h=0`, and OpinionM9 pooled over `k_h=0,2,6`.
Opinion post history already has its defined selection rule. Adding history
to StructureM9 would require a new choice such as random selection or decay,
so this audit neither requires nor schedules a Structure-history experiment.
`original-design` is the plotting default. `zero` is an explicit auxiliary
restriction and is kept in a separate artifact directory; it does not replace
the main design-set comparison. Always name the scope when reproducing an
existing exhibit, whose frozen provenance is not rewritten by a code update.

```powershell
Set-Location D:/Research/extended-hk-model
& .venv/Scripts/python.exe -m experiments.paper.pathway_metrics.figure5 `
  --database E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite `
  --history-scope original-design `
  --output D:/Research/extended-hk-model-paper/artifacts/ehk_figure5_design_20261128
```

The entry reads only the formal `e499b1ab…` data2/data3 cohort: 24,566 runs in
the original design, or 12,285 under `--history-scope zero`. Both use the single
`pathway_index.grouped_summary.summarize` metric core. Run-weighted means and
population SD retain the original timing and `I_s` AUC definitions. The new
`history_groups.csv` gives each policy/history/pathway group's four metrics,
effective counts and weights; `group_composition.csv` also includes history.
Different numbers of retained runs supply the actual pooling weights.

The generated README/REPORT and captions state their selected scope, sources,
weights and full command. `.6/.7` figures use shared unclipped display ranges;
the existing zero-history exhibits are preserved. No simulation, new history
mechanism, manuscript or supplementary-material edit is part of this entry.

The formal Figure 5, raw/history bars, main heatmaps and cluster/repost plot now
use the same default design scope. See the complete [formal plotting commands](../pathway_index/README.md).

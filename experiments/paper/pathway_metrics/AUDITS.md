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

`figure5` draws both `.6` and `.7` variants from the formal matched primary
cohort. Its computation reuses `pathway_index.grouped_summary.summarize`; it
adds the authenticated combined-batch selection, shared display ranges and
explicit direction-change report missing from the historical plotting CLI.

The relocation was verified outside Git data directories: six main-scan audit
CSVs, four pilot numeric CSVs and six Figure 5 numeric CSVs were byte-identical
to their paper-side counterparts. Scientific formulas and cohorts were not
changed. Current source hashes and predecessor identities are recorded in the
paper artifact `ehk_iw_threshold_20261128/code_relocation.json`; earlier source
copies for the first two audits are retained at its documented external archive.

The complete 512-run workflow remains `study` / `report`, documented in
[README.md](README.md). It is a different analysis from these main-scan audits.

# Recomputing original-paper statistics

The adopted indices use one measurement core. `I_p` and `I_s` are
`1 - JS_distance(observed, clustered) / JS_distance(clustered, random)`.
PDFs are normalized on the finite integration grid. Values can be negative and
are at most one. Consensus is not required to have index zero.
`I_h` averages current directed-edge endpoint concordance over positive-outdegree
users and uses the baseline `epsilon-epsilon**2/4`, with a strict confidence
boundary. Production measurements use Gaussian linear binning with 4081 internal
nodes and the original 256-point output grid.

`I_w = integral I_h dI_p + (1-I_p(T))*I_h(T)` by default. The implementation uses
chronological signed trapezoids on its own full zero-to-end adaptive grid.
`--no-endpoint-complement` selects the existing integral-only convention.
There is no alternate formula switch or historical multi-convention audit.

Run from the model repository using its installed environment:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m experiments.paper.analysis.run gradation `
  --raw-root F:/pathway-index-scan/data2/run `
  --metadata-source original-design --origin original-data2 `
  --database outputs/paper/reanalysis_20261127/stats.sqlite --concurrency 4
```

The default `--metadata-source record` requires a real `metadata.json`.
Historical accumulated-state datasets without that file require the explicit
`original-design` declaration. Their parameters and collection configuration are
identified as supplied from the original design, not independently recorded in
raw files; RNG provenance remains unavailable. Population size is read and
validated from actual history. Existing metadata cannot be overridden through
this declaration. The raw files are never rewritten.

The modes are `gradation`, `epsilon`, and `replicate`. Omission of `--tasks` and
`--limit` selects the entire design. `--tasks NAME ...` selects named tasks;
`--limit N` selects a bounded prefix. `--plan-only` performs no writes.
`--raw-root` selects a direct directory; the positional workspace name selects a
configured entry in `sim_ws.json`. These are explicit data locations.

SQLite documents contain requested and recorded metadata separately, the metric
protocol, source identities and typed, pickle-free arrays. The parent process
commits results transactionally. Input identity, formula, numerical parameters
and source changes invalidate cached results. A failed recomputation cannot
supply a stale complete result. Repeated matching runs resume; `--force` recomputes.
Incomplete selections return a nonzero command status. Figure readers reject old
or missing metric protocols and mixed origins or endpoint conventions.

Plans, tasks, parameters, source identities and statistics are explicit dataclasses.
Decoders reject unknown record fields. JSON and SQLite adapters serialize these
records only at their I/O boundaries; historical missing protocol fields remain
missing evidence. Progress prints task starts, completed/cache counts and ETA.

`--include-pathway-probe` explicitly adds the independent supplementary P/E
first-passage labels and requires the selected native probe executable. Ordinary
four-index analysis requires no native executable. The descriptive `I_w` split
is specified independently; it is not a first-passage classifier.

See [the workflow guide](../../../docs/ehk-reanalysis-20261122.md) for paired-study
SQLite import, full metric collection and figure entry points.

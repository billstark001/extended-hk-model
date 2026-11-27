# Opinion and network measurements

All microscopic paper analysis uses `MeasurementEvaluator` with Gaussian linear binning at **B=4081**, retaining the 256-point output grid. The sole `I_p/I_s` formula is `1 - JS-distance(observed, clustered) / JS-distance(clustered, random)`, with each density normalized to unit trapezoidal mass on that finite grid. These reference-similarity indices are at most one and can be negative; consensus need not give `I_p=0`. They are not random-reference divergence ratios. Deterministic distance samples, absolute-step seeds, fitted bandwidths and reference distributions are retained. `I_h` averages current edge concordance over positive-outdegree users, uses the strict confidence boundary and the uniform reference `epsilon-epsilon**2/4`. Constant KDE samples use an analytic Gaussian.

`MeasurementState` extracts opinions, ordered directed edges, edge distances and out-degrees once. An evaluator maintains separate bounded scalar caches for opinion, edge and homophily dependencies. It retains neither a raw record nor a graph. A caller can reuse one evaluator across related records in one process:

```python
from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.state import MeasurementRecord

evaluator = MeasurementEvaluator()  # binned, B=4081
history = MeasurementRecord.from_saved_record(record)  # ingest the typed SMP record once
steps, h, p, s = evaluator.indices(history)
print(evaluator.cache_info().total_hits)

exact = MeasurementEvaluator(method="exact")
exact_steps, exact_h, exact_p, exact_s = exact.indices(history)
```

Measurement histories, cache statistics, density diagnostics and metric protocols are concrete dataclasses. `MeasurementRecord` holds the opinions, confidence radius and graph reader without carrying the stored metadata schema into the numerical evaluator. `DensityDiagnostics` and `CacheInfo` expose named attributes; serialization happens at file boundaries. `parse_metric_protocol` reads stored protocol evidence without filling fields absent from an older document.

The low-level `DistanceCalculator` also defaults to binned evaluation at B=4081 and returns only the adopted pair through `calculate_indices`. The existing explicit `method` and `bins` choices retain exact evaluation and alternate grids for the numerical-accuracy study. Both modes use the same clustered-reference formula. There is no KL metric branch or archived four-component result API. `PreparedOpinion` exposes its fitted density, full pair distances, clustered reference and reference separation for that study, without implementing another estimator.

`trajectory.sample_indices` supplies the existing refinement rule to both KDE modes: all three indices participate in error control and the complete 0..end history is used. Each path keeps its own sampling decisions.

`trajectory.pathway_index(p, h, endpoint_complement=True)` computes signed chronological trapezoids plus `(1-p[-1])*h[-1]`. The default completion extends the terminal point horizontally to `I_p=1`, retaining terminal homophily in the pathway-tendency measure. Pass `endpoint_complement=False` for the observed integral alone. Neither mode sorts or clips the trajectory or result. A single observation has zero observed integral and retains its completion when enabled; empty or mismatched trajectories are rejected.

`METRIC_PROTOCOL` revision 6 records the adopted formula, numerical evaluation and default completed convention. Use `metric_protocol(endpoint_complement=False)` to describe an uncompleted calculation. Analysis and collection identities include the selected convention and the metric, sampler and reader sources, so modes cannot share a cache identity. Existing result documents and portable bundles retain their recorded revision, values and provenance; missing historical flags are never silently relabeled as completed results. Completed historical collections are immutable and recomputation under changed sources or convention requires a new destination. Restoring the clustered-reference formula does not restore historical unseeded sampling or the old post-count homophily calculation.

P/E pathway observations use the existing native probe run reader in one batch, avoiding Python graph serialization. SMP processes saved-run static states incrementally; its response retains only measurement results. Historical accumulated-state files remain supported, although their underlying legacy opinion decoder still loads that old channel as a whole.

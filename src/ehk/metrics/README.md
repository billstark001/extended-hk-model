# Opinion and network measurements

Microscopic paper analysis uses `MeasurementEvaluator` with Gaussian linear binning at **B=4081**, retaining the original 256-point output grid. It uses the original distance samples, absolute-step seeds, fitted bandwidths and reference distributions. `I_h` still averages positive-outdegree users with the strict confidence boundary. Constant KDE samples keep their exact Gaussian evaluation.

`MeasurementState` extracts opinions, ordered directed edges, edge distances and out-degrees once. An evaluator maintains separate bounded scalar caches for opinion, edge and homophily dependencies. It retains neither a raw record nor a graph. A caller can reuse one evaluator across related records in one process:

```python
from ehk.metrics.evaluation import MeasurementEvaluator

evaluator = MeasurementEvaluator()  # binned, B=4081
steps, h, p, s = evaluator.indices(record)
print(evaluator.cache_info())

exact = MeasurementEvaluator(method="exact")
exact_steps, exact_h, exact_p, exact_s = exact.indices(record)
```

The low-level `DistanceCalculator` retains exact KDE as its default, including its original KL and archived worst-reference outputs. `calculate_indices` evaluates only the requested distance-index pair; its explicit `method` selects exact or binned evaluation. The experimental B comparison consumes these same kernels rather than maintaining another numerical implementation.

`trajectory.sample_indices` supplies the existing refinement rule to both KDE modes: all three indices participate in error control and the complete 0..end history is used. Each path keeps its own sampling decisions.

`trajectory.pathway_index(p, h, endpoint_complement=True)` computes signed chronological trapezoids plus `(1-p[-1])*h[-1]`. The default completion extends the terminal point horizontally to `I_p=1`, retaining terminal homophily in the pathway-tendency measure. Pass `endpoint_complement=False` for the observed integral alone. Neither mode sorts or clips the trajectory or result. A single observation has zero observed integral and retains its completion when enabled; empty or mismatched trajectories are rejected. `get_gradation_index_hp` exposes the same keyword.

`METRIC_PROTOCOL` revision 5 records the numerical evaluation and the default completed convention. Use `metric_protocol(endpoint_complement=False)` to describe an uncompleted calculation. Analysis and collection identities include the selected convention and the metric, sampler and reader sources, so modes cannot share a cache identity. Existing result documents and portable bundles retain their recorded revision, values and provenance; missing historical flags are never silently relabeled as completed results. Completed historical collections are immutable and recomputation under changed sources or convention requires a new destination.

P/E pathway observations use the existing native probe run reader in one batch, avoiding Python graph serialization. SMP processes saved-run static states incrementally; its response retains only measurement results. Historical accumulated-state files remain supported, although their underlying legacy opinion decoder still loads that old channel as a whole.

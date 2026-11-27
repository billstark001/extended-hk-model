# Distance-KDE approximation experiment

This independent experiment evaluates linear binning and Gaussian convolution for the observed distance densities used by `I_p` and `I_s`. It reads existing simulation histories and a completed canonical metric-comparison cohort. The formal cohort uses B=4081; this numerical study computes explicit `method="exact"` references internally with the same formula and independent adaptive sampling. It neither advances a model nor replaces the production metrics.

The experiment starts only after the baseline collection, alignment and report confirm complete coverage. Its configuration and receipts bind the baseline generation, selected tasks, raw sources and numerical implementation. Original experiment seeds, plans, simulation-binary identities and result files remain unchanged.

`I_w` includes `(1-I_p(T))*I_h(T)` by default in the exact baseline, fixed-grid approximation and independently adaptive approximation. `--no-endpoint-complement` selects the observed integral alone for all three. The selected convention must match the baseline and is recorded in configuration, path outputs and receipts. Missing endpoint-policy fields are rejected. Changing the convention requires a new destination and matching source cohort.

## Numerical scope

The original distance samples, their ordering and absolute-step RNG seed, KDE sample cap, fitted bandwidths, constant-sample fallback, reference densities, finite integration interval and JS calculation are retained. Only evaluation of the observed nonconstant KDE is approximated. The objective reference distance still uses all node pairs.

The output grid remains 256 points on `[-0.04, 2.04]`. Internal bin counts `1021, 2041, 4081, 8161` divide each output-grid interval into `4, 8, 16, 32` intervals, so output points require no interpolation. Convolution uses zero padding, with no periodic wrap, reflection or additional Gaussian-tail cutoff. Negative FFT roundoff is diagnosed explicitly; indices are not clipped to `[0, 1]`.

## Run

Run from the model repository, using a new destination:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.paper.kde_approximation \
  --preparation outputs/paper/metric_comparison_preparation_20261122_ready \
  --baseline outputs/paper/metric_comparison_20261125 \
  --output outputs/paper/kde_approximation_20261125/pilot \
  --workers 4 --max-tasks 32
```

The representative cohort selects original task groups deterministically and retains every prepared stage and arm within each selected group. Selection reasons and coverage are saved. This cohort is a numerical and performance study, not a replacement for full-cohort intervention estimates. Use `--full` and a separate output directory to evaluate the complete prepared cohort.

A small integration check can use `--max-tasks 1 --benchmark-states 2 --repeats 1 --workers 1 --no-figures`. The default benchmark uses 64 states and three repetitions. `--bins` explicitly selects the candidate internal grids.

## Comparisons

The experiment separates three effects:

1. Approximate versus explicitly recomputed exact `I_p/I_s` at the original prepared observation times.
2. Approximate `I_w` on each path's saved exact adaptive grid versus the internally computed exact index under the same endpoint convention.
3. Independently adaptive approximate `I_w`, accepted steps and activity time versus the fixed-grid result and baseline.

The 16-step observation export is never substituted for the exact adaptive integration grid. Reconstructing exact activity curves can require additional off-grid exact evaluations; their coverage and cost are recorded separately. `I_h` retains its existing computation throughout.

Kernel benchmarks pair exact and approximate methods on identical prepared states. Preparation and density-evaluation costs are distinguished, and single-candidate timing includes its preparation cost. Cache-validation time from the earlier baseline run is not used as an exact-computation benchmark.

The collection stores per-path receipts and named tables in `analysis.sqlite`; process workers return computations to one parent writer. CSV files are exports, and reports read SQLite tables. The collection writes coverage and progress; the report writes error tables, matched intervention comparisons and scientific figures. A report requires complete selected-cohort coverage. Results describe approximation error and measured performance. B=4081 is the formal implementation; the study preserves comparisons against exact evaluation and other resolutions as a numerical audit.

Repeated collection commands with the same frozen configuration reuse committed per-path receipts. Reports require a fresh destination. Selection, worker jobs, numerical diagnostics and receipts use declared dataclasses; only persistence encodes dictionaries.

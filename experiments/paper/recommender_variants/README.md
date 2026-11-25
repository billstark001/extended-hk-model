# Full-trajectory comparisons of EHK recommendation implementations

This package owns the `recommender_variants` design, scheduling, analysis and figures. It compares recommendation implementations over complete trajectories from matched initial states. [`recommendation`](../recommendation/README.md) owns staged switching and online timing; [`recommendation_shared`](../recommendation_shared/README.md) supplies shared runtime and analysis services.

## Design and data compatibility

The core design uses nine `(alpha,q)` cells: `(0.01,0.30)`, `(0.03,0.10)`, `(0.05,0.05)`, `(0.10,0.03)`, `(0.30,0.01)`, `(0.01,0.10)`, `(0.03,0.30)`, `(0.10,0.01)` and `(0.30,0.03)`. Each combines `p=0,0.1,0.25` with 16 independent initial states, giving 432 groups. Every group uses 500 nodes, expected initial out-degree 15, `epsilon=0.45`, ten recommendation slots and `k_h=0`. Every arm runs for 2048 steps.

Four one-factor sensitivity conditions at `(alpha,q,p)=(0.05,0.05,0.1)` vary either `epsilon=0.35/0.55` with `k_h=0`, or `k_h=2/6` with `epsilon=0.45`. Each has 16 repeats sharing the corresponding core initialization seeds, for 496 groups overall. Confidence radius and history length are not crossed. Core and sensitivity results are reported separately.

The original six arms are Opinion, OpinionM9, OpinionRandom, Structure, StructureM9 and StructureRandom. Unsuffixed policies rank by their respective scores. M9 requests one uniform exploration slot and nine ranked slots. The `Random` suffix denotes score-weighted sampling with no additional uniform mixture. Arms share the initial graph, opinions, posts and RNG state. Their initialization seeds are independent of the staged and timing studies.

The OpinionRandom factorial design `zeta in {1,4} x sigma in {0,2}` controls `Steepness` and `OpRandomNoiseStd`. The original `OpinionRandom` is `zeta=1,sigma=2`; the added arms are `OpinionRandomZeta1Noise0`, `OpinionRandomZeta4Noise0` and `OpinionRandomZeta4Noise2`. These add `496*3=1488` trajectories, producing nine arms per group. Recommendation `Tolerance=0.4` remains separate from the confidence radius `epsilon`.

OpinionRandom computes `b=max(1-abs(x_i-x_j)/0.4,0)`, then samples using weights `max(b+(1-2*b)*eta,0)^zeta`, where `eta ~ N(0,sigma^2)`. `sigma` is recommendation-score noise; increasing `zeta` also concentrates high weights produced by noise. Toggling noise changes subsequent RNG consumption. Pairing requires the same initialization and RNG state, without assuming identical process draws throughout diverging trajectories.

Existing data directories, layouts, `study="paper_recommendation"`, `profile="recommender_variants"`, plan schemas, task IDs, coordinates and seeds are retained. The seed namespace keeps its historical value 2, independent of registry order. Historical `campaign.order` metadata preserves frozen fingerprints; this package schedules the study independently. Existing six-arm plans retain their frozen protocol and per-arm executable provenance.

## Entry points and recovery

Preview the design from the EHK repository:

```sh
python -m experiments.paper.run recommender_variants --plan-only
python -m experiments.paper.recommender_variants.study --plan-only
```

Run through a configured `sim_ws.json` workspace, or pass the study directory directly:

```sh
python -m experiments.paper.run recommender_variants --workspace WORKSPACE --concurrency 6
python -m experiments.paper.recommender_variants.study \
  /path/to/recommender_variants --smp-binary /path/to/smp --concurrency 6
```

The workspace entry point uses `SMP_BINARY_PATH` or `--smp-binary`; the direct study entry requires `--smp-binary`. Results use the `recommender_variants` directory. Repeat the same command after interruption to reuse arms whose completed raw records pass validation. Resume requires the frozen plan and bound executable SHA. Run one scheduler per dataset.

Persistent raw records under `raw/<task>/full/<arm>` contain float64 opinion and opinion-sum histories, per-user counts, rewiring/repost events, snapshots, metadata and `smp.log`. `AgentNumber`, `OpinionSum`, `RewiringEvent` and `PostEvent` are enabled; `ViewPostsEvent` and `ViewPostsDetails` are disabled. Exposure counters are saved for each observation interval.

Stop the scheduler before adding factorial arms or changing its bound executable. Explicit migration accepts a workspace root or one study directory; executable replacement requires the single study directory:

```sh
python -m experiments.paper.recommendation_shared.migration \
  /path/to/recommender_variants --smp-binary /path/to/replacement-smp
```

Migration preserves the old plan, bindings and receipt. Original arms retain their provenance; added arms record the executable used for them. Task IDs, seeds, existing results and raw records remain unchanged. This supports datasets spanning the OpinionRandom zero-score-row correction, which clears stale recommendation weights when all current scores are zero. Migration and bundle export use the shared package's entry points.

## Analysis and figures

```sh
python -m experiments.paper.recommender_variants.plot /path/to/recommender_variants
python -m experiments.paper.recommender_variants.plot /path/to/recommender_variants \
  --output /path/to/analysis --tables-only
```

Analysis defaults to the dataset's `analysis/` directory. It exports run-level metrics, trajectories, paired differences, coverage and provenance, including `arm_coverage.csv`, `opinion_random_factorial.csv` and complete-cell factorial tables. Factorial tables report the four paired simple effects of `zeta` and noise, plus their interaction, for matched initial states.

Each contrast uses pairs with all required arms complete. Whole-group completion requires every arm in the frozen plan; six-arm bundles retain their six-arm protocol. Core and sensitivity outputs have separate directories such as `core_e45_kh0/` and `epsilon_e35_kh0/`. `complete_cells/` includes only cells with all planned repeats.

Analysis reads saved `P`, `E`, 16-step first-passage labels, confidence-component endpoints and KDE opinion-peak counts under their respective definitions. The `I_h,I_p,I_s,I_w` family is maintained separately. `coverage.json` records the plan, record SHAs, executable bindings and analysis time. Coverage and run tables retain per-arm provenance.

## Portable bundles and factorial figures

`recommendation_shared.bundle export` validates and copies summaries, runs this study's analysis and compresses tables under `data/tables/`. `--require-complete` requires every arm in the frozen protocol; omit it for a partial export. Export describes the copied record generation without starting simulations.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
factorial_bundle=outputs/paper/recommender-variants-bundle
python -m experiments.paper.recommendation_shared.bundle export \
  --dataset /path/to/recommender_variants \
  --output "$factorial_bundle" --require-complete
python -m experiments.paper.recommender_variants.variants_review prepare \
  --bundle "$factorial_bundle" --output "$factorial_bundle/plots"
python -m experiments.paper.recommender_variants.variants_review compose \
  --directory "$factorial_bundle/plots"
```

`prepare` selects initial states with all four OpinionRandom configurations complete. Original-policy comparisons use the same cohort. Core and sensitivity conditions remain separate. It writes `matched_*`, `paired_vs_m9*`, `plot_data.json` and editable `layout.json`, retaining an existing layout. `compose` reads the saved plot data and layout to generate PDF, SVG, PNG and provenance manifests; it needs no mounted raw volume.

Rebuild cell-level figures from exported summaries:

```sh
python -m experiments.paper.recommender_variants.plot \
  "$factorial_bundle/data/study" --output "$factorial_bundle/detail"
```

This package owns design, execution, reading, analysis and factorial figures for implementation comparisons. The staged package rejects implementation-comparison data. [`recommendation_shared`](../recommendation_shared/README.md) owns bundle export and frozen-plan migration.

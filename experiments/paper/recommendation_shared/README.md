# Shared recommendation runtime and analysis

This package supplies services used by staged interventions and full-trajectory implementation comparisons. Scientific design and scheduling belong to [`recommendation`](../recommendation/README.md) and [`recommender_variants`](../recommender_variants/README.md).

- `runtime`: persistent observations, checkpoint and RNG pairing, per-arm record reuse, plan/binding validation, scheduling and storage checks.
- `analysis`: frozen-plan and record readers, metric tables, paired statistics, exposure counters and coverage helpers.
- `plot`: shared styles, coverage displays and effect plots.
- `bundle`: verified summary copying, dispatch to the owning study's analysis, provenance and compressed tables.
- `migration`: explicit frozen-plan migration, preserving historical plans and per-arm binary bindings.

Policies, parameter grids, intervention rules and comparison logic belong to their study package. Shared services accept study protocols and callbacks. Bundle export chooses the analyzer from the frozen plan's `profile`.

Existing data layouts, plan schemas, study/profile fields, task IDs, seeds and per-arm executable provenance are retained. Each study owns its design, run and analysis entry points. Historical campaign metadata preserves frozen fingerprints; the current study entry point controls execution scope.

## Maintenance commands

```sh
python -m experiments.paper.recommendation_shared.bundle export --dataset /path/to/dataset --output /path/to/bundle --require-complete
python -m experiments.paper.recommendation_shared.migration /path/to/workspace
```

`bundle export` validates and copies saved summaries, dispatching analysis by frozen `profile`. `--require-complete` requires every task and arm in that protocol; omit it for a partial export. Export does not start simulations.

`migration` accepts a workspace root or one study directory. It validates existing tasks, records and bindings before saving the old plan and migration receipt. Binary replacement requires the single `recommender_variants` study directory and `--smp-binary /path/to/smp`. Original and added arms retain their own executable provenance. Stop the affected scheduler before migration.

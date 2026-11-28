# Shared recommendation runtime and analysis

This package supplies services used by staged interventions and full-trajectory implementation comparisons. Scientific design and scheduling belong to [`recommendation`](../recommendation/README.md) and [`recommender_variants`](../recommender_variants/README.md).

- `runtime`: persistent observations, checkpoint and RNG pairing, per-arm reuse, plan/binding validation, scheduling and storage checks.
- `storage`: the canonical `analysis.sqlite` DocumentStore, atomic arm/task commits and analysis tables.
- `analysis`: frozen-plan and record readers, metric tables, paired statistics, exposure counters and coverage helpers.
- `plot`: shared styles, coverage displays and effect plots.
- `bundle`: verified SQLite snapshots, dispatch to the owning study's analysis, provenance and portable tables.
- `import_json`: authenticated one-time import of historical observations into SQLite.
- `migrate_jobs`: explicit frozen native job protocol conversion, retaining scientific state.

Policies, parameter grids, intervention rules and comparison logic belong to their study package. Shared services accept study protocols and callbacks. Bundle export chooses the analyzer from the frozen plan's `profile`.

Scientific raw histories, full recovery checkpoints, frozen designs, task IDs, seeds and generating executable identities are retained. Observation and analysis data use `analysis.sqlite`. One parent-process writer owns the SQLite connection; worker threads submit acknowledged transactions. Native jobs return one structured result on stdout and do not write `observations.json` or per-task JSON files. A finished native checkpoint can reproduce an uncommitted result without advancing the model. File-based cancellation also permits graceful checkpointing on Windows.

Staged, timing and implementation studies commit completed arms immediately. Pending task indexes support execution recovery and stay outside normal scientific readers. Final completion atomically publishes the formal task index and retires its pending index, so incomplete stage sets cannot release the timing gate or count as complete analysis tasks.

Normal readers only accept the database. Import historical JSON explicitly into a separate destination:

```sh
python -m experiments.paper.recommendation_shared.import_json --source /path/to/old-dataset-or-bundle --output /path/to/new-observations --raw-root /path/to/original-raw-dataset
```

Import authenticates frozen identities, bindings and bundle hashes, checks observation grids and denominators, and records original file hashes. Missing raw files remain explicitly missing. The destination records its raw-root mapping, so a local database may refer to a read-only external volume. Import does not alter the source or attach new RNG or binary identities to it.

## Maintenance commands

```sh
python -m experiments.paper.recommendation_shared.bundle export --dataset /path/to/dataset --output /path/to/bundle --require-complete
```

`bundle export` validates observations and exports a consistent SQLite backup, then dispatches analysis by frozen `profile`. Analysis tables are SQLite documents; CSV is an optional export. `--require-complete` requires every task and arm in that protocol; omit it for a partial export. Export does not start simulations. Its manifest keeps the observation database and original raw dataset paths separate.

Frozen plans and executable bindings are immutable. Changed parameters, additional arms or a different executable require a new dataset directory. Historical frozen plans remain readable under their original protocol.

New EHK plans record `ranked_tie_rule="uniform-per-query-without-replacement-v1"` as a fixed scientific contract. It is not a selectable simulator parameter. Build the corrected SMP before creating such a dataset; the existing executable SHA binding authenticates the producer. Keep the original task coordinates, seeds and initial states, but use a new output directory and binding. Do not continue from old Structure caches: they lack the noisy scores required by the corrected rule. Historical raw trajectories remain readable for analysis.

An older plan without this field has an unrecorded rule, not an implied uniform rule; reading and exporting it preserves its original fingerprint. Timing runs require staged evidence with the current recorded rule, so a gate from the old experiments cannot release a corrected timing campaign. Recompute that gate after rerunning staged.

The validated SMP source is [`77c0352`](https://github.com/billstark001/social-media-models/commit/77c0352a872c89c63dc7c516cdfbd4014c17b511). Build `cmd/smp` and `cmd/smp-probe` from the corrected checkout before binding a new study; a stale executable is not updated by changing the Python plan.

Frozen native jobs using the retired file-output protocol must be migrated explicitly with `python -m experiments.paper.recommendation_shared.migrate_jobs /path/to/dataset`. This archives their original bytes and changes output control metadata only. It does not change executable bindings or scientific checkpoints. A separate executable change must satisfy the study's provenance requirements; there is no automatic old-protocol fallback.

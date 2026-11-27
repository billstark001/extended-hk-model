# Staged recommendation interventions in EHK

This package owns the `staged` and `staged_timing` designs, scheduling and analysis. [`recommender_variants`](../recommender_variants/README.md) owns full-trajectory implementation comparisons. [`recommendation_shared`](../recommendation_shared/README.md) supplies persistent execution, paired statistics and bundle export.

Existing data directories, plan schemas, task IDs, seeds and per-arm executable provenance are retained. Historical `campaign.order` metadata preserves frozen fingerprints; the current entry point controls scheduling order.

## Design

The core design uses nine `(alpha,q)` cells: `(0.01,0.30)`, `(0.03,0.10)`, `(0.05,0.05)`, `(0.10,0.03)`, `(0.30,0.01)`, `(0.01,0.10)`, `(0.03,0.30)`, `(0.10,0.01)` and `(0.30,0.03)`. Each combines `p=0,0.1,0.25` with 16 independent initial states, giving 432 groups. The historical main scan used 12 repeats per setting. This design uses 500 nodes, expected initial out-degree 15, `epsilon=0.45`, ten recommendation slots and `k_h=0`. Arms within a group share the initial graph, opinions, posts and RNG state.

`staged` adds four one-factor sensitivity conditions at `(alpha,q,p)=(0.05,0.05,0.1)`: `epsilon=0.35/0.55` with `k_h=0`, and `k_h=2/6` with `epsilon=0.45`. Each has 16 repeats using the corresponding core initialization seeds, for 496 groups overall. Radius and history length are not crossed. `staged_timing` contains only the 432 core groups and uses initialization seeds independent of the staged study. Condition IDs, tables and figures distinguish confidence radius, history length and sensitivity family.

Tasks are ordered by replicate, then parameter cell, covering all cells at repeat 0 before repeat 1. Added cells follow the original seed coordinates; existing task fields and arm configurations are retained. Resume first registers all validated cached records, then schedules unfinished work in that order. Parallel completion order may differ because cell runtimes vary.

- `staged` generates an OpinionM9 reference, observes it every 16 steps and saves checkpoints at the design's candidate steps. The early checkpoint is the first candidate below both thresholds. The middle checkpoint is the first later candidate at or above either threshold that has not met the late criterion. From step 128 onward, the late checkpoint is the first candidate whose two preceding observation intervals each have mean absolute opinion displacement per agent-update at most 0.001 and completed rewiring rate at most 0.01. At each available checkpoint, paired arms retain OpinionM9 or switch to StructureM9 for 512 steps, requesting the same recommendation and exploration slots. Missing stages remain visible in coverage.
- Two late-stage diagnostics supplement the policy comparison. A bridge arm supplies one current original post from the opposite camp within the recipient's confidence range when available, filling an empty slot or replacing the last ranked slot. A repost-filter arm for `p>0` removes recommended placements produced by reposting, including an account reposting its own old post, while retaining repost-production probability `p`. Ordinary policy contrasts use each policy's own candidate generation and ranking.
- `staged_timing` uses independent initial states. Once all 432 core staged groups are complete, the gate compares late minus early switching effects using only repeats with both checkpoints. The timing study runs when the absolute mean difference in 512-step separation effects is at least 0.05 and its 95% t interval excludes zero. Sensitivity conditions are excluded. Both triggers switch to StructureM9 for exactly 512 steps, then restore OpinionM9. The fixed trigger is step 128. The state trigger is the first excess-concordance crossing of 0.10, falling back to step 1536. Two constant policies provide controls. The primary state-versus-fixed objective is lower separation at step 2048.

## Measures

- Opinion separation maximizes twice the smaller population fraction in two opinion intervals of width at most `epsilon/2`, separated by a gap strictly greater than `epsilon`. Its first-passage threshold is 0.40. Excess edge concordance subtracts the current distinct-node-pair concordance from directed-edge concordance, retaining its sign; its threshold is 0.10. Classification uses each measure's first recorded crossing, retaining simultaneous crossings and cases with no crossing by the window endpoint.
- Cross-camp exposure counts actual displayed recommendation slots. Recipient and post opinions must lie on opposite sides of zero, each more than `epsilon/4` from zero. Confidence eligibility uses the post–recipient opinion difference. Rewiring eligibility additionally requires a discordant followee post and an author who can be newly followed. Late figures separate ranked slots; the bridge arm includes inserted posts in that category. Records retain insertion counts, fill rates, completed rewirings and total absolute opinion displacement.
- The four-index family uses equal-user mean endpoint concordance for `I_h` among positive-outdegree users, with uniform baseline `epsilon-epsilon**2/4`. Concordance of received followee posts is separate. `I_p,I_s` use `1 - JS_distance(observed, clustered) / JS_distance(clustered, random)`, with finite-grid density normalization and default Gaussian linear binning at B=4081. They measure similarity to their clustered references, can be negative and are at most one; consensus need not give `I_p=0`. `I_w` uses chronological signed trapezoids plus `(1-I_p(T))*I_h(T)` by default; the shared analysis and metric-comparison collectors accept `--no-endpoint-complement` for the observed integral alone and record the choice. Existing frozen outputs retain their original convention. The historical 0.6 split is descriptive and requires reassessment separately from first-passage labels; see the [reanalysis guide](../analysis/README.md).
- Endpoints report confidence-component classification and KDE opinion-peak counts separately. Confidence-component outcomes that do not meet the absorbing-state criterion are unresolved. Stage contrasts use paired differences. Trajectory shading is one standard error; endpoint and cross-stage differences use 95% t intervals.

Formal runs use persistent `smp`. Each arm's `raw_record` points to `raw/<task>/<phase>/<arm>` under the dataset. Records contain float64 opinions and opinion sums, per-user counters, rewiring/repost events, model snapshots and metadata. `AgentNumber`, `OpinionSum`, `RewiringEvent` and `PostEvent` are enabled; `ViewPostsEvent` and `ViewPostsDetails` are disabled to avoid duplicating every feed in SQLite. Branches replay the common reference prefix from the same initialization, then apply the intervention. Physical checkpoint state and RNG are verified before switching. Raw histories remain readable from step 0; the shared prefix is excluded from intervention-window exposure denominators.

Every observation window retains requested/filled slot counts and completed rewirings. Exploration, ordinary policy and bridge slots separately count display, cross-camp, confidence and rewiring eligibility. Late arms also save compressed per-user, per-slot feeds for subsequent windows, including pre-update opinion, requested count, ordered neighbor/recommendation slots, original post, sharing account, display step and source. Neighbor slots include followees with no post. Repost, confidence and rewiring eligibility can be reconstructed from these facts and model parameters. Paths are relative to the dataset. Read feeds sequentially with `ehk.micro.traces.iter_exposure_trace`. New runs require the current design version and complete exposure fields.

Offline state measurement uses the `smp-probe` configured by `SMP_PROBE_BINARY`, respecting recorded opinion precision. Saved per-user counters or complete ViewPosts events support followee-post concordance. Missing observation windows may yield `unavailable` first-passage labels.

## Entry points

Preview the design from the EHK repository:

```sh
python -m experiments.paper.run staged --plan-only
python -m experiments.paper.run staged_timing --plan-only
```

Build the persistent simulator and offline probe with `make build build-probe` in the SMP repository. Configure the simulator through `SMP_BINARY_PATH` or `--smp-binary`, and the probe separately through `SMP_PROBE_BINARY`. Choose a workspace from `sim_ws.json`:

```sh
python -m experiments.paper.run recommendation --workspace WORKSPACE --concurrency 6
python -m experiments.paper.recommendation.plot /path/to/staged
```

The `recommendation` campaign runs `staged`, then `staged_timing` if its gate passes, through one Python scheduler with at most six native workers. Each study uses its own workspace subdirectory; campaign status is in `recommendation_runtime/status.json`. Implementation comparisons run independently through `recommender_variants`. Individual `staged` and `staged_timing` entry points remain available. Avoid simultaneous schedulers that collectively exceed the intended worker limit.

Before scheduling, an 8-node, 2-step persistence check verifies that the target volume supports writing and reopening native event databases, trajectories and snapshots. Failure stops scheduling. SMP event databases use SQLite WAL; retain `-wal` and `-shm` alongside a running database, and archive after normal closure.

Frozen plans and executable bindings are immutable. Changed parameters, additional arms or a different executable require a new dataset directory. Historical frozen plans remain readable under their original protocol.

After `kill -TERM PID`, repeat the same command to resume. Persistent arm snapshots are retained; completed tasks with valid raw records are skipped. Resume requires the frozen design and bound executable SHA. Changed parameters or executables require a new dataset directory. Run one scheduler per dataset.

Analysis defaults to the dataset's `analysis/` directory and exports run metrics, trajectories, paired differences and per-condition completion/stage coverage. Core and sensitivity figures have separate directories such as `core_e45_kh0/` and `epsilon_e35_kh0/`. Different `epsilon` or `k_h` values are not pooled. Original-paper reanalysis uses a separate `stats-v3.sqlite` document database, with optional read-only archived-score comparisons; see the [reanalysis workflow](../../../docs/ehk-reanalysis-20261122.md).

During execution, `--output` can select an independent analysis directory. Analysis reads the completed-record snapshot at entry and records the plan, record SHAs, executable bindings and time in `coverage.json`. Coverage tables/plots distinguish unfinished from unplanned cells. Cell titles show completed and planned repeat counts. `complete_cells/` includes only fully replicated cells. Completed cells in a sequential run need not represent the entire design; partial cells remain in run tables and their own figures.

`--tables-only` exports tables, coverage and provenance without rendering every cell. `review_plots` supplies individual bridge-pair differences, reference PbS count grids and single-initial-state cross-stage trajectories. The first two require run and pathway-count tables using the same observation convention. Restricting to repeat IDs complete across cells can balance sample counts, but identical repeat IDs in different parameter cells do not share initialization and are not cross-cell pairs.

Branch first-passage labels include the shared reference prefix before switching. If a threshold is already crossed at the intervention window's start, `window_pathway` is `left_censored`; it is not a simultaneous full-path crossing. Reference plots extend to step 2048 and show activity-based checkpoints. Late exposure figures use policy plus bridge slots, with separate total-exposure figures including exploration. Opinion heatmaps remain separate by repeat, as do KDE peak counts and confidence-component endpoints. The staged gate remains pending until every core group is complete.

Persistent SMP runs use `ViewPostsDetails=true` to collect per-slot detail. `ViewPostsEvent=true` alone records the original confidence groups. Explicitly migrate old ViewTweets tables using `smp-migrate events`.

## Portable manuscript figures

Shared bundle export validates and copies saved summaries and computes compressed CSV tables locally. The frozen `profile` selects the staged or implementation analyzer. Export does not start simulations.

```sh
python -m experiments.paper.recommendation_shared.bundle export \
  --dataset /path/to/staged --output /path/to/staged-bundle --require-complete
python -m experiments.paper.recommendation.manuscript prepare \
  --bundle /path/to/staged-bundle --output /path/to/plots
python -m experiments.paper.recommendation.manuscript compose --directory /path/to/plots
```

`manuscript prepare` creates self-contained `plot_data.json` and editable `layout.json` from a local bundle. `compose` reads only these files and works without the raw volume. `panels` can additionally export individual SVG/PDF/PNG panels; whole-figure composition does not read panel images. Keep full records in model `outputs/`; the manuscript needs compact plot data, layouts, vector figures and provenance.

The staged `manuscript` entry supports `prepare`, `panels` and `compose`. [`recommendation_shared`](../recommendation_shared/README.md) owns export and authenticated JSON import commands; [`recommender_variants`](../recommender_variants/README.md) owns factorial figures.

# Reproducible original-paper runs and process controls

The original `epsilon`, `gradation`, `mech`, and `replicate` task names, axis orders and historical `generate_tasks()` metadata remain unchanged. Existing raw inputs are read using their actual saved metadata; seeds are never attached retrospectively to an old unseeded run.

## Original-paper recalculation

New execution uses a separate schema-2 plan with explicit PCG64-DXSM root seeds, replicate/axis coordinates, graph size and expected degree, recommender parameters, collection channels and float64 opinion/opinion-sum precision. Seeds are a stable SHA-256 derivation of a declared namespace, study and coordinate, independent of concurrency and execution order. Each original task gets an independent seed. Change `--seed-namespace` to obtain a separate deterministic ensemble.

Use a **new output directory**. Historical UniqueNames are intentionally retained for the existing analysis interface. The seeded runner rejects a same-name record with different parameters or RNG, including an unseeded legacy record.

```sh
PYTHONPATH=src:. python -m experiments.paper.run gradation --plan-only
PYTHONPATH=src:. python -m experiments.paper.run gradation \
  --run-dir /path/to/new-paper-run/gradation --freeze-only
PYTHONPATH=src:. python -m experiments.paper.run gradation \
  --run-dir /path/to/new-paper-run/gradation --concurrency 4
```

The same commands accept `epsilon`, `replicate` or `mech`. `--workspace` continues to read an existing `sim_ws.json` entry; it is mutually exclusive with `--run-dir`. The frozen plan and executable/Python working-tree provenance are stored under `RUN_DIR/.paper-runtime/STUDY/`. Actual resolved metadata remains in each ordinary SMP scenario directory. The provenance includes binary SHA-256 and Go build information, Python Git revision/diff, and source hashes (including untracked Python modules). A changed frozen plan, executable or source tree requires a new run directory; finish code changes before freezing a production run.

## Paired process controls

This independent design contains four existing mechanism cases (`baseline_pbs`, `baseline_sbp`, `baseline`, `op_recsys`), 12 independent repeats, and three arms per initial state: the full model, `q=0`, and `alpha=0`. No observation is labeled PbS/SbP merely from its case name. All arms share the initial graph, opinions, retained posts and complete RNG state, checked from step-zero checkpoints. They run to a common fixed horizon of 2048 steps, even when a control has stopped changing. Defaults are 500 nodes and expected out-degree 15.

The design saves ordinary SMP raw histories, rewiring/repost events, float64 opinions and opinion sums, plus full recovery checkpoints at fixed steps `0,16,64,256,1024,2048`. Recommendations and repost generation remain active in both controls. `q=0` removes tie replacement; `alpha=0` removes opinion movement. The arms share their root streams, rather than assuming identical process draws after their dynamics diverge.

```sh
PYTHONPATH=src:. python -m experiments.paper.pathway_controls plan
PYTHONPATH=src:. python -m experiments.paper.pathway_controls freeze --directory /path/to/controls
PYTHONPATH=src:. python -m experiments.paper.pathway_controls run --directory /path/to/controls --concurrency 4
PYTHONPATH=src:. python -m experiments.paper.pathway_controls analyze --directory /path/to/controls
PYTHONPATH=src:. python -m experiments.paper.pathway_controls plot --directory /path/to/controls
```

`run` reuses an existing frozen design when no design overrides are supplied. For a small transport test, add `--cases baseline --replicates 1 --nodes 16 --follow-count 4 --horizon 16 --snapshot-steps 0 4 8 16` when freezing. This is a workflow test, not evidence for the 500-node scientific results.

`analyze` recomputes the paper's revised `I_p,I_s,I_h,I_w` using actual raw metadata and the shared analysis implementation. It writes adaptive observation grids and full-trajectory integrals to `analysis/`; it does not impose new `P/E` measures or a first-passage classification. Network inputs are independent of these metrics: `snapshot_inputs/index.json` lists gzip JSON files containing `step`, `node_ids`, `opinions`, the complete directed `edges`, and retained `posts` grouped by sharing account (with the original post author data preserved). `plot` previews those networks using one fixed initial layout per paired state. Checkpoint/source hashes accompany every export. Choosing representative pathway figures requires inspecting the completed full-model trajectories, rather than selecting examples using a control's changed dynamics.

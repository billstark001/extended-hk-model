# Original-paper reanalysis and trajectory figures

This guide covers reanalysis of existing microscopic records and reproducible new runs. Existing raw simulations remain readable. Archived wide-table statistics are preserved separately from the new document database.

## Indices and statistics storage

- `I_p,I_s` use random-reference Jensen–Shannon divergence ratios. Historical JS distance and clustered-reference complements are reconstructed separately in the audit. `I_p` measures departure of the opinion-distance distribution from the random reference; it can rise under consensus and is not bounded by one.
- `I_h` uses the current opinions at directed follow-edge endpoints. Concordance is averaged equally over nodes with positive out-degree and normalized against the uniform baseline `epsilon-epsilon**2/4`. The confidence condition is strictly `distance < epsilon`. Concordance of received followee posts is stored separately.
- `I_w` is the chronological signed trapezoidal integral, without an endpoint complement or clipping to `[0,1]`. It summarizes a trajectory continuously.
- The historical `0.6` cutoff is an unvalidated descriptive split. Grouped plots require an explicit `EHK_IW_CUTOFF` and use low/high `I_w` labels.
- Reanalysis does not call the P/E probe by default. `--include-pathway-probe` adds its independent first-passage labels.

The default statistics database is `outputs/experiments/paper/stats-v3.sqlite`. Standard-library SQLite stores metadata, metric protocols, raw-source identities and arrays in one document per task. Typed NumPy arrays use compressed NPY with pickle disabled. Each document commits in one transaction; adding a statistic does not require a wide-table migration.

Cache identity binds task parameters, recorded raw metadata, available RNG provenance, metric source code and input-file identities. Historical records without seeds retain that missing provenance. Changed inputs and forced reanalysis invalidate previous documents persistently; a failed retry cannot supply an old complete result to plotting. Readers require an explicit selection when multiple metric revisions or raw sources coexist.

## Reanalyze existing raw records

Run these commands from the model repository. Replace the paths with a raw root containing scenario directories such as `s_grad_*` and a separate destination database:

```sh
python -m experiments.paper.analysis.run gradation --raw-root /path/to/existing/raw --origin original-paper --database /path/to/reanalysis/stats-v3.sqlite --plan-only

python -m experiments.paper.analysis.run gradation --raw-root /path/to/existing/raw --origin original-paper --database /path/to/reanalysis/stats-v3.sqlite --limit 1 --concurrency 1

python -m experiments.paper.analysis.run gradation --raw-root /path/to/existing/raw --origin original-paper --database /path/to/reanalysis/stats-v3.sqlite --concurrency 6
```

A successful limited run can be reused by a later full selection. Run `epsilon` and `replicate` similarly; all three modes may share the new statistics database. The positional `MODE INSTANCE_NAME` interface still reads `sim_ws.json`. `--force` recomputes documents under the current metric revision. Missing and failed tasks remain visible in the run status, while successful records can resume independently.

By default, reanalysis also reconstructs historical conventions on the same raw trajectory: the uniform baseline, post versus edge concordance, JS reference direction, distance versus divergence, endpoint complement and sampling coverage. Reconstruction uses deterministic current sampling; exact agreement with old products lacking sampling provenance is not guaranteed. Missing post observations are retained as missing. Use `--skip-metric-audit` for an analysis that does not need this comparison.

## Audit groups and dependent numerical claims

```sh
python -m experiments.paper.pathway_index.audit --database /path/to/reanalysis/stats-v3.sqlite --origin original-paper --analysis-id ANALYSIS_ID --cutoff 0.6 --output /path/to/reanalysis/audit

python -m experiments.paper.pathway_index.grouped_summary --database /path/to/reanalysis/stats-v3.sqlite --origin original-paper --analysis-id ANALYSIS_ID --cutoff 0.6 --output /path/to/reanalysis/audit
```

Use the `ANALYSIS_ID` returned by gradation analysis. The audit exports per-run convention changes, group migrations at the selected cutoff, and edge/post concordance trajectories. `--legacy /path/to/old/stats.db` adds a read-only comparison with archived scores; an archived statistics database is optional.

`grouped_summary` compares the two M9 policies at matched `k_h=0`. It exports run-level observations, group means and standard deviations, and percentage changes with explicit reference-group denominators. Metrics cover rewiring counts, normalized rewiring times, `I_s` area and closed triads. These are descriptive group comparisons. Inspect sample counts, missingness and group migration before reusing a historical cutoff or reported percentage.

## Plotting

The seven original statistics-plot entry points read the document database. Use the default `stats-v3.sqlite`, or set `SIMULATION_STAT_DIR` to its containing directory. `EHK_ANALYSIS_ID` selects a metric revision within a mode, and `EHK_STATS_ORIGIN` selects a raw source. Different modes may have different analysis IDs.

```sh
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.main_scan.plot
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.pathway_index.plot_trajectories
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 python -m experiments.paper.pathway_index.plot
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 python -m experiments.paper.pathway_index.plot_raw
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 python -m experiments.paper.mechanism_cases.plot_cluster_and_repost
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.epsilon_scan.plot
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.replicate.plot
```

The main algorithm comparisons use `k_h=0`; the raw history-group plot retains separate history controls. Readers reject stale documents and do not pool different raw sources or metric revisions.

## Run new simulations

```sh
python -m experiments.paper.run gradation --plan-only
python -m experiments.paper.run gradation --run-dir /path/to/new/raw --freeze-only
python -m experiments.paper.run gradation --run-dir /path/to/new/raw --concurrency 4
```

The same interface supports `epsilon`, `replicate` and `mech`. Use new raw directories. Historical task names and axis order are retained, while new runs record independent PCG root seeds, coordinates, model/recommender parameters, collection settings, float64 precision and executable/source identities. Freeze a formal run after completing code changes; an old unseeded same-name simulation cannot be resumed under a new seed identity.

The independent [process-control workflow](../experiments/paper/pathway_controls/README.md) compares full, `q=0` and `alpha=0` arms from the same complete initial state and RNG. It exports actual directed edges, opinions and posts without requiring P/E thresholds. Small smoke runs check the workflow; representative manuscript figures require completed scientific-scale trajectories.

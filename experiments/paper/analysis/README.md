# Recomputing original-paper statistics

Run from the model repository with its `.venv`. The existing index family is retained: `p_index` is `I_p`, `g_index` is `I_s`, `h_index` is `I_h`, and `pathway_integral` is `I_w`. By default, `I_w = integral I_h dI_p + (1-I_p(T))*I_h(T)`, using signed chronological trapezoids. The completion extends the endpoint to `I_p=1` at terminal homophily, so the tendency measure retains that homophily for consensual endpoints. `I_p` measures deviation from the random distance reference; it can increase under consensus and is not bounded by one. Neither `I_p` nor `I_w` is clipped.

```sh
.venv/bin/python -m experiments.paper.analysis.run gradation \
  --raw-root /path/to/new/raw --database /path/to/new/stats-v3.sqlite \
  --limit 1 --concurrency 1
```

Drop `--limit` to fill the full selected design. The modes are `gradation`, `epsilon`, and `replicate`. The positional workspace alias remains available; `--raw-root` instead accepts a direct directory containing scenario directories. Use `--origin` to assign an explicit source label. `--plan-only` prints the selection without reading raw records or creating the database.

`--endpoint-complement` is the default. Add `--no-endpoint-complement` to calculate the observed integral alone. The boolean is stored in each record and metric protocol and changes the analysis identity, preventing reuse across the two conventions. Historical values retain their original recorded convention; an unrecorded convention is not assigned the new default.

Each committed SQLite document contains the complete **recorded** raw metadata, the requested metadata separately, the metric protocol, source-file identities, and typed NumPy statistics. Missing historical RNG metadata remains unavailable; seeds from a newer execution design are never assigned retrospectively. The protocol identity binds numerical and collector source hashes, binding sources, and relevant Python/package versions. Metadata, trajectory manifests and finish markers are hashed; bulk raw files use size and nanosecond modification time, rather than full byte hashes. Changes invalidate old documents before retrying.

The store uses standard-library SQLite transactions with one writer, compressed `.npy` arrays with pickle disabled, and JSON documents. New fields do not require table migrations. Archived ORM databases are rejected; use a separate new path. Reruns reuse matching complete records; `--force` explicitly invalidates and recomputes them. Failure or unavailable raw inputs leave stale documents marked incomplete. Readers refuse such a selected dataset. Limited and full selections have separate resumable journals while sharing the same metric protocol identity.

The default calculation also reconstructs historical conventions from raw trajectories: the `/8` versus `/4` baseline, post versus edge concordance, clustered versus random reference, JS distance versus divergence, and endpoint completion. The selected main result and both completed/uncompleted full-trajectory values are stored separately. It uses the revised adaptive observation grid. Missing post observations are not imputed; sequential historical comparisons share the available observations and disclose their actual prefix. `remove_endpoint_complement` refers to that common-prefix comparison, not the uncompleted full trajectory. `--skip-metric-audit` omits these extra calculations.

```sh
.venv/bin/python -m experiments.paper.pathway_index.audit \
  --database /path/to/new/stats-v3.sqlite --analysis-id ANALYSIS_ID \
  --origin SOURCE_LABEL --output /path/to/audit
```

The audit writes continuous convention changes and a migration table for the historical `0.6` **low/high index split**, identifying the selected completion convention and exporting both full-trajectory values when recorded. It does not validate this cutoff as a PbS/SbP classifier. `--legacy OLD.db` optionally reads archived scores without modifying them. A subsequent grouped summary retains the convention alongside run-level values and in its receipt.

P/E first-passage measurements are an independent optional comparison. They are not required by the four-index analysis or its default storage. Enable `--include-pathway-probe` to collect them; only then does the protocol require and hash the native probe executable.

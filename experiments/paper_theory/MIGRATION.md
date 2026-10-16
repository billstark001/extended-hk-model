# Theory experiment migration and pruning

The model repository owns active code; original paper artifacts remain
unchanged. `migration_sources.json` records historical source paths and SHA256
values, not a registry of supported modules. Some recorded destinations have
since been retired. It must not be used to discover runnable designs.

The 2026-10-17 cleanup follows the theory structure revision proposal. Current
experiment names, coverage and the shared lifecycle are in
[README.md](README.md).

| Historical implementation | Current disposition |
| --- | --- |
| Energy-scan generation, channel probes and whole-grid plot menus | Removed; the previous design cleanup still applies |
| Energy-scan node/edge fits and single-slot figures | Retained as offline capabilities; old baseline/expanded NPZ readers, fit APIs, and `fit-slot` / `plot-slot` CLI actions |
| Old `scaling`, `release`, `population_closure` sweeps | Generators/plots removed; current named evidence tasks own the relevant comparisons |
| Duplicate `micro_validation/population` design/analysis/plot | Removed; current evidence uses the shared lifecycle and model adapter |
| Historical validation/follow-up grids | Saved 762/346-job plans and read-only record/table adapters remain; all historical generators and executors removed |
| Historical figure menus and parameter-clock fit scripts | Removed; table analyses needed by the supplement remain |
| Current `main_design` | Five named research topics; formal and smoke entry modules separated |
| Nominal-N analytic closure scan | Removed; 90 exclusive formal tasks and 20 exclusive smoke tasks omitted |
| Storage converter `evidence/migrate.py` | Copied to Cache original-data backup, then removed with its CLI/types/tests |
| Runtime adapter, codecs, frozen readers, scientific estimators | Retained behind the shared `common.experiments` lifecycle used by both paper studies |

Retained current task IDs and physical requests are unchanged. The revised
aggregate plan has 1794 tasks and a new fingerprint; old 1884-task data is still
read using its own frozen plan. The two retained historical batches are read directly from their frozen plans,
including their original ordering and metadata. Their
scientific protocols remain distinct from the current manuscript protocol.

The Cache native dataset contains 828 measure NPZ, 1056 micro binary and 228
shared pair records. Its completed migration/analysis/render receipts remain
with the dataset. Original JSON files and the retired converter copy remain in
`/Volumes/Cache/extended-hk-model-paper-original-data/`. No simulations, further
migration, raw-data removal, or bulk publication was performed by this cleanup.

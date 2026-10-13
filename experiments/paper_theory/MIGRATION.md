# Paper experiment migration and refactoring

The model repository owns the active experiment code. Original paper artifacts remain unchanged. Their paths and SHA256 values in `migration_sources.json` record provenance, not imported dependencies or required source locations. The implementation was then reorganized by scientific responsibility; its files are not expected to remain byte-identical to those historical scripts.

| Original artifact code                                         | Current implementation                                                                                                       |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `energy_curve_parameterization/generate.py`                    | Historical `energy_scan/baseline/{design,run,storage,analyze,plot}` libraries                                                |
| `energy_curve_parameterization/extended_log_grid/generate.py`  | Historical `energy_scan/expanded/{design,run,storage,fit,analyze,plot}` libraries                                            |
| `energy_parameter_scaling/generate.py`                         | Historical `energy_scan/scaling/{design,run,relations,analyze,plot}` libraries                                               |
| `edge_energy_continuous_20260928/scripts/analyze.py`           | `release_clocks/analyze/{curves,fitting,tables,driver}`                                                                      |
| Parameter-clock calculations and plots                         | `release_clocks/parameter_clocks/{data,models,analyze,plot}` libraries                                                       |
| Core and followup parameter generators                         | Study-owned `design.py` modules selected by `designs.py` presets                                                             |
| Both custom Go adapters                                        | One `runtime/engine`; shared initialization, projection and stepping adapter with explicit policies                          |
| Revision build, execution, records and estimate                | Study build selection in `runtime/build`; reusable contracts/execution in `src/theory/runtime`; old suite dispatcher removed |
| Generic revision/ehk tools                                     | `src/common`                                                                                                                 |
| Distribution/energy comparisons, clocks, exposure convolutions | `src/common/numerics`, `src/theory/validation`, `src/theory/release`, `src/theory/kernels`                                   |
| Core/followup analysis and scientific reviews                  | `micro_validation`, `release_clocks`, `operator_validation`, `kernel_validation` analysis packages                           |
| Paper-specific review and figures                              | Reusable `manuscript/{reports,figures}` libraries; uncalled fixed figure/claim/budget script assemblers removed              |
| Scientific verification assertions                             | `tests/theory`, `tests/experiments`                                                                                          |
| `run_core.sh`, `run_followup_20261007.sh`                      | Retired; active generation uses `python -m experiments.paper_theory run` with main/main_smoke                                |
| Before/after runtime-reuse verification and timing scripts     | Retired after solver reuse; no active historical comparison workflow                                                         |

There is no `revision` package, dated followup executable or compatibility import/command entry. Historical batch names and the `revision` request-ID placeholder remain only where required to preserve frozen task identities and locate recorded evidence. The independent microscopic and mesoscopic runtimes remain separate repositories.

Historical design/analysis functions preserve frozen identities, seeds, ordering and protocols for reading recorded evidence. They are no longer generation CLI routes. The current main and main_smoke plans retain their complete original contents and task IDs. Superseded evidence paper/smoke generators are removed. Transport/storage changes preserve initial RNG policies, clipping, projection, exact energy normalization, recording cadence and analysis definitions. Quantile-clock estimators remain available, and process branches are not reclassified as independent initial states.

## Data and products

Use the unified commands in [README.md](README.md). Original inputs may stay on external storage or be exposed through ignored links. Input roots and generated output roots are independent. The [installation/data-link guide](../../docs/runtime-dependencies.md) describes all three repositories and the expected input layouts.

Cached analysis preserves recorded numerical protocols and original solver/adapter identities. It does not rewrite them using the current build. New generation records current source/binary identity and must use a separate run location when incompatible with old caches. Every generated build, plan, raw record, fit, report, plot, cache and link remains ignored.

The frozen manuscript selection needs plans, binding/status files and per-task trajectories. Small review CSVs cannot replace that evidence. Historical data can still be read without retaining its code entry points. Pure forwarding wrappers, old function/profile aliases and redundant path re-exports are removed. The canonical CLI is the only Python `main()`; launch orchestration receives parsed arguments directly. Resume requires the original run manifest, while offline readers retain original JSON/NPZ formats. CAS 2026 is a literature citation, not a migrated dataset or rerun target. No bulk-data upload or production simulation is automatic.

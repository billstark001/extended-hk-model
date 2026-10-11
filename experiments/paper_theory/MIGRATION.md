# Paper experiment migration and refactoring

The model repository owns the active experiment code. Original paper artifacts remain unchanged. Their paths and SHA256 values in `migration_sources.json` record provenance, not imported dependencies or required source locations. The implementation was then reorganized by scientific responsibility; its files are not expected to remain byte-identical to those historical scripts.

| Original artifact code | Current implementation |
| --- | --- |
| `energy_curve_parameterization/generate.py` | `energy_scan/baseline/{design,run,storage,analyze,plot,cli}` |
| `energy_curve_parameterization/extended_log_grid/generate.py` | `energy_scan/expanded/{design,run,storage,fit,analyze,plot,cli}` |
| `energy_parameter_scaling/generate.py` | `energy_scan/scaling/{design,run,relations,analyze,plot,cli}` |
| `edge_energy_continuous_20260928/scripts/analyze.py` | `release_clocks/analyze/{curves,fitting,tables,driver}` |
| Parameter-clock calculations and plots | `release_clocks/parameter_clocks/{data,models,analyze,plot,cli}` |
| Core and followup parameter generators | Study-owned `design.py` modules selected by `designs.py` presets |
| Both custom Go adapters | One `runtime/engine`; shared initialization, projection and stepping adapter with explicit policies |
| Revision build, runner, records and estimate | `runtime/{build,executor,runner,store,dataset,estimate}` |
| Generic revision/ehk tools | `src/common` |
| Distribution/energy comparisons, clocks, exposure convolutions | `src/common/numerics`, `src/theory/validation`, `src/theory/release`, `src/theory/kernels` |
| Core/followup analysis and scientific reviews | `micro_validation`, `release_clocks`, `operator_validation`, `kernel_validation` analysis packages |
| Paper-specific review, figure assembly and fixed claim audits | `manuscript/{reports,figures,audit}` |
| Scientific verification assertions | `tests/theory`, `tests/experiments` |
| `run_core.sh`, `run_followup_20261007.sh` | Unified `python -m experiments.paper_theory pipeline`; shell files removed |
| Before/after runtime-reuse verification and timing scripts | Retired after solver reuse; no active historical comparison workflow |

There is no `revision` package, dated followup executable or compatibility import/command entry. Historical batch names and the `revision` request-ID placeholder remain only where required to preserve frozen task identities and locate recorded evidence. The independent microscopic and mesoscopic runtimes remain separate repositories.

Refactoring preserves six full validation/ensemble designs and their two smoke designs, including task identities, seeds and ordering. It preserves separate initial RNG policies, clipping, projection, exact energy normalization, conditional branch auditing and analysis definitions. Quantile clocks are not replaced with integral clocks, and process branches are not reclassified as independent initial states.

## Data and products

Use the unified commands in [README.md](README.md). Original inputs may stay on external storage or be exposed through ignored links. Input roots and generated output roots are independent. The [installation/data-link guide](../../docs/runtime-dependencies.md) describes all three repositories and the expected input layouts.

Cached analysis preserves recorded numerical protocols and original solver/adapter identities. It does not rewrite them using the current build. New generation records current source/binary identity and must use a separate run location when incompatible with old caches. Every generated build, plan, raw record, fit, report, plot, cache and link remains ignored.

The frozen manuscript selection needs plans, binding/status files and per-task trajectories. Small review CSVs cannot replace that evidence. Historical data can still be read without retaining its code entry points. CAS 2026 is a literature citation, not a migrated dataset or rerun target. No bulk-data upload or production simulation is automatic.

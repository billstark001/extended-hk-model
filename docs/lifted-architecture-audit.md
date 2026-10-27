# Lifted module audit — 2026-10-28

The audit keeps model dynamics, frozen task identities, seeds, checkpoint formats
and raw bindings unchanged. No original experiment is launched or resumed during
the refactor. Python callers use the owning module directly; no forwarding modules
are retained for the internal names that moved. CLI and stored-data compatibility
remain supported.

## Responsibility changes

| Responsibility | Owning module after the audit |
| --- | --- |
| Incremental journal indexing and generic remaining-work aggregation | `src/common/experiments/progress.py` |
| Dependency queue lifecycle through the existing `TaskAdapter` | `src/common/experiments/execution.py` and `scheduling.py` |
| Per-stage journal routing and committed coverage | `src/common/experiments/lifecycle.py` |
| Cumulative wall-limit extension, preserving used time | `src/common/runtime/budget.py` |
| Immutable archival of attempt receipts | `src/common/experiments/receipts.py` |
| Authenticated restoration of a standard adapter binary/module manifest | `src/common/runtime/go.py` |
| Lifted timing-row contract and model-specific cost prediction | `src/lifted/runtime/costs.py` and `forecast.py` |
| Cost-evidence files and lightweight metadata reads | `experiments/paper_lifted/evidence/costs.py` |
| Engine selection, pilot ownership and scientific binding assembly | `experiments/paper_lifted/bindings.py` |
| Family/ratio fairness and replica-chunk priority | `experiments/paper_lifted/scheduling.py` |
| Adaptive tail, validation and local-window design | `experiments/paper_lifted/evidence/refinement.py` |

`progress.py` now assembles observation and presentation rather than owning the
cost model or deciding scheduling priorities. The pure lifted forecast receives
metadata through an injected reader and imports no experiment module or file
codec. One `measurement()` function supplies the row schema used by calibration,
offline runtime-cost tables and live prediction.

`study.py` authenticates model records, adapts budget cancellation, chooses the
execution policy and delegates to the shared lifecycle. It no longer owns another
ready-stream executor or constructs adaptive scientific cases inline. The common
executor acknowledges children only after a validated cache hit or successful
commit; deferred persistence never releases children. Stage coverage includes
records committed in previous attempts.

`run.py` retains startup, session ownership, resource configuration and phase
orchestration. Engine selection belongs to the study's bindings module; immutable
binary copying and cumulative-ledger mechanics belong to common services.

## Existing reuse retained

- Lifted initialization continues to use `theory.runtime.population`, validated
  `theory.Parameters` and the existing theory Go initializer. The source adapter,
  initialization law and stream policy are preserved byte-for-byte.
- Public SMP/smp-meso libraries retain ownership of stepping, recommendation,
  projection and classification. No solver is copied into the study.
- The existing `TaskAdapter`, task execution, cancellation, `RunStorage`, immutable
  plan/manifest helpers and plotting primitives remain the execution foundation.
- Commitment/outcome bounds stay in `src/lifted`. Energy-release clocks and
  conditional commitment bounds have different definitions; this audit does not
  merge them through an unrelated statistical API.

## Deferred migrations, with inline TODOs

| Location | Follow-up direction and reason for postponement |
| --- | --- |
| `paper_lifted/bindings.py` | Extract the uniform fixed-degree initializer and the population request surface into a shared model-runtime owner when both studies migrate their schemas; preserve seed policy, native checkpoints and measured binaries. |
| `paper_theory/evidence/bindings.py::restore_binaries` | Normalize the heterogeneous kinetic/instrumentation manifests before adopting `common.runtime.go.restore_adapter`; retain APFS execution staging and FreshReadPolicy. |
| `paper_theory/run.py::launch` | Decouple legacy worker/budget manifests after preserving native batch, spool and publication ownership rules. |
| `src/ehk/micro/simulation.py::SimulationAdapter` | Adopt the common journal/ETA/deferred reporting after SMP binding cancellation and completion evidence are compatible; retain directory/SQLite completion and isolated-batch ownership. |
| `src/common/runtime/costs.py::estimate` | Move the legacy jobs/meta/payload translation to its study before sharing pure cost aggregation with task-based studies. |

The ehk/theory changes in this audit are comments only. Their runtime APIs,
sampling, task order, file schemas and execution policies are unchanged.

## Compatibility and checks

Protected request/model/design/math sources and native runtime/binary hashes still
must match. Extracted monitoring, resource and execution helpers are explicitly
classified as operational source revisions and remain separately audited.
Optional follow-up policy remains study-owned: its tasks are separately frozen,
and saved follow-up plans are read without regeneration.

Before moving the adaptive designer, a synthetic committed dataset exercised all
five follow-up stages. After the move, its **282 task IDs, parameters, seeds,
parent-record hashes and stage ordering match exactly**. Only the new source
snapshot and its containing plan fingerprint differ for a newly generated plan.

Native validation resumes an old frozen smoke dataset with a different worker
count and compares every numerical result with the old uninterrupted baseline.
Cached resumes, cumulative budget extension, standalone analysis/plotting and a
real native deadline cancellation are checked again. Generic service tests also
verify that deferred/failed persistence cannot release a child and that final
coverage is published on failure.

Local receipts, comparison data and logs are under
`outputs/paper_lifted/architecture-20261028/`; `verification.json` records the final
checks. The main experiment's plan, budget, binding and cumulative ledger are
verified without modification.

Final checks: lint and format passed; mypy passed for 65 source files; the full
suite passed **215 tests and five subtests** in 140.77 seconds. Native continuation
and deadline checks passed, including old-data resume and independent CSV export.

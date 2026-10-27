# Global progress and continuation — 2026-10-28

The original run stopped at its configured core deadline. It retained
17,162/37,186 committed tasks, leaving 20,024 unfinished tasks. The displayed
`micro_probe_491ab7c644c32133c83982ac` failure was a budget timeout followed by
a retry that immediately encountered the expired deadline. No completed raw
record exists for that cancelled chunk. The original records, frozen plan,
binding, budget and cumulative ledger have not been changed by this update.
Matplotlib's font-cache message is ordinary initialization.

## Continue the fixed experiment

Run from the model repository in a foreground terminal:

```sh
cd /Users/billstark001/Desktop/programs/extended-hk-model
caffeinate -i .venv/bin/python -m experiments.paper_lifted resume \
  --work-dir /Volumes/Cache/extended-hk-model/paper_lifted/formal-20261027-phases \
  --workers 4 --backend accelerate \
  --schedule balanced --core-only --wall-hours 20
```

This command has **not** been run against the original experiment during
implementation. It resumes the same frozen tasks and input checkpoints, retaining
all existing raw records. The cancelled chunk starts again from its committed
input with its original seed.

`--wall-hours 20` is a **total cumulative limit**, including the approximately
6.6 hours already spent. It leaves approximately 13.4 execution hours. It is
idempotent: repeating the command does not add another twenty hours or reset used
time. The unchanged `budget.json` records the original calibration; the actual
limit lives in `wall_budget.json`, with explicit requests audited in
`wall_budget_extensions.jsonl`. Reductions of an existing limit are rejected.
The extension and execution configuration are applied only after obtaining the
original exclusive session lock.

`--core-only` completes all 37,186 already frozen tasks before considering any
optional adaptive work. It includes all frozen mesoscopic, size, bridge, moment,
reference and noise-control tasks. It suppresses only the subsequent adaptive
tail/validation/window selection. With this option, forty minutes are reserved
for export; the unused adaptive reserve is available to the fixed tasks.
The process ends as soon as the selected work and exports complete.

The current operational estimate is roughly eight additional hours at four
workers. This includes unobserved mesoscopic strata priced from the pilot and
is not a completion guarantee. A later explicit increase of the cumulative
limit can resume any remaining tasks. Remove `--core-only` on a later resume
to run the separately frozen adaptive follow-up workflow.

By default this command runs execution, analysis and plotting. To separate them:

```sh
# Add this to the continuation command to perform execution alone:
# --phase run
.venv/bin/python -m experiments.paper_lifted analyze --work-dir /Volumes/Cache/extended-hk-model/paper_lifted/formal-20261027-phases
.venv/bin/python -m experiments.paper_lifted plot --work-dir /Volumes/Cache/extended-hk-model/paper_lifted/formal-20261027-phases
```

## Observe progress

Use another foreground terminal:

```sh
cd /Users/billstark001/Desktop/programs/extended-hk-model
.venv/bin/python -m experiments.paper_lifted status \
  --work-dir /Volumes/Cache/extended-hk-model/paper_lifted/formal-20261027-phases \
  --watch --interval 15
```

`status --progress` emits one JSON snapshot instead. The watcher is read-only;
interrupting it does not interrupt the experiment. It consumes local journal
increments and small metadata samples, without repeatedly hashing all native
checkpoints. Ordinary `status` and scientific analysis retain their full
authentication checks.

The executor prints a global heartbeat every fifteen seconds, including total
completion, micro/coarse probe coverage, active workers, failures, deferrals,
remaining-work ETA and remaining dispatch budget. A task exceeding its estimated
duration makes the ETA unknown until new evidence arrives, instead of implying
zero remaining time. Unknown strata and estimates exceeding the remaining budget
are explicitly identified. Progress estimates are operational; they do not change
sampling or enter scientific statistics.

Costs update from actual task times, native step rates and original mother-path
futures. Completed geometric-boundary queries are priced separately from queries
requiring long continuation. Mesoscopic cost strata distinguish ratio, grid,
variant, layer and query age. Cached tasks are counted once across resumes.

## Scheduling and compatibility

`balanced` uses a dependency-aware round-robin queue across micro, mesoscopic and
diagnostic families and ratios. Children are released only after their parent
record commits. Within a lane, first replicate chunks cover queried states before
additional chunks. The next ready task takes a free slot immediately; there is
no four-task batch barrier and no requirement to finish all microscopic probes
before starting mesoscopic work. `--schedule stages` retains the original staged
order for comparison.

Workers are execution configuration. New freezes derive `reference_workers` from
their pilot and need no `--workers`. Legacy `budget.workers` is treated as a cost
reference, not a requirement. Resume may change `--workers` within 1–8 while task
IDs, seeds, checkpoints, sample counts and backend remain fixed. Timing estimates
depend on actual concurrency and hardware; a worker change does not preserve a
six-to-eight-hour timing prediction.

The original scientific binding and raw records remain immutable. Protected
model/request/design/math sources, runtime source manifests and executable hashes
still must match. Execution-only changes are recorded separately in
`execution.json` and `executions/<id>.json`; previous `attempt.json` receipts are
archived there before the last-attempt pointer is replaced. Backend changes and
scientific source changes still fail validation.

Budget cancellation now emits `deferred`, makes no scientific completion record
and is not retried. Genuine computation/protocol errors remain failures. Exit code
2 means incomplete coverage, including a normal budget stop; `attempt.json` and
the terminal summary provide `stop_reason` to distinguish the cases. Pending work
and physical unresolved paths remain separate in probability exports.

## Validation artifacts

Lint and format checks pass; mypy passes for 59 source files. The full suite has
**210 passed tests and five passed subtests** (116.87 seconds). Native Go
continuation tests pass. A final six-worker cached resume also preserves all raw
hashes, retains the cumulative limit without repeating the extension option, and
archives the previous receipt.

Local drivers, receipts and logs are under
`outputs/paper_lifted/execution-20261028/`. Native validation uses smoke profiles:

- `native_driver.py` / `native_report.json`: legacy frozen data resumed with
  three workers; every numerical result matches the old two-worker baseline.
  A one-worker cached resume changes no raw bytes. An explicit cumulative budget
  increase preserves used time and the frozen plan/binding.
- `deadline_driver.py` / `deadline_report.json`: a real running native task is
  cancelled at its limit, deferred once, never retried and produces no raw record.
- `pytest-final.log`: full regression suite, with separate lint, format, mypy and
  native Go continuation checks recorded in `verification.json`.

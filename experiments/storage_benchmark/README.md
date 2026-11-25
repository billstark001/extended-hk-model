# Isolated research-storage comparison

This experiment never selects or migrates the production store. All databases,
dependency packages, requests, logs and results live under an explicit output
directory. Its default comparison has three application interfaces over SQLite:

- `current`: the existing `DocumentStore`, with eager NPY/zlib/base64 decoding.
- `tinymongo`: TinyMongo 1.3.1's native SQLite backend, the same array codec and
  nine scalar indexes. A benchmark-only connection override enforces FULL.
- `indexed_split`: SQLite with the same nine scalar indexes and separate
  compressed NPY BLOB rows. Metadata, layout and arrays commit together.

The opt-in `sqlite_optimized` variant retains the compressed NPY codec and
transactional metadata/payload snapshot. It uses streaming result cursors,
compound query indexes and integer payload references; `indexed_split` remains
the unchanged baseline.

See [CORPUS.md](CORPUS.md) for the measured sources and explicit synthetic-size
assumptions, and [RECOVERY_PROTOCOL.md](RECOVERY_PROTOCOL.md) for fault injection.

## Reproducing a campaign

Use the repository Python environment. Install benchmark dependencies in an
isolated target, without changing `pyproject.toml` or the shared environment:

```sh
mkdir -p outputs/benchmarks/storage_20261122
.venv/bin/python -m pip install --target outputs/benchmarks/storage_20261122/vendor --report outputs/benchmarks/storage_20261122/install-report.json tinymongo==1.3.1 tinydb==3.15.2 portalocker==4.4.2 psutil==7.1.3
export PYTHONPATH=outputs/benchmarks/storage_20261122/vendor:src:.
.venv/bin/python -m experiments.storage_benchmark.run prepare --directory outputs/benchmarks/storage_20261124/formal --vendor outputs/benchmarks/storage_20261122/vendor --repetitions 3 --query-repetitions 2
.venv/bin/python -m experiments.storage_benchmark.run run --directory outputs/benchmarks/storage_20261124/formal
.venv/bin/python -m experiments.storage_benchmark.run report --directory outputs/benchmarks/storage_20261124/formal
```

The corrected campaign is complete in that directory; see `FINDINGS.md` and
`validation.json`. For a fresh reproduction, choose another empty directory.
An existing isolated vendor directory can be reused without reinstalling it.

To compare only the two split SQLite variants, freeze a fresh campaign with
`--backends indexed_split sqlite_optimized`. `--actions` selects a nonempty
subset of the eight default actions or the optional `scope_metadata` and
`scope_full` queries; `load` always runs first. For example:

```sh
.venv/bin/python -m experiments.storage_benchmark.recovery --backend indexed_split --directory outputs/benchmarks/storage_optimization_20261124/recovery_final/indexed_split --rounds 6
.venv/bin/python -m experiments.storage_benchmark.recovery --backend sqlite_optimized --directory outputs/benchmarks/storage_optimization_20261124/recovery_final/sqlite_optimized --rounds 6
.venv/bin/python -m experiments.storage_benchmark.run prepare --directory outputs/benchmarks/storage_optimization_20261124/formal --vendor outputs/benchmarks/storage_20261122/vendor --backends indexed_split sqlite_optimized --actions selective_metadata scan_metadata scan_full nested_metadata state_update replace_full --repetitions 3 --query-repetitions 1
.venv/bin/python -m experiments.storage_benchmark.run run --directory outputs/benchmarks/storage_optimization_20261124/formal
```

Omitting these options retains the original three backends and eight actions.
Backend order follows the selected list, rotating within it for each profile
and repetition. Selected actions run in protocol order: reads first, followed
by `state_update` and `replace_full`. Choices are part of the frozen plan and
fingerprint; `run` and `report` do not accept overrides. Recovery accepts one
explicit backend per invocation, so both variants require their own six-round
evidence. Reports for only the selected backends can be linked from the plan;
old recovery results do not validate a changed implementation.

`prepare` refuses to replace an existing plan. `run` verifies frozen plan,
catalog, source pack and source-code hashes, then resumes matching completed
jobs. Each backend/repetition uses a separate database. One measured child runs
at a time, with a fresh process for every action, a 4 GiB RSS bound and a 900 s
action timeout. Backend order rotates across repetitions and profiles. The
coordinator stops below 8 GiB available disk space. Do not run simulation,
recovery or test jobs concurrently with performance measurement.

Before generating any catalog, preparation compares documents from the live
sources with documents regenerated after the real sorted-JSON source-pack
roundtrip. It checks the first/last task of every family, all mechanism tasks,
each analysis revision and adjacent revision boundaries, plus actual-source
examples. This preflight ignores the catalog's `--limit`, so a small trial still
checks mechanism and revision cases. `source-pack-preflight.json` records exact
header and dtype/shape/array-byte content hashes and is frozen in the plan. A
mismatch prevents catalog generation and all database writes. New campaigns
use corpus revision v2, which assigns mechanism RNG draws in sorted field order;
the 2026-11-22 `formal` and `formal_v2` evidence remains unchanged.

An interrupted attempt without a completed matching result is rejected on resume.
Start a new campaign rather than silently mixing insertion and replacement, or
repeating a partially applied state toggle. A completed campaign can still resume
by skipping all recorded successful jobs.

Worker failures update `status.json` to `failed` or `interrupted`, preserving the
failed job and completed count. Best-effort `ps` snapshots before and after each
new job are saved to `environment-observations.jsonl`, outside measured calls.
They contain executable names and CPU/RSS observations, without command-line
arguments. They help identify competing computation; they do not isolate CPU
resources or turn operating-system percentage estimates into instantaneous load.

## Measurement boundaries

All stores use WAL and `synchronous=FULL`, checked at runtime. Every put or
state update is one committed operation. Load latency includes validation,
serialization, compression and commit inside the backend call. Corpus
generation and independent content hashes are outside that timer; load
end-to-end time is also retained. Initial open/index-creation and final explicit
checkpoint are separate fields, rather than hidden in per-document latency.
SQLite's normal automatic checkpoints remain enabled.

Point reads and state updates use up to 200 deterministic selections. Full
replacement uses 20 distinct keys at each full database size, after all other
workloads. Each replacement changes a scalar epoch and rewrites the complete
document. Its input read and output validation are outside the put timer. Queries
materialize identical logical results and validate IDs and exact scalar/array
values outside the timed call. Query repetitions within one child retain first
and subsequent timings; each process repetition is independent. Fresh processes
do **not** imply cold disk: the operating system page cache is uncontrolled.

RSS is sampled every 10 ms over the whole action, including untimed correctness
checks and checkpointing. Baseline, peak, delta, process-lifetime high-water and
parent-observed peaks are retained separately. It is not a query-only allocation
measurement. Query result lists are released before the next repetition. The
current implementation therefore measures its storage API's eager decoding,
without additional downstream plotting/dataframe copies.

The initial native TinyMongo `replace_one(..., upsert=True)` insertion pilot is
retained in the superseded `formal` output directory. It revealed a collection
scan per new insertion. The corrected adapter used in `formal_v2` and later campaigns uses the public
`insert_many([document])` prepared path for new keys, still committing exactly
one document at a time. Existing keys use `replace_one`; the separate
`replace_full` workload exposes its collection-scan cost rather than assuming
insertion and full recomputation have the same performance. This is a
single-writer adapter; the new-key check is not a concurrent-writer upsert claim.

The main campaign covers single-writer summary/statistics storage, not raw
simulation archives or several concurrent writers. Payload separation uses
BLOBs in the same SQLite database, rather than external files. Recovery results
are writer-process SIGKILL tests, not power-loss durability tests. None of these
measurements alone establishes which application API is easiest to maintain.

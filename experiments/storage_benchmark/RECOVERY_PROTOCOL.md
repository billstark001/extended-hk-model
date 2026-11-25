# Writer process interruption and recovery protocol

This is an isolated correctness test for the `current`, `tinymongo`, `indexed_split` and opt-in `sqlite_optimized` benchmark adapters. It kills only a writer subprocess with POSIX `SIGKILL`. The benchmark coordinator remains alive. It does not simulate an operating system crash, power loss or an unplugged drive, and its runtime is not a throughput measurement.

All adapters must use SQLite WAL with `synchronous=FULL`. The runner reads their actual settings and rejects a different configuration. It does not disable syncing or change the production analysis store.

## Confirmed and unconfirmed operations

The coordinator generates deterministic documents containing scalar identity and state fields, JSON metadata, flat `float64`/`int32` arrays and a small nested `float32` array. Metadata records every array leaf's path, dtype, shape and SHA256. A second document hash covers the complete logical document, including completion state, nesting and the array contents.

The writer commits one adapter operation and sends its operation index and expected document hash through a pipe. The coordinator verifies that ACK, appends it to its own JSONL journal, flushes and calls `fsync`. The writer cannot start another operation until the coordinator permits it. Thus the confirmed boundary comes from the coordinator's durable journal, not from a backend row count or a child-side completion flag.

Two confirmed anchor keys are never overwritten. A third key is replaced with a new metadata/array generation and subsequently receives a multi-field `complete`/`stale`/`state_epoch` update. A final key tests insertion before acknowledgement.

For an update to an existing key, recovery may return the latest ACKed version or the next issued, unacknowledged whole version. Earlier ACKs do not require retaining overwritten historical versions as current values. Recovery must correspond to one of these complete transaction prefixes; a mixture of the old metadata, new arrays and partially updated status fails.

## Fault positions

The default six independent rounds cover:

| Round | Fault position | Target operation |
| --- | --- | --- |
| 0 | After the coordinator has fsynced the ACK | Initial mutable document insert |
| 1 | Inside the adapter's transaction, before COMMIT | Replace metadata and array generation |
| 2 | After the coordinator has fsynced the ACK | Multi-field completion and stale-state update |
| 3 | Inside the adapter's transaction, before COMMIT | The same multi-field state update |
| 4 | After the coordinator has fsynced the ACK | New key insert |
| 5 | Inside the adapter's transaction, before COMMIT | New key insert |

The pre-commit position uses the adapter's opt-in `recovery_before_commit` hook after user-document DML and before COMMIT executes. The child blocks inside this hook until the coordinator kills it. This is a definite uncommitted transaction boundary, rather than a timing guess made after emitting a message immediately before `put()`. The hook is absent from ordinary timing runs. Timeouts turn a missing hook or stalled child into a failed round.

## Independent reopening and checks

After the writer has exited with `-SIGKILL`, the coordinator opens a new backend instance in read-only mode. It verifies:

1. All ACKed immutable keys and every required key still exist.
2. Recovered values match an allowed complete transaction prefix by SHA256.
3. Array payloads remain decoded arrays whose dtype, shape and bytes agree with their metadata.
4. Completion and stale-state fields were updated atomically.
5. `get()` agrees with the full-document query and `count()` agrees with its length.
6. SQLite `integrity_check` returns `ok`.

An operation with no confirmed ACK may be absent or present as the next complete version. The runner does not infer durability from whether that unconfirmed operation happened to survive. Application-specific read-only behavior of an adapter, including any third-party catalog initialization, belongs in the broader benchmark report.

The evidence directory preserves the parent ACK journal, per-round result and hashes, and the recovered database. Each round records SHA-256 identities of `backends.py` and `recovery.py`, plus `sqlite_optimized.py` for that backend, and fails if any recorded source changes during the round. A failed or timed-out round stays a failure in the overall report; a partially successful run is not labelled complete and passing.

## Running separately from performance timing

Use a new empty output directory and the benchmark's isolated dependency path, for example:

```sh
PYTHONPATH=outputs/benchmarks/storage/vendor:src:. .venv/bin/python -m experiments.storage_benchmark.recovery --backend current --directory outputs/benchmarks/storage/recovery/current --rounds 6
```

Run the same command for the other adapters with separate directories. At least five independent rounds are required; six exercise the entire default schedule. Do not run these tests in the background while measuring storage performance. This protocol retains the workflow's single writer; a two- or four-writer contention experiment would be a separate interface test, not part of its recovery result.

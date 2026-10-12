# hdd-bridge

Copy completed simulation folders from a fast source disk to an archive disk.
The source is retained. Requires **Go 1.25+**, using `os.Root` to constrain file
operations to the selected roots, including during symlink races.

## Build and usage

From the repository root:

```sh
(cd tools/hdd-bridge && go build .)
tools/hdd-bridge/hdd-bridge -src /ssd/sim-cache -dst /hdd/sim-archive
tools/hdd-bridge/hdd-bridge -src /ssd/sim-cache -dst /hdd/sim-archive -interval 30s -minelapsed 2m
tools/hdd-bridge/hdd-bridge -src /ssd/sim-cache -dst /hdd/sim-archive -once -verify -fsync
```

On Windows, build in `tools/hdd-bridge` and run, for example:

```powershell
.\hdd-bridge.exe -src D:/sim-cache -dst E:/sim-archive -verify -fsync
```

Both disk roots must already exist and must not overlap, including via symlinks.
A missing root is an error; the tool never creates a replacement mount directory.
If either root disappears or changes identity while running, scans fail until
the original root is restored or the tool is restarted.

## Eligibility and transfer

- Only immediate source subdirectories are scanned. Their contents are copied recursively.
- A folder needs a regular `finished*` file. The newest marker must be at least
  `-minelapsed` old (default 5 minutes). A future timestamp delays eligibility.
- Any top-level `lock*` entry blocks the folder, including directories and dangling links.
- Source symlinks and special files fail the folder rather than silently producing
  an incomplete archive. Destination symlinks and file/directory type conflicts
  at required paths also fail without deleting the conflicting data.
- Every scan checks destination files, even after a successful transfer. Missing
  files and source changes are repaired without requiring the finished marker to change.
- The fast default compares exact file size, modification time and permission bits.
  A successful transfer also records the actual destination metadata, allowing
  unchanged files on coarse timestamp filesystems to be skipped without hiding
  later source changes.
  `-verify` compares SHA-256 content, detects same-size/same-time corruption,
  and reads new copies back before replacement. It reads more data from both disks.
- Files use exclusive, randomly named temporary files in the destination directory.
  A rename replaces the old file only after the new copy is complete. Failed or
  cancelled copies retain the old file. Interrupted copies are retried incrementally.
- Source file identity, size and modification time are checked during copying;
  the complete source tree and its locks are rechecked before pruning and completion.
- A destination-wide OS lock prevents two instances writing the same root. It is
  released on exit/crash. The persistent `.hdd-bridge.lock` file must not be removed
  while an instance is running.
- `.hdd-bridge*` names are reserved in source folders. Recognizable abandoned
  `.hdd-bridge-<32 hex digits>.tmp` files are cleaned on the next eligible sync.
  Other `.tmp` files are ordinary data.

## Completion, deletion and durability

`.hdd-bridge.done` is an atomically replaced JSON record of the last successful
folder transfer. It records source path, tree token, time and verification mode.
It is invalidated before transfer writes and is never used to bypass subsequent
file checks. Old text-format completion records are upgraded automatically.
Consumers that need a completed archive should use this record; a copied
`finished*` marker alone does not guarantee the whole destination is ready.
Transfers replace individual files, not the whole directory as one transaction.

**Destination deletion is now opt-in.** `-delete` removes extra paths only inside
eligible simulation folders, after copying and rechecking the source. It never
removes archive folders just because the source folder disappeared. Without it,
extra archive data is preserved. Source data is never deleted in either mode.

`-fsync` flushes file data and completion records. On Unix it also flushes parent
directories, including newly created directories. Windows directory-entry
flushing and atomic rename guarantees depend on the filesystem; device caches
also determine power-loss durability. Keep the destination dedicated to this
tool while it runs, and do not introduce mounts/bind mounts inside either tree.
Writers must create `lock*` before changing an already finished simulation and
remove it only after all files and finished markers have been closed. Metadata
checks cannot detect a writer that edits bytes while deliberately preserving all
metadata; `-verify` is recommended for archive verification.

Defaults: scan delay 1 minute after each scan, marker delay 5 minutes, buffer
4 MiB, no pruning, no read-back verification, no fsync. `-buffer-mib` accepts
1..256. SIGINT (and Unix SIGTERM) cancels between file I/O operations and cleans
current temporary files; an OS call blocked by a failed disk may still delay exit.

`-once` returns 0 when the scan succeeds (including no eligible folders), 1 when
any folder or disk fails, and 2 for invalid arguments. Watch mode logs failures
and retries on later scans, continuing with other folders. Cancellation exits 130.

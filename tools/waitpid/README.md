# waitpid

Wait for an already running PID to exit, then run a follow-up command once.
Requires **Go 1.21+**. macOS and Windows use `golang.org/x/sys` for native process control.

## Build and usage

From the repository root:

```sh
(cd tools/waitpid && go build .)
tools/waitpid/waitpid 12345 python -m experiments.paper.run --help
tools/waitpid/waitpid -poll 1s -timeout 12h 12345 ./next-run.sh --fast
tools/waitpid/waitpid -shell 12345 'echo done && ./next-run.sh'
```

On Windows, build in `tools/waitpid` and use `waitpid.exe`. Shell mode uses
`cmd.exe /D /S /C`; Unix uses `sh -c`, independently of the interactive shell.
Pass one quoted command string with `-shell`. Direct mode preserves each argument
exactly and inherits the current working directory, environment and standard I/O.

Options must precede the PID. Arguments after the PID belong to the follow-up
command. A PID already absent starts the command immediately. The target's exit
status is unavailable when attaching to an unrelated process, so the command
runs after either success or failure of that process. This monitors one process,
not all of its descendants: select the simulation's coordinating process if its
workers also need to finish.

## Monitoring and cancellation

- Windows opens one process handle with `SYNCHRONIZE` access and keeps it until
  monitoring ends. PID reuse cannot redirect an attached handle.
- Linux uses `/proc/<pid>/stat` start ticks; macOS uses the native process start
  time with microsecond precision. A changed identity means the original process
  has exited. Unix zombies count as exited without waiting for the parent to reap.
- BSD uses bounded, locale-independent `ps` probes with second-resolution start
  times; same-second PID reuse remains a limitation on that fallback.
- No tool can recover the original identity if the PID was already reused before
  attachment. Attach promptly and check the PID you provide.
- Missing permissions, unreadable process metadata or failed probes abort
  monitoring and do not run the command. Invalid/overflowing PIDs and waiting on
  the wrapper's own PID are rejected.
- `-timeout` covers attachment and monitoring only, independently of `-poll`.
  It does not limit the runtime of the follow-up command. Default: no timeout,
  500 ms polling. The command never starts after a monitoring timeout/cancellation.
- SIGINT and Unix SIGTERM cancel waiting. After the follow-up starts, Unix forwards
  the signal to its separate process group, allows two seconds for cleanup, then
  kills remaining group members. Commands that detach into another process group
  are outside that group. Windows creates the child suspended, assigns it to a
  Job Object, then resumes its primary thread. Cancellation terminates the job,
  including descendants, without a race before job assignment. A wrapper crash
  also closes the job and terminates its processes. If job setup fails (for
  example, an incompatible enclosing job), the child is killed before running.
  Normal command completion releases the job so intentional background children
  can continue. Windows cancellation uses termination rather than Unix signals.

Use the compiled executable for scheduling. `go run` adds an outer Go process,
which changes signal handling and reports a nonzero program exit as its own
failure rather than preserving the program's numeric exit code. For a scheduled
command requiring interactive terminal job control, run it separately instead
of under this wrapper's Unix process group.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Target exited/was absent and follow-up succeeded |
| 1 | Monitoring or another execution error |
| 2 | Invalid arguments |
| 124 | Monitoring timed out |
| 126 | Follow-up could not execute due to permissions |
| 127 | Follow-up executable was not found |
| 130 / 143 | Cancelled by SIGINT / Unix SIGTERM |
| Other | Follow-up's exit status; Unix signal termination uses 128 + signal |

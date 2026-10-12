# Workflow tools

These are independent Go command modules. Build from each tool's directory;
the Python project does not install them. macOS, Linux and Windows are supported;
BSD variants retain a polling implementation but are not runtime-tested here.

| Tool | Purpose | Go version |
| --- | --- | --- |
| [hdd-bridge](hdd-bridge/README.md) | Copy completed simulation folders from SSD to HDD | 1.25+ |
| [waitpid](waitpid/README.md) | Attach a follow-up command to an already running PID | 1.21+ |

From the repository root:

```sh
(cd tools/hdd-bridge && go build .)
(cd tools/waitpid && go build .)

tools/hdd-bridge/hdd-bridge -src /ssd/sim-cache -dst /hdd/sim-archive
tools/waitpid/waitpid -timeout 12h 12345 python -m experiments.paper.run --help
```

Windows executables have an `.exe` suffix. Windows users can run `go build .`
inside each directory using PowerShell or cmd.exe. Compiled binaries are ignored.
Use compiled `waitpid` for scheduling: `go run` wraps the program and does not
preserve its exact exit status or signal behavior.

Checks:

```sh
(cd tools/hdd-bridge && go test -race ./... && go vet ./...)
(cd tools/waitpid && go test -race ./... && go vet ./...)
```

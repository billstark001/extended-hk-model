package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const (
	monitorErrorExitCode = 1
	timeoutExitCode      = 124
)

type processMonitor interface {
	alive(context.Context) (bool, error)
	close() error
}

func main() { os.Exit(run()) }

func run() int {
	poll := flag.Duration("poll", 500*time.Millisecond, "Polling interval.")
	timeout := flag.Duration("timeout", 0, "Maximum monitoring time; 0 waits forever.")
	shell := flag.Bool("shell", false, "Run exactly one command string through sh or cmd.exe.")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <pid> <command> [args...]\n       %s -shell [options] <pid> \"<shell command>\"\n\nOptions precede the PID. A PID already absent runs the command immediately.\n", os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	args := flag.Args()
	if len(args) < 2 {
		flag.Usage()
		return 2
	}
	pid, err := parsePID(args[0])
	if err != nil {
		logf("%v", err)
		return 2
	}
	if pid == os.Getpid() {
		logf("Refusing to wait on the current process")
		return 2
	}
	if *poll <= 0 || *timeout < 0 {
		logf("Require positive -poll and nonnegative -timeout")
		return 2
	}
	command := args[1:]
	if command[0] == "" || (*shell && (len(command) != 1 || strings.TrimSpace(command[0]) == "")) {
		logf("Require a nonempty command; -shell expects exactly one string")
		return 2
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, shutdownSignals()...)
	defer signal.Stop(signals)
	var interrupted atomic.Int32
	go func() {
		select {
		case sig := <-signals:
			interrupted.Store(int32(signalCode(sig)))
			cancel()
		case <-ctx.Done():
		}
	}()
	cancelCode := func() int {
		if code := interrupted.Load(); code != 0 {
			return int(code)
		}
		return 130
	}
	waitCtx := ctx
	var endWait context.CancelFunc
	if *timeout > 0 {
		waitCtx, endWait = context.WithTimeout(ctx, *timeout)
		defer endWait()
	}
	logf("Watching PID %d; next command: %s", pid, formatCommand(command))
	monitor, err := openProcessMonitor(waitCtx, pid)
	if err != nil {
		if waitCtx.Err() != nil {
			return waitExitCode(waitCtx, cancelCode)
		}
		logf("Cannot attach to PID %d: %v", pid, err)
		return monitorErrorExitCode
	}
	defer monitor.close()
	if code := waitForProcess(waitCtx, monitor, *poll, cancelCode); code != 0 {
		return code
	}
	if ctx.Err() != nil {
		return cancelCode()
	}
	// The timeout applies only to monitoring; the follow-up has its own lifetime.
	if endWait != nil {
		endWait()
	}
	logf("Running: %s", formatCommand(command))
	if *shell {
		if runtime.GOOS == "windows" {
			return runCommand(ctx, "cmd.exe", []string{"/D", "/S", "/C", command[0]}, cancelCode)
		}
		return runCommand(ctx, "sh", []string{"-c", command[0]}, cancelCode)
	}
	return runCommand(ctx, command[0], command[1:], cancelCode)
}

func parsePID(raw string) (int, error) {
	pid, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || pid == 0 || pid > maxPID || pid > uint64(^uint(0)>>1) {
		return 0, fmt.Errorf("invalid PID: %q", raw)
	}
	return int(pid), nil
}

func waitExitCode(ctx context.Context, cancelCode func() int) int {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		logf("Monitoring timeout")
		return timeoutExitCode
	}
	return cancelCode()
}

func waitForProcess(ctx context.Context, monitor processMonitor, interval time.Duration, cancelCode func() int) int {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		if ctx.Err() != nil {
			return waitExitCode(ctx, cancelCode)
		}
		alive, err := monitor.alive(ctx)
		if ctx.Err() != nil {
			return waitExitCode(ctx, cancelCode)
		}
		if err != nil {
			logf("Monitoring failed: %v", err)
			return monitorErrorExitCode
		}
		if !alive {
			logf("Target process exited (or was already absent)")
			return 0
		}
		select {
		case <-ctx.Done():
			return waitExitCode(ctx, cancelCode)
		case <-ticker.C:
		}
	}
}

type commandControl interface {
	stop(int)
	kill()
	close()
}

func runCommand(ctx context.Context, name string, args []string, cancelCode func() int) int {
	if ctx.Err() != nil {
		return cancelCode()
	}
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	prepareCommand(cmd)
	if err := cmd.Start(); err != nil {
		if ctx.Err() != nil {
			return cancelCode()
		}
		logf("Cannot start %q: %v", name, err)
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
			return 127
		}
		if errors.Is(err, os.ErrPermission) {
			return 126
		}
		return 1
	}
	control, err := attachCommand(ctx, cmd)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		if ctx.Err() != nil {
			return cancelCode()
		}
		logf("Cannot manage follow-up process: %v", err)
		return 1
	}
	defer control.close()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if ctx.Err() != nil {
			return cancelCode()
		}
		return commandExitCode(err)
	case <-ctx.Done():
		control.stop(cancelCode())
		grace := time.NewTimer(2 * time.Second)
		defer grace.Stop()
		select {
		case <-done:
		case <-grace.C:
			control.kill()
			<-done
		}
		// Kill remaining members even if the group leader exited promptly.
		control.kill()
		return cancelCode()
	}
}

func commandExitCode(err error) int {
	if err == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return processExitCode(exitErr.ProcessState)
	}
	logf("Command wait failed: %v", err)
	return 1
}

func formatCommand(args []string) string {
	out := make([]string, len(args))
	for i, arg := range args {
		if arg == "" || strings.ContainsAny(arg, " \t\n\r\"'`$&|;<>*?()[]{}") {
			out[i] = strconv.Quote(arg)
		} else {
			out[i] = arg
		}
	}
	return strings.Join(out, " ")
}

func logf(format string, args ...any) { fmt.Fprintf(os.Stderr, "[waitpid] "+format+"\n", args...) }

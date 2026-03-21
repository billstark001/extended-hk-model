package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	monitorErrorExitCode = 1
	timeoutExitCode      = 124
)

func main() {
	var (
		pollInterval = flag.Duration("poll", 500*time.Millisecond, "Polling interval.")
		maxWait      = flag.Duration("timeout", 0, "Maximum wait time. Use 0 to wait forever.")
		useShell     = flag.Bool("shell", false, "Run a single command string through the system shell.")
	)
	flag.Usage = func() {
		prog := os.Args[0]
		fmt.Fprintf(os.Stderr, "waitpid waits for a PID to exit and then runs a command.\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options] <pid> <command> [args...]\n", prog)
		fmt.Fprintf(os.Stderr, "  %s -shell [options] <pid> \"<shell command>\"\n\n", prog)
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s 12345 python main.py\n", prog)
		fmt.Fprintf(os.Stderr, "  %s -poll 1s 12345 ./build.sh --fast\n", prog)
		fmt.Fprintf(os.Stderr, "  %s -shell -timeout 60s 12345 \"echo done && dir\"\n", prog)
	}
	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		flag.Usage()
		os.Exit(2)
	}

	pid, err := strconv.Atoi(args[0])
	if err != nil || pid <= 0 {
		fmt.Fprintf(os.Stderr, "[waitpid] invalid PID: %q\n", args[0])
		os.Exit(2)
	}
	if pid == os.Getpid() {
		fmt.Fprintln(os.Stderr, "[waitpid] refusing to wait on the current process")
		os.Exit(2)
	}
	if *pollInterval <= 0 {
		fmt.Fprintln(os.Stderr, "[waitpid] -poll must be greater than 0")
		os.Exit(2)
	}
	if *maxWait < 0 {
		fmt.Fprintln(os.Stderr, "[waitpid] -timeout cannot be negative")
		os.Exit(2)
	}

	commandArgs := args[1:]
	if *useShell && len(commandArgs) != 1 {
		fmt.Fprintln(os.Stderr, "[waitpid] -shell expects exactly one command string")
		os.Exit(2)
	}

	logf("Watching PID %d and waiting to run: %s", pid, formatCommand(commandArgs))

	exitCode := waitForPID(pid, *pollInterval, *maxWait)
	if exitCode != 0 {
		os.Exit(exitCode)
	}

	if *useShell {
		logf("Running shell command: %s", commandArgs[0])
		os.Exit(runShell(commandArgs[0]))
	}

	logf("Running command: %s", formatCommand(commandArgs))
	os.Exit(runCommand(commandArgs[0], commandArgs[1:]))
}

// waitForPID blocks until the target process exits or the timeout expires.
func waitForPID(pid int, interval, maxWait time.Duration) int {
	alive, err := isProcessAlive(pid)
	if err != nil {
		logf("Initial probe failed: %v", err)
	} else if !alive {
		logf("PID %d is not running", pid)
		return 0
	}

	msg := fmt.Sprintf("PID %d is running; polling every %s", pid, interval)
	if maxWait > 0 {
		msg += fmt.Sprintf(" with timeout %s", maxWait)
	}
	logf(msg)

	var deadline time.Time
	if maxWait > 0 {
		deadline = time.Now().Add(maxWait)
	}

	const maxConsecutiveErrors = 5
	consecutiveErrors := 0

	for {
		time.Sleep(interval)

		if !deadline.IsZero() && time.Now().After(deadline) {
			fmt.Fprintf(os.Stderr, "[waitpid] timeout: PID %d did not exit within %s\n", pid, maxWait)
			return timeoutExitCode
		}

		alive, err = isProcessAlive(pid)
		if err != nil {
			consecutiveErrors++
			fmt.Fprintf(os.Stderr, "[waitpid] probe error (%d/%d): %v\n",
				consecutiveErrors, maxConsecutiveErrors, err)
			if consecutiveErrors >= maxConsecutiveErrors {
				fmt.Fprintf(os.Stderr, "[waitpid] aborting after too many probe errors for PID %d\n", pid)
				return monitorErrorExitCode
			}
			continue
		}
		consecutiveErrors = 0

		if !alive {
			logf("PID %d exited", pid)
			return 0
		}
	}
}

// runShell executes a command string through the system shell.
func runShell(cmdStr string) int {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", cmdStr)
	} else {
		cmd = exec.Command("sh", "-c", cmdStr)
	}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "[waitpid] failed to run shell command: %v\n", err)
		return 1
	}
	return 0
}

func runCommand(name string, args []string) int {
	cmd := exec.Command(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}

		var execErr *exec.Error
		if errors.As(err, &execErr) {
			fmt.Fprintf(os.Stderr, "[waitpid] executable lookup failed for %q: %v\n", name, err)
			return 127
		}

		fmt.Fprintf(os.Stderr, "[waitpid] failed to run %q: %v\n", name, err)
		return 1
	}
	return 0
}

func formatCommand(args []string) string {
	formatted := make([]string, len(args))
	for i, arg := range args {
		if arg == "" || strings.ContainsAny(arg, " \t\n\r\"'`$&|;<>*?()[]{}") {
			formatted[i] = strconv.Quote(arg)
			continue
		}
		formatted[i] = arg
	}
	return strings.Join(formatted, " ")
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[waitpid] "+format+"\n", args...)
}

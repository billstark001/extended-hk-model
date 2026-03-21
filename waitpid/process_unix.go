//go:build !windows

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

// isProcessAlive checks whether a PID is alive on Unix-like systems.
//
// PID reuse is an inherent limitation: after a process exits, a new process may
// reuse the same PID. The window is usually small enough to accept here.
func isProcessAlive(pid int) (bool, error) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		// On Unix, FindProcess should succeed for a numeric PID. Keep the guard.
		return false, nil
	}

	// Signal(0) does not deliver a signal; it only probes process existence.
	err = proc.Signal(syscall.Signal(0))
	switch {
	case err == nil:
		// The process exists, but it may already be a zombie.
		return !isZombie(pid), nil

	case errors.Is(err, os.ErrProcessDone), errors.Is(err, syscall.ESRCH):
		// The process no longer exists.
		return false, nil

	case errors.Is(err, syscall.EPERM):
		// Permission is denied, but the process still exists.
		return !isZombie(pid), nil

	default:
		return false, fmt.Errorf("signal(pid=%d, 0): %w", pid, err)
	}
}

// isZombie reports whether the process already exited but has not been reaped.
// Linux reads /proc/<pid>/stat; macOS and BSD variants use ps.
func isZombie(pid int) bool {
	switch runtime.GOOS {
	case "linux":
		data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			return false
		}
		// Format: "<pid> (<comm>) <state> ..."
		// comm may contain spaces or parentheses, so use the last ')'.
		s := string(data)
		i := strings.LastIndex(s, ")")
		if i < 0 || i+2 >= len(s) {
			return false
		}
		return s[i+2] == 'Z'

	case "darwin", "freebsd", "openbsd", "netbsd", "dragonfly":
		out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "state=").Output()
		if err != nil {
			return false
		}
		// States starting with 'Z' indicate a zombie (for example "Z" or "Z+").
		return strings.HasPrefix(strings.TrimSpace(string(out)), "Z")
	}
	return false
}

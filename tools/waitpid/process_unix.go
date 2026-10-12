//go:build freebsd || openbsd || netbsd || dragonfly

package main

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const maxPID = 1<<31 - 1

type unixMonitor struct {
	pid   int
	start string
	done  bool
}

func openProcessMonitor(ctx context.Context, pid int) (processMonitor, error) {
	m := &unixMonitor{pid: pid}
	_, err := m.alive(ctx)
	return m, err
}

func (m *unixMonitor) close() error { return nil }

func processExists(pid int) (bool, error) {
	err := syscall.Kill(pid, 0)
	if errors.Is(err, syscall.ESRCH) {
		return false, nil
	}
	if err == nil || errors.Is(err, syscall.EPERM) {
		return true, nil
	}
	return false, err
}

func (m *unixMonitor) alive(ctx context.Context) (bool, error) {
	if m.done {
		return false, nil
	}
	exists, err := processExists(m.pid)
	if err != nil || !exists {
		m.done = err == nil
		return false, err
	}
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(probeCtx, "ps", "-p", strconv.Itoa(m.pid), "-o", "lstart=", "-o", "stat=")
	// The ps date fields must not depend on the caller's locale.
	cmd.Env = append(cmd.Environ(), "LC_ALL=C")
	out, err := cmd.Output()
	if err != nil {
		if probeCtx.Err() != nil {
			return false, probeCtx.Err()
		}
		if exists, probeErr := processExists(m.pid); probeErr != nil || !exists {
			m.done = probeErr == nil
			return false, probeErr
		}
		return false, fmt.Errorf("ps failed for PID %d: %w", m.pid, err)
	}
	fields := strings.Fields(string(out))
	if len(fields) != 6 {
		return false, fmt.Errorf("unexpected ps output for PID %d: %q", m.pid, out)
	}
	start, state := strings.Join(fields[:5], " "), fields[5]
	if m.start == "" {
		m.start = start
	}
	m.done = start != m.start || strings.HasPrefix(state, "Z") || strings.HasPrefix(state, "X")
	return !m.done, nil
}

//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

const maxPID = 1<<31 - 1

type linuxMonitor struct {
	pid   int
	start string
	done  bool
}

func openProcessMonitor(ctx context.Context, pid int) (processMonitor, error) {
	m := &linuxMonitor{pid: pid}
	_, err := m.alive(ctx)
	return m, err
}

func (m *linuxMonitor) close() error { return nil }

func (m *linuxMonitor) alive(ctx context.Context) (bool, error) {
	if m.done {
		return false, nil
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", m.pid))
	if errors.Is(err, os.ErrNotExist) {
		if err := confirmProcessGone(m.pid); err != nil {
			return false, err
		}
		m.done = true
		return false, nil
	}
	if err != nil {
		return false, err
	}
	state, start, err := parseProcStat(string(data), m.pid)
	if err != nil {
		return false, err
	}
	if m.start == "" {
		m.start = start
	}
	m.done = start != m.start || state == "Z" || state == "X" || state == "x"
	return !m.done, nil
}

func parseProcStat(data string, pid int) (string, string, error) {
	left, right := strings.Index(data, "("), strings.LastIndex(data, ")")
	if left < 1 || right < left || strings.TrimSpace(data[:left]) != strconv.Itoa(pid) {
		return "", "", fmt.Errorf("invalid /proc/%d/stat header", pid)
	}
	fields := strings.Fields(data[right+1:])
	// Field 3 is state. Field 22 is process start time in clock ticks.
	if len(fields) <= 19 || len(fields[0]) != 1 {
		return "", "", fmt.Errorf("truncated /proc/%d/stat", pid)
	}
	if _, err := strconv.ParseUint(fields[19], 10, 64); err != nil {
		return "", "", fmt.Errorf("invalid process start time: %w", err)
	}
	return fields[0], fields[19], nil
}

// hidepid mounts or an unavailable /proc can hide a live process. Missing
// metadata alone must never trigger the follow-up command.
func confirmProcessGone(pid int) error {
	err := syscall.Kill(pid, 0)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	if err == nil || errors.Is(err, syscall.EPERM) {
		return fmt.Errorf("PID %d still exists but /proc metadata is unavailable", pid)
	}
	return err
}

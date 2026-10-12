//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"

	"golang.org/x/sys/unix"
)

const maxPID = 1<<31 - 1

type darwinMonitor struct {
	pid            int
	start          unix.Timeval
	attached, done bool
}

func openProcessMonitor(ctx context.Context, pid int) (processMonitor, error) {
	m := &darwinMonitor{pid: pid}
	_, err := m.alive(ctx)
	return m, err
}

func (m *darwinMonitor) close() error { return nil }

func (m *darwinMonitor) alive(ctx context.Context) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if m.done {
		return false, nil
	}
	proc, err := unix.SysctlKinfoProc("kern.proc.pid", m.pid)
	if err != nil {
		// An absent PID produces an empty sysctl result (EIO in x/sys's typed
		// wrapper). Confirm absence, since other EIO failures must fail closed.
		if errors.Is(err, unix.EIO) || errors.Is(err, unix.ESRCH) {
			probeErr := unix.Kill(m.pid, 0)
			if errors.Is(probeErr, unix.ESRCH) {
				m.done = true
				return false, nil
			}
		}
		return false, fmt.Errorf("sysctl PID %d: %w", m.pid, err)
	}
	if int(proc.Proc.P_pid) != m.pid {
		return false, errors.New("sysctl returned an unexpected PID")
	}
	if !m.attached {
		m.start, m.attached = proc.Proc.P_starttime, true
	}
	// SZOMB is 5 in Darwin's sys/proc.h. Native start time includes microseconds.
	m.done = proc.Proc.P_starttime != m.start || proc.Proc.P_stat == 5
	return !m.done, nil
}

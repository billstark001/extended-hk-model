//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"syscall"
)

const maxPID = 1<<32 - 1
const errorInvalidParameter syscall.Errno = 87

type windowsMonitor struct{ handle syscall.Handle }

func openProcessMonitor(ctx context.Context, pid int) (processMonitor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Keep this handle for the entire wait: PID reuse cannot change its identity.
	// SYNCHRONIZE is sufficient; unnecessary query rights can deny valid waits.
	handle, err := syscall.OpenProcess(syscall.SYNCHRONIZE, false, uint32(pid))
	if errors.Is(err, errorInvalidParameter) {
		return &windowsMonitor{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("OpenProcess(%d): %w", pid, err)
	}
	return &windowsMonitor{handle: handle}, nil
}

func (m *windowsMonitor) close() error {
	if m.handle == 0 {
		return nil
	}
	return syscall.CloseHandle(m.handle)
}

func (m *windowsMonitor) alive(ctx context.Context) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if m.handle == 0 {
		return false, nil
	}
	result, err := syscall.WaitForSingleObject(m.handle, 0)
	if err != nil {
		return false, err
	}
	switch result {
	case syscall.WAIT_OBJECT_0:
		return false, nil
	case syscall.WAIT_TIMEOUT:
		return true, nil
	default:
		return false, fmt.Errorf("unexpected wait result: 0x%x", result)
	}
}

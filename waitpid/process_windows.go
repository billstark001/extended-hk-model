//go:build windows

package main

import (
	"fmt"
	"syscall"
)

const (
	errorInvalidParameter syscall.Errno = 87

	// PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
	// Combine both flags to stay compatible from XP through Windows 11.
	processQueryAccess uint32 = 0x0400 | 0x1000
)

// isProcessAlive checks whether a PID is still alive on Windows.
//
// Strategy: OpenProcess -> WaitForSingleObject(handle, 0) in non-blocking mode.
//   - WAIT_OBJECT_0 (0)   means the process already exited.
//   - WAIT_TIMEOUT  (258) means the process is still running.
//
// Windows handle-based zombies are already covered because WaitForSingleObject
// reports WAIT_OBJECT_0 after process termination.
func isProcessAlive(pid int) (bool, error) {
	handle, err := syscall.OpenProcess(
		syscall.SYNCHRONIZE|processQueryAccess,
		false,
		uint32(pid),
	)
	if err != nil {
		switch err {
		case syscall.ERROR_ACCESS_DENIED:
			// Protected system processes may deny access while still existing.
			return true, nil
		case errorInvalidParameter:
			// The PID does not exist.
			return false, nil
		default:
			// Propagate unknown errors so the caller can decide how to fail.
			return false, fmt.Errorf("OpenProcess(pid=%d): %w", pid, err)
		}
	}
	defer syscall.CloseHandle(handle)

	// Non-blocking wait: return the process signal state immediately.
	result, err := syscall.WaitForSingleObject(handle, 0)
	if err != nil {
		return false, fmt.Errorf("WaitForSingleObject(pid=%d): %w", pid, err)
	}

	switch result {
	case 0: // WAIT_OBJECT_0: the process has terminated.
		return false, nil
	case syscall.WAIT_TIMEOUT: // 0x102: the process is still running.
		return true, nil
	default:
		return false, fmt.Errorf("WaitForSingleObject(pid=%d): unexpected return value 0x%08x", pid, result)
	}
}

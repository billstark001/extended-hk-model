//go:build windows

package main

import (
	"os"
	"syscall"
	"unsafe"
)

func shutdownSignals() []os.Signal { return []os.Signal{os.Interrupt} }

var lockFileEx = syscall.NewLazyDLL("kernel32.dll").NewProc("LockFileEx")

func lockDestination(file *os.File) error {
	var overlapped syscall.Overlapped
	// LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY. Closing releases the
	// byte-range lock, including after a crash. The lock file is never unlinked.
	ok, _, err := lockFileEx.Call(file.Fd(), 3, 0, 1, 0, uintptr(unsafe.Pointer(&overlapped)))
	if ok == 0 {
		return err
	}
	return nil
}

// Windows does not support fsync on Go directory handles. File data and state
// are flushed; namespace durability remains filesystem-dependent.
func syncDirectory(root *os.Root, name string) error { return nil }

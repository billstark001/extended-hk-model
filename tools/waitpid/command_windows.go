//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func shutdownSignals() []os.Signal { return []os.Signal{os.Interrupt} }
func signalCode(sig os.Signal) int { return 130 }
func prepareCommand(cmd *exec.Cmd) {
	// No user code can run/spawn descendants before job assignment.
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_SUSPENDED}
}
func processExitCode(state *os.ProcessState) int { return state.ExitCode() }

type windowsCommandControl struct {
	job        windows.Handle
	cmd        *exec.Cmd
	terminated bool
}

func attachCommand(ctx context.Context, cmd *exec.Cmd) (_ commandControl, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			windows.CloseHandle(job)
		}
	}()
	limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err = windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits))); err != nil {
		return nil, err
	}
	process, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(process)
	if err = windows.AssignProcessToJobObject(job, process); err != nil {
		return nil, fmt.Errorf("assign process to job: %w", err)
	}
	if err = resumePrimaryThread(ctx, uint32(cmd.Process.Pid)); err != nil {
		return nil, err
	}
	return &windowsCommandControl{job: job, cmd: cmd}, nil
}

// os/exec closes CreateProcess's thread handle. Reopen the suspended primary
// thread using documented Toolhelp APIs, rather than an undocumented native
// process-resume syscall. A suspended new process has only its primary thread.
func resumePrimaryThread(ctx context.Context, pid uint32) error {
	snapshot, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(snapshot)
	entry := windows.ThreadEntry32{Size: uint32(unsafe.Sizeof(windows.ThreadEntry32{}))}
	err = windows.Thread32First(snapshot, &entry)
	for err == nil {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.OwnerProcessID == pid {
			thread, err := windows.OpenThread(windows.THREAD_SUSPEND_RESUME, false, entry.ThreadID)
			if err != nil {
				return err
			}
			defer windows.CloseHandle(thread)
			if err := ctx.Err(); err != nil {
				return err
			}
			previous, err := windows.ResumeThread(thread)
			if err != nil {
				return err
			}
			if previous != 1 {
				return fmt.Errorf("unexpected primary-thread suspend count: %d", previous)
			}
			return nil
		}
		entry.Size = uint32(unsafe.Sizeof(entry))
		err = windows.Thread32Next(snapshot, &entry)
	}
	if errors.Is(err, windows.ERROR_NO_MORE_FILES) {
		return fmt.Errorf("primary thread not found for PID %d", pid)
	}
	return err
}

func (c *windowsCommandControl) stop(code int) {
	c.terminated = true
	if err := windows.TerminateJobObject(c.job, uint32(code)); err != nil {
		logf("Cannot terminate follow-up job: %v", err)
		_ = c.cmd.Process.Kill()
	}
}
func (c *windowsCommandControl) kill() { c.stop(137) }
func (c *windowsCommandControl) close() {
	if !c.terminated {
		// Preserve normal shell/background-command semantics after a normal exit.
		limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
		if _, err := windows.SetInformationJobObject(c.job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits))); err != nil {
			logf("Cannot release follow-up job limits: %v", err)
		}
	}
	_ = windows.CloseHandle(c.job)
}

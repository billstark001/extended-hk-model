//go:build darwin || linux || freebsd || openbsd || netbsd || dragonfly

package main

import (
	"context"
	"os"
	"os/exec"
	"syscall"
)

func shutdownSignals() []os.Signal { return []os.Signal{os.Interrupt, syscall.SIGTERM} }
func signalCode(sig os.Signal) int { return 128 + int(sig.(syscall.Signal)) }
func prepareCommand(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

type unixCommandControl struct{ pid int }

func attachCommand(ctx context.Context, cmd *exec.Cmd) (commandControl, error) {
	if err := ctx.Err(); err != nil {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		return nil, err
	}
	return &unixCommandControl{cmd.Process.Pid}, nil
}
func (c *unixCommandControl) stop(code int) { _ = syscall.Kill(-c.pid, syscall.Signal(code-128)) }
func (c *unixCommandControl) kill()         { _ = syscall.Kill(-c.pid, syscall.SIGKILL) }
func (c *unixCommandControl) close()        {}
func processExitCode(state *os.ProcessState) int {
	if status, ok := state.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		return 128 + int(status.Signal())
	}
	return state.ExitCode()
}

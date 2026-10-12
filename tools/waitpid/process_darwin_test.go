//go:build darwin

package main

import (
	"context"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

func TestPIDReuseIsExit(t *testing.T) {
	m := &darwinMonitor{pid: os.Getpid(), start: unix.Timeval{}, attached: true}
	alive, err := m.alive(context.Background())
	if err != nil || alive {
		t.Fatalf("reused PID is alive=%t, err=%v", alive, err)
	}
}

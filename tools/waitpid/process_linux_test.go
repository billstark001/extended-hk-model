//go:build linux

package main

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestProcStatParsingHandlesParentheses(t *testing.T) {
	fields := []string{"Z"}
	for i := 1; i < 19; i++ {
		fields = append(fields, "0")
	}
	fields = append(fields, "123456")
	state, start, err := parseProcStat("123 (worker (part) with ) spaces) "+strings.Join(fields, " "), 123)
	if err != nil || state != "Z" || start != "123456" {
		t.Fatalf("state=%q, start=%q, err=%v", state, start, err)
	}
	for _, data := range []string{"", "123 (truncated) Z", "456 (worker) " + strings.Join(fields, " ")} {
		if _, _, err := parseProcStat(data, 123); err == nil {
			t.Fatal("accepted", data)
		}
	}
}

func TestPIDReuseIsExit(t *testing.T) {
	m := &linuxMonitor{pid: os.Getpid(), start: "different-start-time"}
	alive, err := m.alive(context.Background())
	if err != nil || alive {
		t.Fatalf("reused PID is alive=%t, err=%v", alive, err)
	}
}

func TestUnavailableProcDoesNotMeanExited(t *testing.T) {
	if err := confirmProcessGone(os.Getpid()); err == nil {
		t.Fatal("missing metadata would run follow-up for a live process")
	}
}

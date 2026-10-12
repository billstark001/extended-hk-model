//go:build windows

package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestWindowsCancellationTerminatesDescendants(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	output := filepath.Join(t.TempDir(), "pid")
	done := make(chan int, 1)
	go func() { done <- runCommand(ctx, os.Args[0], helperArgs("spawn-descendant", output), cancelCode) }()
	var pid int
	deadline := time.Now().Add(5 * time.Second)
	for pid == 0 {
		if data, err := os.ReadFile(output); err == nil {
			pid, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}
		if time.Now().After(deadline) {
			t.Fatal("descendant did not start")
		}
		time.Sleep(5 * time.Millisecond)
	}
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(handle)
	cancel()
	select {
	case code := <-done:
		if code != 130 {
			t.Fatal(code)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("follow-up did not stop")
	}
	result, err := windows.WaitForSingleObject(handle, 2000)
	if err != nil || result != windows.WAIT_OBJECT_0 {
		t.Fatalf("descendant survived cancellation: result=%d, err=%v", result, err)
	}
}

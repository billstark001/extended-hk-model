package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

type fakeMonitor struct {
	probes int
	stopAt int
	err    error
}

func (m *fakeMonitor) alive(context.Context) (bool, error) {
	m.probes++
	return m.stopAt == 0 || m.probes < m.stopAt, m.err
}
func (m *fakeMonitor) close() error { return nil }
func cancelCode() int               { return 130 }

func TestWaitStopsOnExit(t *testing.T) {
	m := &fakeMonitor{stopAt: 3}
	if code := waitForProcess(context.Background(), m, time.Millisecond, cancelCode); code != 0 || m.probes != 3 {
		t.Fatalf("code=%d, probes=%d", code, m.probes)
	}
}

func TestTimeoutIndependentOfPollInterval(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	start := time.Now()
	if code := waitForProcess(ctx, &fakeMonitor{}, time.Second, cancelCode); code != timeoutExitCode {
		t.Fatal(code)
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatal("timeout overshot:", elapsed)
	}
}

func TestCancellationPreventsProbe(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m := &fakeMonitor{}
	if code := waitForProcess(ctx, m, time.Second, cancelCode); code != 130 || m.probes != 0 {
		t.Fatalf("code=%d, probes=%d", code, m.probes)
	}
}

func TestProbeErrorFailsClosed(t *testing.T) {
	m := &fakeMonitor{err: errors.New("permission denied")}
	if code := waitForProcess(context.Background(), m, time.Millisecond, cancelCode); code != 1 {
		t.Fatal(code)
	}
}

func TestPIDValidation(t *testing.T) {
	for _, value := range []string{"0", "-1", "1.2", "abc", "4294967296", "9223372036854775807"} {
		if _, err := parsePID(value); err == nil {
			t.Fatal("accepted", value)
		}
	}
	if got, err := parsePID("12345"); err != nil || got != 12345 {
		t.Fatalf("pid=%d, err=%v", got, err)
	}
}

func TestFormatCommand(t *testing.T) {
	got := formatCommand([]string{"tool", "a b", "", "literal$()"})
	if got != `tool "a b" "" "literal$()"` {
		t.Fatal(got)
	}
}

// Helper subprocesses keep command argument and exit-code tests independent of
// the installed shell, Python, or other external executables.
func TestHelperProcess(t *testing.T) {
	index := -1
	for i, arg := range os.Args {
		if arg == "--waitpid-helper" {
			index = i
			break
		}
	}
	if index < 0 {
		return
	}
	args := os.Args[index+1:]
	switch args[0] {
	case "sleep":
		time.Sleep(10 * time.Second)
	case "exit":
		code, _ := strconv.Atoi(args[1])
		os.Exit(code)
	case "record":
		if err := os.WriteFile(args[1], []byte(strings.Join(args[2:], "\x00")), 0o600); err != nil {
			os.Exit(99)
		}
	case "spawn-descendant":
		child := exec.Command(os.Args[0], helperArgs("pid-sleep", args[1])...)
		if err := child.Run(); err != nil {
			os.Exit(97)
		}
	case "pid-sleep":
		if err := os.WriteFile(args[1], []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(99)
		}
		time.Sleep(10 * time.Second)
	case "ready-sleep":
		if err := os.WriteFile(args[1], []byte("ready"), 0o600); err != nil {
			os.Exit(99)
		}
		time.Sleep(10 * time.Second)
	default:
		os.Exit(98)
	}
	os.Exit(0)
}

func helperArgs(args ...string) []string {
	return append([]string{"-test.run=^TestHelperProcess$", "--", "--waitpid-helper"}, args...)
}

func TestRunCommandPreservesArgumentsAndExitStatus(t *testing.T) {
	output := filepath.Join(t.TempDir(), "output")
	values := []string{"a b", "", "$()", "&&", "中文"}
	args := helperArgs(append([]string{"record", output}, values...)...)
	if code := runCommand(context.Background(), os.Args[0], args, cancelCode); code != 0 {
		t.Fatal(code)
	}
	data, err := os.ReadFile(output)
	if err != nil || string(data) != strings.Join(values, "\x00") {
		t.Fatalf("args=%q, err=%v", data, err)
	}
	if code := runCommand(context.Background(), os.Args[0], helperArgs("exit", "37"), cancelCode); code != 37 {
		t.Fatal(code)
	}
	if code := runCommand(context.Background(), "waitpid-nonexistent-command-3bf012", nil, cancelCode); code != 127 {
		t.Fatal(code)
	}
}

func TestCanceledCommandDoesNotRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	output := filepath.Join(t.TempDir(), "output")
	if code := runCommand(ctx, os.Args[0], helperArgs("record", output, "unexpected"), cancelCode); code != 130 {
		t.Fatal(code)
	}
	if _, err := os.Stat(output); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("command ran: %v", err)
	}
}

func TestRunningCommandCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	output := filepath.Join(t.TempDir(), "ready")
	done := make(chan int, 1)
	go func() { done <- runCommand(ctx, os.Args[0], helperArgs("ready-sleep", output), cancelCode) }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(output); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("child did not start")
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	select {
	case code := <-done:
		if code != 130 {
			t.Fatal(code)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("child did not stop on cancellation")
	}
}

func TestNativeMonitor(t *testing.T) {
	child := exec.Command(os.Args[0], helperArgs("sleep")...)
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { child.Process.Kill(); child.Wait() }()
	m, err := openProcessMonitor(context.Background(), child.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer m.close()
	alive, err := m.alive(context.Background())
	if err != nil || !alive {
		t.Fatalf("alive=%t, err=%v", alive, err)
	}
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	// Do not reap yet: Unix zombie detection must also complete the monitor.
	deadline := time.Now().Add(3 * time.Second)
	for {
		alive, err = m.alive(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !alive {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("exited/zombie process remained alive")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestAbsentPIDIsImmediateAndLatched(t *testing.T) {
	child := exec.Command(os.Args[0], helperArgs("exit", "0")...)
	if err := child.Run(); err != nil {
		t.Fatal(err)
	}
	m, err := openProcessMonitor(context.Background(), child.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer m.close()
	if code := waitForProcess(context.Background(), m, time.Second, cancelCode); code != 0 {
		t.Fatal(code)
	}
}

func Example_formatCommand() {
	fmt.Println(formatCommand([]string{"python", "script.py", "a b"}))
	// Output: python script.py "a b"
}

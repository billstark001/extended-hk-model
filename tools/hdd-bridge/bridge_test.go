package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeTestFile(t *testing.T, name, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	// Use an exactly representable, old time so minelapsed never makes tests flaky.
	stamp := time.Unix(1700000000, 0)
	if err := os.Chtimes(name, stamp, stamp); err != nil {
		t.Fatal(err)
	}
}

func newTestBridge(t *testing.T, modify func(*config)) *bridge {
	t.Helper()
	cfg := config{srcDir: t.TempDir(), dstDir: t.TempDir(), bufferSize: 32}
	if modify != nil {
		modify(&cfg)
	}
	b, err := openBridge(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.close)
	return b
}

func seedSimulation(t *testing.T, b *bridge) {
	t.Helper()
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "finished-1.msgpack"), "finished")
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "nested", "data"), "data")
	if err := os.MkdirAll(filepath.Join(b.cfg.srcDir, "run", "empty"), 0o755); err != nil {
		t.Fatal(err)
	}
}

func assertContent(t *testing.T, name, want string) {
	t.Helper()
	got, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("%s = %q; want %q", name, got, want)
	}
}

func assertNoState(t *testing.T, b *bridge) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(b.cfg.dstDir, "run", completionStateFile)); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("unexpected completion state: %v", err)
	}
}

func TestValidatePathsNeverMutatesInput(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "src")
	dst := filepath.Join(root, "dst")
	if err := os.Mkdir(src, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, dst, "keep")
	if _, _, err := validatePaths(src, dst); err == nil {
		t.Fatal("accepted a destination file")
	}
	assertContent(t, dst, "keep")
	missing := filepath.Join(root, "missing")
	if _, _, err := validatePaths(missing, src); err == nil {
		t.Fatal("accepted missing source")
	}
	if _, err := os.Stat(missing); !errors.Is(err, fs.ErrNotExist) {
		t.Fatal("created missing source")
	}
	if _, _, err := validatePaths(src, filepath.Join(src, "child")); err == nil {
		t.Fatal("accepted nested path")
	}
	if _, err := os.Stat(filepath.Join(src, "child")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatal("created nested target")
	}
	if _, _, err := validatePaths(src, src); err == nil {
		t.Fatal("accepted identical roots")
	}
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(src, alias); err == nil {
		if _, _, err := validatePaths(alias, src); err == nil {
			t.Fatal("accepted aliased roots")
		}
	}
}

func TestEligibility(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	ready, err := inspectFolder(b.src, "run", time.Minute)
	if err != nil || !ready {
		t.Fatalf("ready=%t, err=%v", ready, err)
	}
	// Even a directory or dangling symlink named lock* blocks copying.
	lock := filepath.Join(b.cfg.srcDir, "run", "lock-active")
	if err := os.Mkdir(lock, 0o755); err != nil {
		t.Fatal(err)
	}
	ready, err = inspectFolder(b.src, "run", 0)
	if err != nil || ready {
		t.Fatalf("lock not honored: %t, %v", ready, err)
	}
	if err := os.Remove(lock); err != nil {
		t.Fatal(err)
	}
	marker := filepath.Join(b.cfg.srcDir, "run", "finished-new")
	writeTestFile(t, marker, "finished")
	future := time.Now().Add(time.Hour)
	if err := os.Chtimes(marker, future, future); err != nil {
		t.Fatal(err)
	}
	ready, err = inspectFolder(b.src, "run", 0)
	if err != nil || ready {
		t.Fatalf("future marker not honored: %t, %v", ready, err)
	}
}

func TestFinishedSymlinkDoesNotQualify(t *testing.T) {
	b := newTestBridge(t, nil)
	dir := filepath.Join(b.cfg.srcDir, "run")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(b.cfg.srcDir, "real")
	writeTestFile(t, target, "finished")
	if err := os.Symlink(target, filepath.Join(dir, "finished")); err != nil {
		t.Skip(err)
	}
	ready, err := inspectFolder(b.src, "run", 0)
	if err != nil || ready {
		t.Fatalf("symlink qualifies: %t, %v", ready, err)
	}
}

func TestIncrementalCopyAndRepair(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	ctx := context.Background()
	if err := b.scanOnce(ctx); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(b.cfg.dstDir, "run", "nested", "data")
	assertContent(t, dst, "data")
	old, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.scanOnce(ctx); err != nil {
		t.Fatal(err)
	}
	now, _ := os.Stat(dst)
	if !os.SameFile(old, now) {
		t.Fatal("unchanged file copied again")
	}
	if err := os.Remove(dst); err != nil {
		t.Fatal(err)
	}
	if err := b.scanOnce(ctx); err != nil {
		t.Fatal(err)
	}
	assertContent(t, dst, "data")
	// Data changes without touching the finished marker must be detected.
	src := filepath.Join(b.cfg.srcDir, "run", "nested", "data")
	writeTestFile(t, src, "next")
	stamp := time.Unix(1700000000, 500000000)
	if err := os.Chtimes(src, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	if err := b.scanOnce(ctx); err != nil {
		t.Fatal(err)
	}
	assertContent(t, dst, "next")
	if info, err := os.Stat(filepath.Join(b.cfg.dstDir, "run", "empty")); err != nil || !info.IsDir() {
		t.Fatalf("empty directory missing: %v", err)
	}
	var state completionState
	raw, err := os.ReadFile(filepath.Join(b.cfg.dstDir, "run", completionStateFile))
	if err != nil || json.Unmarshal(raw, &state) != nil || state.Version != 1 || state.Source != b.cfg.srcDir {
		t.Fatalf("invalid state: %s, %v", raw, err)
	}
}

func TestVerifyRepairsSameMetadataCorruption(t *testing.T) {
	b := newTestBridge(t, func(cfg *config) { cfg.verify = true })
	seedSimulation(t, b)
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(b.cfg.dstDir, "run", "nested", "data")
	writeTestFile(t, dst, "evil") // Same size and mtime as source.
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertContent(t, dst, "data")
}

func TestDeleteIsOptInAndConfined(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	extra := filepath.Join(b.cfg.dstDir, "run", "extra", "keep")
	unrelated := filepath.Join(b.cfg.dstDir, "other", "keep")
	writeTestFile(t, extra, "extra")
	writeTestFile(t, unrelated, "unrelated")
	// A legitimate source filename ending in the old temp suffix survives.
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "data.hddbridge.tmp"), "legitimate")
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertContent(t, extra, "extra")
	b.cfg.deleteExtraneous = true
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(extra); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("extra not removed: %v", err)
	}
	assertContent(t, unrelated, "unrelated")
	assertContent(t, filepath.Join(b.cfg.dstDir, "run", "data.hddbridge.tmp"), "legitimate")
}

func TestSourceSymlinkFailsWithoutCompletion(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	outside := filepath.Join(t.TempDir(), "secret")
	writeTestFile(t, outside, "outside")
	if err := os.Symlink(outside, filepath.Join(b.cfg.srcDir, "run", "link")); err != nil {
		t.Skip(err)
	}
	if err := b.scanOnce(context.Background()); err == nil {
		t.Fatal("silently skipped symlink")
	}
	assertNoState(t, b)
	assertContent(t, outside, "outside")
}

func TestDestinationSymlinkCannotEscape(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	outside := t.TempDir()
	writeTestFile(t, filepath.Join(outside, "data"), "keep")
	if err := os.Mkdir(filepath.Join(b.cfg.dstDir, "run"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(b.cfg.dstDir, "run", "nested")); err != nil {
		t.Skip(err)
	}
	if err := b.scanOnce(context.Background()); err == nil {
		t.Fatal("accepted destination symlink")
	}
	assertContent(t, filepath.Join(outside, "data"), "keep")
	assertNoState(t, b)
}

func TestDestinationConflictsPreserveExistingData(t *testing.T) {
	for _, conflict := range []string{"file-as-directory", "directory-as-file"} {
		t.Run(conflict, func(t *testing.T) {
			b := newTestBridge(t, nil)
			seedSimulation(t, b)
			if conflict == "file-as-directory" {
				writeTestFile(t, filepath.Join(b.cfg.dstDir, "run", "nested"), "keep")
			} else {
				writeTestFile(t, filepath.Join(b.cfg.dstDir, "run", "nested", "data", "keep"), "keep")
			}
			if err := b.scanOnce(context.Background()); err == nil {
				t.Fatal("accepted destructive type conflict")
			}
			if conflict == "file-as-directory" {
				assertContent(t, filepath.Join(b.cfg.dstDir, "run", "nested"), "keep")
			} else {
				assertContent(t, filepath.Join(b.cfg.dstDir, "run", "nested", "data", "keep"), "keep")
			}
			assertNoState(t, b)
		})
	}
}

func TestDestinationLockReleasesOnClose(t *testing.T) {
	b := newTestBridge(t, nil)
	if second, err := openBridge(b.cfg); err == nil {
		second.close()
		t.Fatal("two bridges acquired destination")
	}
	b.lock.Close()
	b.lock = nil
	second, err := openBridge(b.cfg)
	if err != nil {
		t.Fatal("stale lock prevents restart:", err)
	}
	second.close()
}

func TestDiskRootReplacementStopsWrites(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	old := b.cfg.dstDir + "-old"
	if err := os.Rename(b.cfg.dstDir, old); err != nil {
		t.Skip(err)
	}
	t.Cleanup(func() { os.RemoveAll(old) })
	if err := os.Mkdir(b.cfg.dstDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := b.scanOnce(context.Background()); err == nil {
		t.Fatal("accepted replaced mount path")
	}
	entries, err := os.ReadDir(b.cfg.dstDir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("wrote to replacement root: %v, %v", entries, err)
	}
}

func TestChangedSourceAndNewLockPreventCompletion(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	before, err := takeSnapshot(context.Background(), b.src, "run")
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "new"), "new")
	if err := b.validateSnapshot(context.Background(), "run", before); err == nil {
		t.Fatal("missed source mutation")
	}
	before, err = takeSnapshot(context.Background(), b.src, "run")
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "lock"), "locked")
	if err := b.validateSnapshot(context.Background(), "run", before); err == nil {
		t.Fatal("missed new lock")
	}
	// A copy already in progress may finish reading, but cannot publish its file.
	source := filepath.Join("run", "nested", "data")
	writeTestFile(t, filepath.Join(b.cfg.dstDir, source), "keep")
	if err := b.copyFile(context.Background(), source, before.files[source], before.files["run"]); err == nil {
		t.Fatal("published file while source locked")
	}
	assertContent(t, filepath.Join(b.cfg.dstDir, source), "keep")
}

func TestFailedCopyKeepsOldFileAndCleansTemp(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	source := filepath.Join("run", "nested", "data")
	info, err := b.src.Stat(source)
	if err != nil {
		t.Fatal(err)
	}
	folderInfo, err := b.src.Stat("run")
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(b.cfg.dstDir, source), "keep")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := b.copyFile(ctx, source, info, folderInfo); !errors.Is(err, context.Canceled) {
		t.Fatalf("copy err = %v", err)
	}
	assertContent(t, filepath.Join(b.cfg.dstDir, source), "keep")
	entries, _ := os.ReadDir(filepath.Join(b.cfg.dstDir, "run", "nested"))
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), tempPrefix) {
			t.Fatal("temporary file leaked")
		}
	}
	// A failed rename must also preserve the original destination tree.
	if err := os.Remove(filepath.Join(b.cfg.dstDir, source)); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(b.cfg.dstDir, source, "keep"), "keep")
	if err := b.copyFile(context.Background(), source, info, folderInfo); err == nil {
		t.Fatal("rename to nonempty directory succeeded")
	}
	assertContent(t, filepath.Join(b.cfg.dstDir, source, "keep"), "keep")
}

func TestFsyncCopy(t *testing.T) {
	b := newTestBridge(t, func(cfg *config) { cfg.fsync = true })
	seedSimulation(t, b)
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestFolderErrorsDoNotBlockOtherFolders(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "a-bad", "finished"), "done")
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "a-bad", ".hdd-bridge.done"), "reserved")
	if err := b.scanOnce(context.Background()); err == nil {
		t.Fatal("error not propagated")
	}
	assertContent(t, filepath.Join(b.cfg.dstDir, "run", "nested", "data"), "data")
}

func TestCompletionInvalidatedOnFailure(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	// A new regular source file conflicts with an existing destination directory.
	writeTestFile(t, filepath.Join(b.cfg.srcDir, "run", "new"), "new")
	writeTestFile(t, filepath.Join(b.cfg.dstDir, "run", "new", "keep"), "keep")
	if err := b.scanOnce(context.Background()); err == nil {
		t.Fatal("missing failure")
	}
	assertNoState(t, b)
}

type shortWriter struct{}

func (shortWriter) Write(data []byte) (int, error) { return len(data) - 1, nil }

func TestCopyContext(t *testing.T) {
	if _, err := copyContext(context.Background(), shortWriter{}, bytes.NewBufferString("data"), make([]byte, 32)); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var output bytes.Buffer
	if _, err := copyContext(ctx, &output, bytes.NewBufferString("data"), make([]byte, 32)); !errors.Is(err, context.Canceled) || output.Len() != 0 {
		t.Fatalf("cancel = %v, output = %q", err, output.String())
	}
}

func TestCrashTempsAreCleanedWithoutDelete(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	stale := filepath.Join(b.cfg.dstDir, "run", tempPrefix+strings.Repeat("a", 32)+".tmp")
	unrelated := filepath.Join(b.cfg.dstDir, "run", "notes.tmp")
	writeTestFile(t, stale, "partial")
	writeTestFile(t, unrelated, "keep")
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("crash temp not removed: %v", err)
	}
	assertContent(t, unrelated, "keep")
}

func TestDeleteSymlinkDoesNotDeleteTarget(t *testing.T) {
	b := newTestBridge(t, func(cfg *config) { cfg.deleteExtraneous = true })
	seedSimulation(t, b)
	outside := t.TempDir()
	writeTestFile(t, filepath.Join(outside, "keep"), "keep")
	if err := os.Mkdir(filepath.Join(b.cfg.dstDir, "run"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(b.cfg.dstDir, "run", "extra")); err != nil {
		t.Skip(err)
	}
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertContent(t, filepath.Join(outside, "keep"), "keep")
}

func TestRecordedCoarseDestinationTimeSkipsOnlyUnchangedSource(t *testing.T) {
	b := newTestBridge(t, nil)
	seedSimulation(t, b)
	source := filepath.Join("run", "nested", "data")
	srcInfo, err := b.src.Lstat(source)
	if err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(b.cfg.dstDir, source)
	writeTestFile(t, target, "data")
	rounded := srcInfo.ModTime().Add(-time.Second)
	if err := os.Chtimes(target, rounded, rounded); err != nil {
		t.Fatal(err)
	}
	dstInfo, err := b.dst.Lstat(source)
	if err != nil {
		t.Fatal(err)
	}
	record := &fileRecord{signature(srcInfo), signature(dstInfo)}
	same, err := b.sameDestinationFile(context.Background(), source, srcInfo, record)
	if err != nil || !same {
		t.Fatalf("unchanged rounded destination recopied: %t, %v", same, err)
	}
	updated := srcInfo.ModTime().Add(time.Millisecond)
	if err := os.Chtimes(filepath.Join(b.cfg.srcDir, source), updated, updated); err != nil {
		t.Fatal(err)
	}
	srcInfo, err = b.src.Lstat(source)
	if err != nil {
		t.Fatal(err)
	}
	same, err = b.sameDestinationFile(context.Background(), source, srcInfo, record)
	if err != nil || same {
		t.Fatalf("source change missed: %t, %v", same, err)
	}
}

func TestVerifyPreservesChangedPermissions(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("Windows does not expose Unix permission bits")
	}
	b := newTestBridge(t, func(cfg *config) { cfg.verify = true })
	seedSimulation(t, b)
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(b.cfg.srcDir, "run", "nested", "data")
	if err := os.Chmod(source, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := b.scanOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(filepath.Join(b.cfg.dstDir, "run", "nested", "data"))
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("mode=%v, err=%v", info, err)
	}
}

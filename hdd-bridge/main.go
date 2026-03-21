package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	completionStateFile = ".hdd-bridge.done"
	tempFileSuffix      = ".hddbridge.tmp"
)

type config struct {
	srcDir           string
	dstDir           string
	interval         time.Duration
	minElapsed       time.Duration
	bufferSize       int
	deleteExtraneous bool
	fsync            bool
}

type folderState struct {
	token          string
	finishedMarker string
	finishedTime   time.Time
}

type syncStats struct {
	copiedFiles     int
	skippedFiles    int
	deletedPaths    int
	skippedSymlinks int
	copiedBytes     int64
}

func main() {
	var cfg config
	bufferMiB := flag.Int("buffer-mib", 4, "Copy buffer size in MiB.")

	flag.StringVar(&cfg.srcDir, "src", "", "Source directory on the fast disk.")
	flag.StringVar(&cfg.dstDir, "dst", "", "Destination directory on the slow disk.")
	flag.DurationVar(&cfg.interval, "interval", time.Minute, "How often to scan the source directory.")
	flag.DurationVar(&cfg.minElapsed, "minelapsed", 5*time.Minute, "Minimum age of a finished marker before copying starts.")
	flag.BoolVar(&cfg.deleteExtraneous, "delete", true, "Delete destination files that no longer exist in the source tree.")
	flag.BoolVar(&cfg.fsync, "fsync", false, "Call fsync on copied files before rename. This improves durability but reduces throughput.")
	flag.Usage = func() {
		prog := os.Args[0]
		fmt.Fprintf(os.Stderr, "hdd-bridge incrementally mirrors completed simulation folders from a fast source disk to a slower destination disk.\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s -src <ssd-dir> -dst <hdd-dir> [options]\n\n", prog)
		fmt.Fprintf(os.Stderr, "A source subdirectory becomes eligible when it contains at least one file whose name starts with 'finished', has no file starting with 'lock', and the newest finished marker is older than -minelapsed.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if cfg.srcDir == "" || cfg.dstDir == "" {
		fmt.Fprintln(os.Stderr, "[hdd-bridge] both -src and -dst are required")
		os.Exit(1)
	}
	if cfg.interval <= 0 {
		fmt.Fprintln(os.Stderr, "[hdd-bridge] -interval must be greater than 0")
		os.Exit(1)
	}
	if cfg.minElapsed < 0 {
		fmt.Fprintln(os.Stderr, "[hdd-bridge] -minelapsed cannot be negative")
		os.Exit(1)
	}
	if *bufferMiB <= 0 {
		fmt.Fprintln(os.Stderr, "[hdd-bridge] -buffer-mib must be greater than 0")
		os.Exit(1)
	}

	var err error
	cfg.srcDir, cfg.dstDir, err = validatePaths(cfg.srcDir, cfg.dstDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[hdd-bridge] invalid paths: %v\n", err)
		os.Exit(1)
	}
	cfg.bufferSize = *bufferMiB * 1024 * 1024

	bufferPool := &sync.Pool{
		New: func() any {
			buf := make([]byte, cfg.bufferSize)
			return &buf
		},
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	logf("Watching %s -> %s (interval=%s, minelapsed=%s, buffer=%d MiB, delete=%t, fsync=%t)", cfg.srcDir, cfg.dstDir, cfg.interval, cfg.minElapsed, *bufferMiB, cfg.deleteExtraneous, cfg.fsync)

	if err := scanOnce(cfg, bufferPool); err != nil {
		logf("Initial scan failed: %v", err)
	}

	ticker := time.NewTicker(cfg.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logf("Stopped")
			return
		case <-ticker.C:
			if err := scanOnce(cfg, bufferPool); err != nil {
				logf("Scan failed: %v", err)
			}
		}
	}
}

func validatePaths(srcDir, dstDir string) (string, string, error) {
	srcAbs, err := filepath.Abs(srcDir)
	if err != nil {
		return "", "", fmt.Errorf("resolve source path: %w", err)
	}
	dstAbs, err := filepath.Abs(dstDir)
	if err != nil {
		return "", "", fmt.Errorf("resolve destination path: %w", err)
	}

	srcAbs, err = evalPath(srcAbs)
	if err != nil {
		return "", "", fmt.Errorf("resolve source path: %w", err)
	}
	if err := ensureDirectory(srcAbs, 0o755); err != nil {
		return "", "", fmt.Errorf("prepare source directory: %w", err)
	}
	if err := ensureDirectory(dstAbs, 0o755); err != nil {
		return "", "", fmt.Errorf("prepare destination directory: %w", err)
	}
	dstAbs, err = evalPath(dstAbs)
	if err != nil {
		return "", "", fmt.Errorf("resolve destination path: %w", err)
	}

	if sameOrNestedPath(srcAbs, dstAbs) || sameOrNestedPath(dstAbs, srcAbs) {
		return "", "", fmt.Errorf("source and destination must be disjoint directories")
	}

	return srcAbs, dstAbs, nil
}

func scanOnce(cfg config, bufferPool *sync.Pool) error {
	entries, err := os.ReadDir(cfg.srcDir)
	if err != nil {
		return fmt.Errorf("read source directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		sourcePath := filepath.Join(cfg.srcDir, entry.Name())
		state, ready, err := inspectFolder(sourcePath, cfg.minElapsed)
		if err != nil {
			logf("Skipping %s: %v", entry.Name(), err)
			continue
		}
		if !ready {
			continue
		}

		targetPath := filepath.Join(cfg.dstDir, entry.Name())
		upToDate, err := hasCompletionToken(targetPath, state.token)
		if err != nil {
			logf("Could not read completion state for %s: %v", entry.Name(), err)
		}
		if upToDate {
			continue
		}

		logf("Syncing %s -> %s", sourcePath, targetPath)
		stats, err := syncFolder(sourcePath, targetPath, cfg, bufferPool)
		if err != nil {
			logf("Sync failed for %s: %v", entry.Name(), err)
			continue
		}
		if err := writeCompletionState(targetPath, sourcePath, state, stats); err != nil {
			logf("Failed to write completion state for %s: %v", entry.Name(), err)
			continue
		}

		logf("Completed %s: copied %d files (%s), skipped %d, deleted %d, skipped symlinks %d", entry.Name(), stats.copiedFiles, humanBytes(stats.copiedBytes), stats.skippedFiles, stats.deletedPaths, stats.skippedSymlinks)
	}

	return nil
}

func inspectFolder(folderPath string, minElapsed time.Duration) (folderState, bool, error) {
	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return folderState{}, false, err
	}

	var latestInfo fs.FileInfo
	var latestName string
	hasFinished := false

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "lock") {
			return folderState{}, false, nil
		}
		if !strings.HasPrefix(name, "finished") {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			return folderState{}, false, err
		}
		hasFinished = true
		if latestInfo == nil || info.ModTime().After(latestInfo.ModTime()) {
			latestInfo = info
			latestName = name
		}
	}

	if !hasFinished || latestInfo == nil {
		return folderState{}, false, nil
	}
	if time.Since(latestInfo.ModTime()) < minElapsed {
		return folderState{}, false, nil
	}

	state := folderState{
		token:          fmt.Sprintf("%s:%d:%d", latestName, latestInfo.ModTime().UnixNano(), latestInfo.Size()),
		finishedMarker: latestName,
		finishedTime:   latestInfo.ModTime(),
	}
	return state, true, nil
}

func hasCompletionToken(targetPath, token string) (bool, error) {
	data, err := os.ReadFile(filepath.Join(targetPath, completionStateFile))
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}

	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "token=") {
			return strings.TrimPrefix(line, "token=") == token, nil
		}
	}
	return false, nil
}

func syncFolder(src, dst string, cfg config, bufferPool *sync.Pool) (syncStats, error) {
	stats := syncStats{}
	if err := ensureDirectory(dst, 0o755); err != nil {
		return stats, err
	}

	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		relPath, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if relPath == "." {
			return nil
		}

		targetPath := filepath.Join(dst, relPath)
		if d.Type()&os.ModeSymlink != 0 {
			stats.skippedSymlinks++
			logf("Skipping symlink %s", path)
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return err
		}

		if d.IsDir() {
			return ensureDirectory(targetPath, info.Mode())
		}
		if !info.Mode().IsRegular() {
			logf("Skipping unsupported file type %s", path)
			return nil
		}

		identical, err := sameDestinationFile(info, targetPath)
		if err != nil {
			return err
		}
		if identical {
			stats.skippedFiles++
			return nil
		}

		if err := copyFileAtomic(path, targetPath, info, cfg, bufferPool); err != nil {
			return err
		}
		stats.copiedFiles++
		stats.copiedBytes += info.Size()
		return nil
	})
	if err != nil {
		return stats, err
	}

	if cfg.deleteExtraneous {
		deleted, err := deleteExtraneousPaths(src, dst)
		stats.deletedPaths += deleted
		if err != nil {
			return stats, err
		}
	}

	return stats, nil
}

func sameDestinationFile(srcInfo fs.FileInfo, dstPath string) (bool, error) {
	dstInfo, err := os.Lstat(dstPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if dstInfo.IsDir() || dstInfo.Mode()&os.ModeSymlink != 0 {
		if err := os.RemoveAll(dstPath); err != nil {
			return false, err
		}
		return false, nil
	}
	if dstInfo.Size() != srcInfo.Size() {
		return false, nil
	}
	return sameModTime(dstInfo.ModTime(), srcInfo.ModTime()), nil
}

func copyFileAtomic(srcPath, dstPath string, srcInfo fs.FileInfo, cfg config, bufferPool *sync.Pool) (err error) {
	if err := ensureDirectory(filepath.Dir(dstPath), 0o755); err != nil {
		return err
	}

	in, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer in.Close()

	tempPath := fmt.Sprintf("%s.%d%s", dstPath, os.Getpid(), tempFileSuffix)
	out, err := os.OpenFile(tempPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, srcInfo.Mode().Perm())
	if err != nil {
		return err
	}

	defer func() {
		if err != nil {
			_ = os.Remove(tempPath)
		}
	}()

	bufferPtr := bufferPool.Get().(*[]byte)
	defer bufferPool.Put(bufferPtr)

	if _, err = io.CopyBuffer(out, in, *bufferPtr); err != nil {
		_ = out.Close()
		return err
	}
	if cfg.fsync {
		if err = out.Sync(); err != nil {
			_ = out.Close()
			return err
		}
	}
	if err = out.Close(); err != nil {
		return err
	}
	if err = os.Chtimes(tempPath, srcInfo.ModTime(), srcInfo.ModTime()); err != nil {
		return err
	}
	if err = os.Remove(dstPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	if err = os.Rename(tempPath, dstPath); err != nil {
		return err
	}
	return nil
}

func deleteExtraneousPaths(src, dst string) (int, error) {
	toDelete := make([]string, 0)
	err := filepath.WalkDir(dst, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		relPath, err := filepath.Rel(dst, path)
		if err != nil {
			return err
		}
		if relPath == "." {
			return nil
		}

		base := filepath.Base(path)
		if base == completionStateFile {
			return nil
		}
		if strings.HasSuffix(base, tempFileSuffix) {
			toDelete = append(toDelete, path)
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}

		sourcePath := filepath.Join(src, relPath)
		_, err = os.Lstat(sourcePath)
		if os.IsNotExist(err) {
			toDelete = append(toDelete, path)
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		return err
	})
	if err != nil {
		return 0, err
	}

	sort.Slice(toDelete, func(i, j int) bool {
		return pathDepth(toDelete[i]) > pathDepth(toDelete[j])
	})

	removed := 0
	for _, path := range toDelete {
		if err := os.RemoveAll(path); err != nil && !os.IsNotExist(err) {
			return removed, err
		}
		removed++
	}
	return removed, nil
}

func writeCompletionState(targetPath, sourcePath string, state folderState, stats syncStats) error {
	if err := ensureDirectory(targetPath, 0o755); err != nil {
		return err
	}

	content := strings.Join([]string{
		"token=" + state.token,
		"source=" + sourcePath,
		"finished_marker=" + state.finishedMarker,
		"finished_time=" + state.finishedTime.UTC().Format(time.RFC3339Nano),
		"copied_at=" + time.Now().UTC().Format(time.RFC3339Nano),
		fmt.Sprintf("copied_files=%d", stats.copiedFiles),
		fmt.Sprintf("skipped_files=%d", stats.skippedFiles),
		fmt.Sprintf("deleted_paths=%d", stats.deletedPaths),
		fmt.Sprintf("copied_bytes=%d", stats.copiedBytes),
		fmt.Sprintf("skipped_symlinks=%d", stats.skippedSymlinks),
	}, "\n") + "\n"

	return os.WriteFile(filepath.Join(targetPath, completionStateFile), []byte(content), 0o644)
}

func ensureDirectory(path string, mode fs.FileMode) error {
	info, err := os.Lstat(path)
	if err == nil {
		if info.IsDir() {
			return nil
		}
		if err := os.RemoveAll(path); err != nil {
			return err
		}
	}
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.MkdirAll(path, mode.Perm())
}

func evalPath(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		if os.IsNotExist(err) {
			return path, nil
		}
		return "", err
	}
	return resolved, nil
}

func sameOrNestedPath(base, target string) bool {
	relPath, err := filepath.Rel(base, target)
	if err != nil {
		return false
	}
	if relPath == "." {
		return true
	}
	prefix := ".." + string(os.PathSeparator)
	return relPath != ".." && !strings.HasPrefix(relPath, prefix)
}

func sameModTime(left, right time.Time) bool {
	delta := left.Sub(right)
	if delta < 0 {
		delta = -delta
	}
	return delta < time.Second
}

func pathDepth(path string) int {
	return strings.Count(filepath.Clean(path), string(os.PathSeparator))
}

func humanBytes(size int64) string {
	if size < 1024 {
		return fmt.Sprintf("%d B", size)
	}

	units := []string{"KiB", "MiB", "GiB", "TiB"}
	value := float64(size)
	unit := "B"
	for _, nextUnit := range units {
		value /= 1024
		unit = nextUnit
		if value < 1024 {
			break
		}
	}
	return fmt.Sprintf("%.2f %s", value, unit)
}

func logf(format string, args ...any) {
	prefixArgs := append([]any{time.Now().Format(time.RFC3339)}, args...)
	fmt.Fprintf(os.Stderr, "[hdd-bridge] %s "+format+"\n", prefixArgs...)
}

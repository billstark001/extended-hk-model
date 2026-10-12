package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	completionStateFile = ".hdd-bridge.done"
	lockFile            = ".hdd-bridge.lock"
	tempPrefix          = ".hdd-bridge-"
)

type bridge struct {
	cfg              config
	src, dst         *os.Root
	srcInfo, dstInfo fs.FileInfo
	lock             *os.File
	buffer           []byte
}

type snapshot struct {
	files map[string]fs.FileInfo
	names []string
	token string
}

type syncStats struct {
	copiedFiles, skippedFiles, deletedPaths int
	copiedBytes                             int64
}

type completionState struct {
	Version  int                   `json:"version"`
	Token    string                `json:"token"`
	Source   string                `json:"source"`
	CopiedAt time.Time             `json:"copied_at"`
	Verified bool                  `json:"verified"`
	Files    map[string]fileRecord `json:"files"`
}

type fileSignature struct {
	Size    int64  `json:"size"`
	ModTime int64  `json:"mtime_ns"`
	Mode    uint32 `json:"mode"`
}

type fileRecord struct {
	Source      fileSignature `json:"source"`
	Destination fileSignature `json:"destination"`
}

func signature(info fs.FileInfo) fileSignature {
	return fileSignature{info.Size(), info.ModTime().UnixNano(), uint32(info.Mode())}
}

func validatePaths(src, dst string) (string, string, error) {
	resolve := func(path string) (string, fs.FileInfo, error) {
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", nil, err
		}
		real, err := filepath.EvalSymlinks(abs)
		if err != nil {
			return "", nil, err
		}
		info, err := os.Stat(real)
		if err != nil {
			return "", nil, err
		}
		if !info.IsDir() {
			return "", nil, fmt.Errorf("%s is not a directory", path)
		}
		return real, info, nil
	}
	src, si, err := resolve(src)
	if err != nil {
		return "", "", fmt.Errorf("source must be an existing directory: %w", err)
	}
	dst, di, err := resolve(dst)
	if err != nil {
		return "", "", fmt.Errorf("destination must be an existing directory: %w", err)
	}
	if os.SameFile(si, di) || sameOrNestedPath(src, dst) || sameOrNestedPath(dst, src) {
		return "", "", errors.New("source and destination must be disjoint directories")
	}
	return src, dst, nil
}

func sameOrNestedPath(base, target string) bool {
	rel, err := filepath.Rel(base, target)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func openBridge(cfg config) (_ *bridge, err error) {
	cfg.srcDir, cfg.dstDir, err = validatePaths(cfg.srcDir, cfg.dstDir)
	if err != nil {
		return nil, err
	}
	if cfg.bufferSize <= 0 {
		return nil, errors.New("copy buffer must be positive")
	}
	b := &bridge{cfg: cfg, buffer: make([]byte, cfg.bufferSize)}
	defer func() {
		if err != nil {
			b.close()
		}
	}()
	b.src, err = os.OpenRoot(cfg.srcDir)
	if err != nil {
		return nil, err
	}
	b.dst, err = os.OpenRoot(cfg.dstDir)
	if err != nil {
		return nil, err
	}
	b.srcInfo, err = b.src.Stat(".")
	if err != nil {
		return nil, err
	}
	b.dstInfo, err = b.dst.Stat(".")
	if err != nil {
		return nil, err
	}
	if err = b.checkRoots(); err != nil {
		return nil, err
	}
	if info, e := b.dst.Lstat(lockFile); e == nil && !info.Mode().IsRegular() {
		return nil, errors.New("destination lock path must be a regular file")
	} else if e != nil && !errors.Is(e, fs.ErrNotExist) {
		return nil, e
	}
	b.lock, err = b.dst.OpenFile(lockFile, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err = lockDestination(b.lock); err != nil {
		return nil, fmt.Errorf("lock destination (another bridge may be running): %w", err)
	}
	return b, nil
}

func (b *bridge) close() {
	if b.lock != nil {
		b.lock.Close()
	}
	if b.src != nil {
		b.src.Close()
	}
	if b.dst != nil {
		b.dst.Close()
	}
}

// A retained directory handle must not silently redirect work after an unmount,
// path replacement, or rename. Never recreate missing disk roots.
func (b *bridge) checkRoots() error {
	for _, item := range []struct {
		path string
		info fs.FileInfo
	}{{b.cfg.srcDir, b.srcInfo}, {b.cfg.dstDir, b.dstInfo}} {
		now, err := os.Stat(item.path)
		if err != nil {
			return err
		}
		if !now.IsDir() || !os.SameFile(item.info, now) {
			return fmt.Errorf("disk root changed: %s; restart after restoring the mount", item.path)
		}
	}
	return nil
}

func inspectFolder(root *os.Root, folder string, minElapsed time.Duration) (bool, error) {
	info, err := root.Lstat(folder)
	if err != nil {
		return false, err
	}
	if !info.IsDir() {
		return false, fmt.Errorf("source folder is not a real directory: %s", folder)
	}
	entries, err := fs.ReadDir(root.FS(), filepath.ToSlash(folder))
	if err != nil {
		return false, err
	}
	var latest time.Time
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "lock") {
			return false, nil
		}
		if !strings.HasPrefix(entry.Name(), "finished") {
			continue
		}
		info, err := root.Lstat(filepath.Join(folder, entry.Name()))
		if err != nil {
			return false, err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		if latest.IsZero() || info.ModTime().After(latest) {
			latest = info.ModTime()
		}
	}
	return !latest.IsZero() && time.Since(latest) >= minElapsed, nil
}

func takeSnapshot(ctx context.Context, root *os.Root, folder string) (snapshot, error) {
	s := snapshot{files: make(map[string]fs.FileInfo)}
	err := fs.WalkDir(root.FS(), filepath.ToSlash(folder), func(path string, d fs.DirEntry, walkErr error) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if walkErr != nil {
			return walkErr
		}
		if strings.HasPrefix(d.Name(), ".hdd-bridge") {
			return fmt.Errorf("reserved source name: %s", path)
		}
		info, err := root.Lstat(filepath.FromSlash(path))
		if err != nil {
			return err
		}
		if !info.IsDir() && !info.Mode().IsRegular() {
			return fmt.Errorf("unsupported source type (including symlinks): %s", path)
		}
		s.files[filepath.FromSlash(path)] = info
		s.names = append(s.names, filepath.FromSlash(path))
		return nil
	})
	if err != nil {
		return snapshot{}, err
	}
	sort.Strings(s.names)
	h := sha256.New()
	for _, name := range s.names {
		info := s.files[name]
		// JSON escaping prevents names containing newlines from colliding.
		encoded, _ := json.Marshal([]any{name, uint32(info.Mode()), info.Size(), info.ModTime().UnixNano()})
		h.Write(encoded)
	}
	s.token = hex.EncodeToString(h.Sum(nil))
	return s, nil
}

func sameSnapshot(a, b snapshot) bool {
	if a.token != b.token || len(a.files) != len(b.files) {
		return false
	}
	for name, info := range a.files {
		if !os.SameFile(info, b.files[name]) {
			return false
		}
	}
	return true
}

func (b *bridge) validateSnapshot(ctx context.Context, folder string, before snapshot) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := b.checkRoots(); err != nil {
		return err
	}
	ready, err := inspectFolder(b.src, folder, b.cfg.minElapsed)
	if err != nil {
		return err
	}
	if !ready {
		return errors.New("source became locked or unfinished during transfer")
	}
	after, err := takeSnapshot(ctx, b.src, folder)
	if err != nil {
		return err
	}
	if !sameSnapshot(before, after) {
		return errors.New("source changed during transfer; retry on the next scan")
	}
	return nil
}

func (b *bridge) scanOnce(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := b.checkRoots(); err != nil {
		return err
	}
	entries, err := fs.ReadDir(b.src.FS(), ".")
	if err != nil {
		return err
	}
	var failures []error
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		ready, err := inspectFolder(b.src, name, b.cfg.minElapsed)
		if err == nil && !ready {
			continue
		}
		if err == nil {
			err = b.syncFolder(ctx, name)
		}
		if err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", name, err))
			logf("Sync failed for %s: %v", name, err)
		}
	}
	return errors.Join(failures...)
}

func ensureDirectory(root *os.Root, path string) error {
	// Validate each component; never replace an existing file or follow a symlink.
	for cur := path; cur != "."; cur = filepath.Dir(cur) {
		info, err := root.Lstat(cur)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("destination directory conflicts with an existing non-directory: %s", cur)
		}
	}
	return root.MkdirAll(path, 0o755)
}

func (b *bridge) syncFolder(ctx context.Context, folder string) error {
	before, err := takeSnapshot(ctx, b.src, folder)
	if err != nil {
		return err
	}
	if err := b.validateSnapshot(ctx, folder, before); err != nil {
		return err
	}
	if err := ensureDirectory(b.dst, folder); err != nil {
		return err
	}
	// Invalidate the old success record before any data mutation, even on retries.
	statePath := filepath.Join(folder, completionStateFile)
	oldState, _ := b.dst.ReadFile(statePath)
	var old completionState
	_ = json.Unmarshal(oldState, &old)
	if err := b.dst.Remove(statePath); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if b.cfg.fsync {
		if err := syncDirectory(b.dst, folder); err != nil {
			return err
		}
	}
	if err := b.cleanupTemps(ctx, folder); err != nil {
		return err
	}
	stats := syncStats{}
	// Walk order creates parents first. Publish source finished markers last.
	names := append([]string(nil), before.names...)
	sort.SliceStable(names, func(i, j int) bool { return isFinished(names[j], folder) && !isFinished(names[i], folder) })
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := b.checkRoots(); err != nil {
			return err
		}
		info := before.files[name]
		if info.IsDir() {
			if err := ensureDirectory(b.dst, name); err != nil {
				return err
			}
			continue
		}
		if err := ensureDirectory(b.dst, filepath.Dir(name)); err != nil {
			return err
		}
		var previous *fileRecord
		if old.Version == 1 && old.Source == b.cfg.srcDir {
			if record, ok := old.Files[name]; ok {
				previous = &record
			}
		}
		identical, err := b.sameDestinationFile(ctx, name, info, previous)
		if err != nil {
			return err
		}
		if identical {
			stats.skippedFiles++
			continue
		}
		if err := b.copyFile(ctx, name, info, before.files[folder]); err != nil {
			return fmt.Errorf("copy %s: %w", name, err)
		}
		stats.copiedFiles++
		stats.copiedBytes += info.Size()
	}
	if err := b.validateSnapshot(ctx, folder, before); err != nil {
		return err
	}
	records := make(map[string]fileRecord)
	destinationInfos := make(map[string]struct {
		name string
		info fs.FileInfo
	})
	for _, name := range before.names {
		info := before.files[name]
		actual, err := b.dst.Lstat(name)
		if err != nil {
			return err
		}
		if info.IsDir() && !actual.IsDir() || !info.IsDir() && (!actual.Mode().IsRegular() || actual.Size() != info.Size()) {
			return fmt.Errorf("destination changed before completion: %s", name)
		}
		// A case-insensitive archive cannot represent two distinct case-sensitive
		// source paths. Detect aliases rather than record an incomplete success.
		key := strings.ToLower(name)
		if existing, exists := destinationInfos[key]; exists && os.SameFile(existing.info, actual) && !os.SameFile(before.files[existing.name], info) {
			return fmt.Errorf("destination aliases distinct source paths: %s and %s", existing.name, name)
		}
		destinationInfos[key] = struct {
			name string
			info fs.FileInfo
		}{name, actual}
		if info.IsDir() {
			continue
		}
		records[name] = fileRecord{signature(info), signature(actual)}
	}
	if b.cfg.deleteExtraneous {
		stats.deletedPaths, err = b.deleteExtraneous(ctx, folder, before)
		if err != nil {
			return err
		}
	}
	if err := b.validateSnapshot(ctx, folder, before); err != nil {
		return err
	}
	if b.cfg.fsync {
		// Persist newly created directories, including their parent entries.
		for i := len(before.names) - 1; i >= 0; i-- {
			name := before.names[i]
			if before.files[name].IsDir() {
				if err := syncDirectory(b.dst, name); err != nil {
					return err
				}
			}
		}
		if err := syncDirectory(b.dst, "."); err != nil {
			return err
		}
	}

	if err := b.validateSnapshot(ctx, folder, before); err != nil {
		return err
	}
	// Reuse the timestamp when an unchanged tree has just been rechecked.
	state := completionState{Version: 1, Token: before.token, Source: b.cfg.srcDir, CopiedAt: time.Now().UTC(), Verified: b.cfg.verify, Files: records}
	if old.Version == state.Version && old.Token == state.Token && old.Source == state.Source && old.Verified == state.Verified && stats.copiedFiles == 0 && stats.deletedPaths == 0 {
		state.CopiedAt = old.CopiedAt
	}
	content, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if err := b.writeAtomic(ctx, statePath, append(content, '\n'), 0o644); err != nil {
		return err
	}
	if stats.copiedFiles > 0 || stats.deletedPaths > 0 || len(oldState) == 0 {
		logf("Completed %s: copied %d files (%d bytes), skipped %d, deleted %d", folder, stats.copiedFiles, stats.copiedBytes, stats.skippedFiles, stats.deletedPaths)
	}
	return nil
}

func isFinished(name, folder string) bool {
	return filepath.Dir(name) == folder && strings.HasPrefix(filepath.Base(name), "finished")
}

func (b *bridge) sameDestinationFile(ctx context.Context, name string, srcInfo fs.FileInfo, previous *fileRecord) (bool, error) {
	dstInfo, err := b.dst.Lstat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if !dstInfo.Mode().IsRegular() {
		return false, fmt.Errorf("destination file conflicts with a non-regular file: %s", name)
	}
	if srcInfo.Size() != dstInfo.Size() {
		return false, nil
	}
	if !b.cfg.verify {
		// Accept a filesystem's observed timestamp/permissions only when both
		// the source and the destination still match a prior successful transfer.
		// This avoids recopying forever on coarse timestamp filesystems.
		if previous != nil && previous.Source == signature(srcInfo) && previous.Destination == signature(dstInfo) {
			return true, nil
		}
		return srcInfo.ModTime().Equal(dstInfo.ModTime()) && srcInfo.Mode().Perm() == dstInfo.Mode().Perm(), nil
	}
	srcHash, err := b.hashFile(ctx, b.src, name, srcInfo)
	if err != nil {
		return false, err
	}
	dstHash, err := b.hashFile(ctx, b.dst, name, dstInfo)
	if err != nil || srcHash != dstHash {
		return false, err
	}
	if srcInfo.Mode().Perm() == dstInfo.Mode().Perm() {
		return true, nil
	}
	// Some archive filesystems cannot represent all permission bits. Keep their
	// observed mode only when the source mode has not changed since success.
	return previous != nil && previous.Source.Mode == signature(srcInfo).Mode && previous.Destination.Mode == signature(dstInfo).Mode, nil
}

func stableFile(before, after fs.FileInfo) bool {
	return after.Mode().IsRegular() && os.SameFile(before, after) && before.Size() == after.Size() && before.ModTime().Equal(after.ModTime()) && before.Mode() == after.Mode()
}

// This loop deliberately avoids io.Copy's ReaderFrom/WriterTo fast paths: it
// checks cancellation between bounded reads and detects short writes.
func copyContext(ctx context.Context, dst io.Writer, src io.Reader, buffer []byte) (int64, error) {
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, readErr := src.Read(buffer)
		if n > 0 {
			written, err := dst.Write(buffer[:n])
			total += int64(written)
			if err != nil {
				return total, err
			}
			if written != n {
				return total, io.ErrShortWrite
			}
		}
		if readErr == io.EOF {
			return total, nil
		}
		if readErr != nil {
			return total, readErr
		}
	}
}

func (b *bridge) openStable(root *os.Root, name string, info fs.FileInfo) (*os.File, error) {
	current, err := root.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !stableFile(info, current) {
		return nil, fmt.Errorf("file changed before open: %s", name)
	}
	file, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	opened, err := file.Stat()
	if err != nil || !stableFile(info, opened) {
		file.Close()
		if err == nil {
			err = fmt.Errorf("file changed while opening: %s", name)
		}
		return nil, err
	}
	return file, nil
}

func (b *bridge) hashFile(ctx context.Context, root *os.Root, name string, info fs.FileInfo) (string, error) {
	file, err := b.openStable(root, name, info)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	n, err := copyContext(ctx, hash, file, b.buffer)
	if err != nil {
		return "", err
	}
	after, err := file.Stat()
	if err != nil {
		return "", err
	}
	if n != info.Size() || !stableFile(info, after) {
		return "", fmt.Errorf("file changed while hashing: %s", name)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (b *bridge) createTemp(name string, mode fs.FileMode) (*os.File, string, error) {
	for range 10 {
		var nonce [16]byte
		if _, err := rand.Read(nonce[:]); err != nil {
			return nil, "", err
		}
		temp := filepath.Join(filepath.Dir(name), tempPrefix+hex.EncodeToString(nonce[:])+".tmp")
		file, err := b.dst.OpenFile(temp, os.O_CREATE|os.O_EXCL|os.O_RDWR, mode.Perm())
		if errors.Is(err, fs.ErrExist) {
			continue
		}
		return file, temp, err
	}
	return nil, "", errors.New("cannot allocate a unique temporary file")
}

func (b *bridge) copyFile(ctx context.Context, name string, info, folderInfo fs.FileInfo) (err error) {
	in, err := b.openStable(b.src, name, info)
	if err != nil {
		return err
	}
	defer in.Close()
	out, temp, err := b.createTemp(name, info.Mode())
	if err != nil {
		return err
	}
	defer func() { out.Close(); b.dst.Remove(temp) }()
	hash := sha256.New()
	var writer io.Writer = out
	if b.cfg.verify {
		writer = io.MultiWriter(out, hash)
	}
	n, err := copyContext(ctx, writer, in, b.buffer)
	if err != nil {
		return err
	}
	after, err := in.Stat()
	if err != nil {
		return err
	}
	current, err := b.src.Lstat(name)
	if err != nil {
		return err
	}
	if n != info.Size() || !stableFile(info, after) || !stableFile(info, current) {
		return errors.New("source changed during copy")
	}
	if b.cfg.verify {
		if _, err := out.Seek(0, io.SeekStart); err != nil {
			return err
		}
		readBack := sha256.New()
		if _, err := copyContext(ctx, readBack, out, b.buffer); err != nil {
			return err
		}
		if hex.EncodeToString(hash.Sum(nil)) != hex.EncodeToString(readBack.Sum(nil)) {
			return errors.New("copy checksum mismatch")
		}
	}
	if err := out.Chmod(info.Mode().Perm()); err != nil {
		return err
	}
	if err := b.dst.Chtimes(temp, info.ModTime(), info.ModTime()); err != nil {
		return err
	}
	if b.cfg.fsync {
		if err := out.Sync(); err != nil {
			return err
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := b.checkRoots(); err != nil {
		return err
	}
	folder := strings.Split(name, string(os.PathSeparator))[0]
	if err := b.checkSourceDirectory(folder, folderInfo); err != nil {
		return err
	}
	// Rename replaces a regular destination directly. Never unlink the old copy.
	if err := b.dst.Rename(temp, name); err != nil {
		return err
	}
	if b.cfg.fsync {
		return syncDirectory(b.dst, filepath.Dir(name))
	}
	return nil
}

func (b *bridge) writeAtomic(ctx context.Context, name string, data []byte, mode fs.FileMode) error {
	out, temp, err := b.createTemp(name, mode)
	if err != nil {
		return err
	}
	defer func() { out.Close(); b.dst.Remove(temp) }()
	if _, err := out.Write(data); err != nil {
		return err
	}
	if b.cfg.fsync {
		if err := out.Sync(); err != nil {
			return err
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := b.checkRoots(); err != nil {
		return err
	}
	if err := b.dst.Rename(temp, name); err != nil {
		return err
	}
	if b.cfg.fsync {
		return syncDirectory(b.dst, filepath.Dir(name))
	}
	return nil
}

func (b *bridge) deleteExtraneous(ctx context.Context, folder string, source snapshot) (int, error) {
	var extra []string
	err := fs.WalkDir(b.dst.FS(), filepath.ToSlash(folder), func(path string, d fs.DirEntry, walkErr error) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if walkErr != nil {
			return walkErr
		}
		name := filepath.FromSlash(path)
		if name == filepath.Join(folder, completionStateFile) {
			return nil
		}
		if _, exists := source.files[name]; !exists {
			extra = append(extra, name)
			if d.IsDir() {
				return fs.SkipDir
			}
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	for i, name := range extra {
		// Check source ancestors and the exact missing path before each removal.
		// Full-tree checks bookend pruning; rescanning every file for every extra
		// path would make large archives quadratic.
		if err := ctx.Err(); err != nil {
			return i, err
		}
		if err := b.checkRoots(); err != nil {
			return i, err
		}
		for parent := filepath.Dir(name); sameOrNestedPath(folder, parent); parent = filepath.Dir(parent) {
			if expected, exists := source.files[parent]; exists {
				if err := b.checkSourceDirectory(parent, expected); err != nil {
					return i, err
				}
			}
			if parent == folder {
				break
			}
		}
		if _, err := b.src.Lstat(name); !errors.Is(err, fs.ErrNotExist) {
			if err == nil {
				err = fmt.Errorf("source appeared before pruning: %s", name)
			}
			return i, err
		}
		if err := b.checkSourceDirectory(folder, source.files[folder]); err != nil {
			return i, err
		}
		if err := b.dst.RemoveAll(name); err != nil {
			return i, err
		}
		if b.cfg.fsync {
			if err := syncDirectory(b.dst, filepath.Dir(name)); err != nil {
				return i + 1, err
			}
		}
	}
	return len(extra), nil
}

// Only our reserved, randomly named regular temp files are cleaned without
// -delete. Source trees cannot contain this namespace.
func (b *bridge) cleanupTemps(ctx context.Context, folder string) error {
	return fs.WalkDir(b.dst.FS(), filepath.ToSlash(folder), func(path string, d fs.DirEntry, walkErr error) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if walkErr != nil {
			return walkErr
		}
		name := d.Name()
		if !strings.HasPrefix(name, tempPrefix) || !strings.HasSuffix(name, ".tmp") {
			return nil
		}
		nonce := strings.TrimSuffix(strings.TrimPrefix(name, tempPrefix), ".tmp")
		if len(nonce) != 32 {
			return nil
		}
		if _, err := hex.DecodeString(nonce); err != nil {
			return nil
		}
		info, err := b.dst.Lstat(filepath.FromSlash(path))
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		if err := b.checkRoots(); err != nil {
			return err
		}
		return b.dst.Remove(filepath.FromSlash(path))
	})
}

// Creating/removing a lock or other entry changes the source directory. This
// avoids listing all top-level files for every copied file while detecting a
// writer's lock before publication. Full snapshots check nested file changes.
func (b *bridge) checkSourceDirectory(name string, before fs.FileInfo) error {
	after, err := b.src.Lstat(name)
	if err != nil {
		return err
	}
	if !after.IsDir() || !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return fmt.Errorf("source directory changed (possibly a new lock): %s", name)
	}
	return nil
}

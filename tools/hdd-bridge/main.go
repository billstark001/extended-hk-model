package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"
)

type config struct {
	srcDir, dstDir                        string
	interval, minElapsed                  time.Duration
	bufferSize                            int
	deleteExtraneous, fsync, verify, once bool
}

func main() { os.Exit(run()) }

func run() int {
	var cfg config
	bufferMiB := flag.Int("buffer-mib", 4, "Copy buffer size in MiB (1..256).")
	flag.StringVar(&cfg.srcDir, "src", "", "Existing source directory on the fast disk.")
	flag.StringVar(&cfg.dstDir, "dst", "", "Existing destination directory on the slow disk.")
	flag.DurationVar(&cfg.interval, "interval", time.Minute, "How often to scan the source directory.")
	flag.DurationVar(&cfg.minElapsed, "minelapsed", 5*time.Minute, "Minimum age of the newest finished marker.")
	flag.BoolVar(&cfg.deleteExtraneous, "delete", false, "Remove extra destination paths inside eligible simulation folders.")
	flag.BoolVar(&cfg.fsync, "fsync", false, "Sync copied files and directory entries before recording completion.")
	flag.BoolVar(&cfg.verify, "verify", false, "Compare SHA-256 contents, including existing files and copy read-back.")
	flag.BoolVar(&cfg.once, "once", false, "Scan once and exit; return 1 if any eligible folder fails.")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s -src <ssd-dir> -dst <hdd-dir> [options]\n\nCopies immediate subdirectories with regular finished* files and no lock* entry.\nBoth roots must already exist and must not overlap. Source data is never deleted.\n\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 0 || cfg.srcDir == "" || cfg.dstDir == "" || cfg.interval <= 0 || cfg.minElapsed < 0 || *bufferMiB < 1 || *bufferMiB > 256 {
		fmt.Fprintln(os.Stderr, "[hdd-bridge] invalid arguments: require -src/-dst, positive -interval, nonnegative -minelapsed and -buffer-mib 1..256")
		return 2
	}
	cfg.bufferSize = *bufferMiB * 1024 * 1024
	bridge, err := openBridge(cfg)
	if err != nil {
		logf("Cannot start: %v", err)
		return 1
	}
	defer bridge.close()
	ctx, stop := signal.NotifyContext(context.Background(), shutdownSignals()...)
	defer stop()
	logf("Watching %s -> %s (delete=%t, fsync=%t, verify=%t)", bridge.cfg.srcDir, bridge.cfg.dstDir, cfg.deleteExtraneous, cfg.fsync, cfg.verify)
	for {
		err := bridge.scanOnce(ctx)
		if ctx.Err() != nil {
			logf("Stopped")
			return 130
		}
		if err != nil {
			logf("Scan failed: %v", err)
		}
		if cfg.once {
			if err != nil {
				return 1
			}
			return 0
		}
		timer := time.NewTimer(cfg.interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			logf("Stopped")
			return 130
		case <-timer.C:
		}
	}
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[hdd-bridge] %s "+format+"\n", append([]any{time.Now().Format(time.RFC3339)}, args...)...)
}

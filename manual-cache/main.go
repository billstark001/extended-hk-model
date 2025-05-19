package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// 递归拷贝目录
func copyDir(src string, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relPath, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(dst, relPath)
		if info.IsDir() {
			return os.MkdirAll(targetPath, info.Mode())
		} else {
			return copyFile(path, targetPath)
		}
	})
}

// 拷贝文件
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if err == nil {
			err = cerr
		}
	}()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}
	err = out.Sync()
	return err
}

func shouldCopyFolder(folderPath string, minElapsed time.Duration) (bool, error) {
	hasFinished := false
	hasLock := false
	var finishedFileTime time.Time

	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return false, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "finished") {
			hasFinished = true
			info, err := entry.Info()
			if err != nil {
				return false, err
			}
			// 取finished文件的创建时间（CreateTime），Go标准库没有直接获取创建时间，取 ctime 或 mtime
			finishedFileTime = info.ModTime()
		}
		if strings.HasPrefix(name, "lock") {
			hasLock = true
		}
	}
	if hasFinished && !hasLock {
		// 检查时间
		if time.Since(finishedFileTime) > minElapsed {
			return true, nil
		}
	}
	return false, nil
}

func main() {
	var (
		srcDir     string
		dstDir     string
		interval   int
		minElapsed int
	)

	flag.StringVar(&srcDir, "src", "", "源目录")
	flag.StringVar(&dstDir, "dst", "", "目标目录")
	flag.IntVar(&interval, "interval", 60, "扫描间隔秒")
	flag.IntVar(&minElapsed, "minelapsed", 300, "finished文件最小时长秒")
	flag.Parse()

	if srcDir == "" || dstDir == "" {
		fmt.Println("请指定-src 源目录 和 -dst 目标目录")
		os.Exit(1)
	}

	fmt.Printf("开始监控目录: %s, 目标目录: %s, 扫描间隔: %ds, 时间间隔: %ds\n", srcDir, dstDir, interval, minElapsed)

	ticker := time.NewTicker(time.Duration(interval) * time.Second)
	defer ticker.Stop()

	for {
		// 扫描目录下所有一级子文件夹
		entries, err := os.ReadDir(srcDir)
		if err != nil {
			fmt.Println("读取目录失败:", err)
			time.Sleep(10 * time.Second)
			continue
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			folderPath := filepath.Join(srcDir, entry.Name())
			ok, err := shouldCopyFolder(folderPath, time.Duration(minElapsed)*time.Second)
			if err != nil {
				fmt.Printf("检查文件夹 %s 出错: %v\n", folderPath, err)
				continue
			}
			if ok {
				targetPath := filepath.Join(dstDir, entry.Name())
				fmt.Printf("[%s] 满足条件，开始拷贝到 %s...\n", entry.Name(), targetPath)
				err := copyDir(folderPath, targetPath)
				if err != nil {
					fmt.Printf("拷贝出错: %v\n", err)
				} else {
					fmt.Println("拷贝完成。")
				}
			}
		}
		<-ticker.C
	}
}

package main

import (
	"encoding/json"
	"os"
	"testing"
)

var benchmarkResult MicroResult

// The parameter generator supplies every workload; this file defines no scan.
func BenchmarkMicroTask(b *testing.B) {
	data, err := os.ReadFile(os.Getenv("EHK_ENGINE_BENCH_INPUT"))
	if err != nil {
		b.Fatal(err)
	}
	var jobs []struct {
		ID      string `json:"id"`
		Payload Task   `json:"payload"`
	}
	if err := json.Unmarshal(data, &jobs); err != nil {
		b.Fatal(err)
	}
	for _, job := range jobs {
		b.Run(job.ID, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				result, err := runMicro(job.Payload)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkResult = result
			}
		})
	}
}

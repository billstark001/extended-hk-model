// Thin instrumentation only: stepping, restoration and classification belong
// to the public SMP and smp-meso libraries.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/vmihailenco/msgpack/v5"
	"io"
	"os"
	"smp-meso/config"
	"smp-meso/kinetic"
	"strings"
	"time"
)

type seed struct {
	Replicate int    `json:"replicate"`
	Value     uint64 `json:"value"`
}
type request struct {
	ID           string             `json:"id"`
	Kind         string             `json:"kind"`
	Source       string             `json:"source"`
	SourceSHA    string             `json:"source_sha256"`
	SourceKind   string             `json:"source_kind"`
	Mode         string             `json:"mode"`
	Steps        int                `json:"steps"`
	Observe      []int              `json:"observe_steps"`
	Seeds        []seed             `json:"seeds"`
	Output       string             `json:"output_dir"`
	Variant      string             `json:"variant"`
	Model        config.RunRequest  `json:"model"`
	Measure      kinetic.RunRequest `json:"measure"`
	BridgeAgents int                `json:"bridge_agents"`
	GroupCenter  float64            `json:"group_center"`
}
type stateRef struct {
	Step          int    `json:"step"`
	PhysicalHash  string `json:"physical_hash"`
	Checkpoint    string `json:"checkpoint"`
	CheckpointSHA string `json:"checkpoint_sha256"`
	Category      string `json:"category"`
	KAll          int    `json:"k_all"`
}
type pathResult struct {
	Replicate     int                      `json:"replicate"`
	Step          int                      `json:"step"`
	Category      string                   `json:"category"`
	KAll          int                      `json:"k_all"`
	StopReason    string                   `json:"stop_reason"`
	FirstHitStep  *int                     `json:"first_hit_step,omitempty"`
	Checkpoint    string                   `json:"checkpoint,omitempty"`
	CheckpointSHA string                   `json:"checkpoint_sha256,omitempty"`
	Delta         []float64                `json:"delta,omitempty"`
	Final         *kinetic.InspectionFrame `json:"final,omitempty"`
}
type result struct {
	States             []stateRef           `json:"states"`
	Paths              []pathResult         `json:"paths"`
	ExecutedSteps      int                  `json:"executed_steps"`
	ElapsedSeconds     float64              `json:"elapsed_seconds"`
	SourcePhysicalHash string               `json:"source_physical_hash"`
	Trace              *kinetic.TraceResult `json:"trace,omitempty"`
}

func run(r request) (result, error) {
	started := time.Now()
	out := result{States: []stateRef{}, Paths: []pathResult{}}
	if r.ID == "" || r.Steps < 0 || len(r.Seeds) == 0 || r.SourceSHA == "" {
		return out, fmt.Errorf("id, source hash, seeds and nonnegative steps required")
	}
	if r.Mode != "branch" && r.Mode != "resume" {
		return out, fmt.Errorf("invalid mode")
	}
	if r.Mode == "resume" && len(r.Seeds) != 1 {
		return out, fmt.Errorf("resume requires one original random stream")
	}
	if strings.HasSuffix(r.Kind, "moments") && r.Steps != 1 {
		return out, fmt.Errorf("moments require exactly one update")
	}
	if err := r.Model.Validate(); err != nil {
		return out, err
	}
	if r.Model.Dynamics.Type != "hk" || r.Model.Recommender.Type != "random" {
		return out, fmt.Errorf("first study requires HK + Random")
	}
	seen := map[int]bool{}
	for _, s := range r.Seeds {
		if s.Replicate < 0 || seen[s.Replicate] {
			return out, fmt.Errorf("invalid replicate IDs")
		}
		seen[s.Replicate] = true
	}
	if strings.HasPrefix(r.Kind, "micro_") {
		err := runMicro(r, &out)
		out.ElapsedSeconds = time.Since(started).Seconds()
		return out, err
	}
	err := runMeso(r, &out)
	out.ElapsedSeconds = time.Since(started).Seconds()
	return out, err
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 65536), 64<<20)
	encoder := msgpack.NewEncoder(os.Stdout)
	encoder.SetCustomStructTag("json")
	for scanner.Scan() {
		var r request
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&r)
		if err == nil {
			var extra any
			if decoder.Decode(&extra) != io.EOF {
				err = fmt.Errorf("one JSON object required")
			}
		}
		var out result
		if err == nil {
			out, err = run(r)
		}
		response := map[string]any{"id": r.ID, "status": "ok", "result": out}
		if err != nil {
			response = map[string]any{"id": r.ID, "status": "error", "error": err.Error()}
		}
		if err := encoder.Encode(response); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

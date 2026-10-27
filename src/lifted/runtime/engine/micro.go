package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"smp/observables"
	smprng "smp/rng"
	"smp/simulation"
	"smp/terminal"
)

func fileHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	value := sha256.Sum256(data)
	return hex.EncodeToString(value[:]), nil
}
func physicalHash(values ...any) string {
	data, _ := json.Marshal(values)
	value := sha256.Sum256(data)
	return hex.EncodeToString(value[:])
}
func label(k int, absorbed bool) string {
	if !absorbed {
		return "unresolved"
	}
	if k >= 4 {
		return "k4plus"
	}
	return fmt.Sprintf("k%d", k)
}
func classifyMicro(m *simulation.Scenario, epsilon float64) (terminal.Result, error) {
	return terminal.ClassifyOpinions(m.Model.GetOpinions(), epsilon, 0.5/float64(len(m.Model.GetOpinions())), 0, 0)
}
func microProjection(m *simulation.Scenario, bins int) ([]float64, []float64, error) {
	bounds := make([]float64, bins+1)
	for i := range bounds {
		bounds[i] = -1 + 2*float64(i)/float64(bins)
	}
	state, err := observables.MeasureBinnedState(m.Model.GetOpinions(), m.Model.GetGraph(), bounds)
	if err != nil {
		return nil, nil, err
	}
	return state.Population, state.Edge, nil
}
func saveMicro(m *simulation.Scenario, directory, name string) (string, string, error) {
	if err := os.MkdirAll(directory, 0755); err != nil {
		return "", "", err
	}
	path := filepath.Join(directory, name+".msgpack.lz4")
	if err := m.SaveCheckpointTo(path); err != nil {
		return "", "", err
	}
	sha, err := fileHash(path)
	return path, sha, err
}
func loadMicro(r request, s seed, prepared *simulation.PreparedCheckpoint, snapshot *simulation.RawSnapshotData) (*simulation.Scenario, error) {
	metadata := *snapshot.Metadata
	if metadata.DynamicsType != simulation.DynamicsTypeHK || metadata.RecsysFactoryType != "Random" ||
		metadata.HKParams.RepostRate != 0 || metadata.PostRetainCount != 1 ||
		metadata.NodeCount != r.Model.Population || metadata.NodeFollowCount != r.Model.OutDegree ||
		metadata.RecsysCount != r.Model.RecommendationCount || metadata.HKParams.Influence != r.Model.Dynamics.Influence ||
		metadata.HKParams.Tolerance != r.Model.Dynamics.Tolerance || metadata.HKParams.RewiringRate != r.Model.Dynamics.RewiringRate {
		return nil, fmt.Errorf("checkpoint physical protocol differs")
	}
	if r.Mode == "branch" {
		metadata.RNG = smprng.FixedSpec(s.Value, s.Value^0xbeef)
	}
	m := simulation.NewScenarioWithOptions("", &metadata, simulation.ScenarioOptions{Quiet: true, EnableDumps: false})
	if err := m.InitFromPreparedCheckpoint(prepared, r.Mode == "branch"); err != nil {
		return nil, err
	}
	g := m.Model.GetGraph()
	for i := range m.Model.GetOpinions() {
		if g.From(int64(i)).Len() != r.Model.OutDegree {
			return nil, fmt.Errorf("checkpoint out-degree differs")
		}
	}
	if r.BridgeAgents > 0 {
		if snapshot.CompletedStep != 0 || r.Kind != "micro_atlas" || r.BridgeAgents >= r.Model.Population || r.GroupCenter <= 0 || r.GroupCenter >= 1 {
			return nil, fmt.Errorf("bridge law applies only to step-zero atlas")
		}
		ops := make([]float64, r.Model.Population)
		for i := r.BridgeAgents; i < len(ops); i++ {
			sign := 1.0
			if i%2 == 0 {
				sign = -1
			}
			ops[i] = sign*r.GroupCenter + .01*(2*float64(i)/float64(len(ops))-1)
		}
		if err := m.InitFromState(&simulation.InitialState{Graph: g, Opinions: ops}); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func runMicro(r request, out *result) error {
	if r.SourceKind != "micro" {
		return fmt.Errorf("micro continuation requires a micro checkpoint")
	}
	snapshot, sha, err := simulation.ReadCheckpoint(r.Source)
	if err != nil {
		return err
	}
	if sha != r.SourceSHA {
		return fmt.Errorf("source checkpoint hash differs")
	}
	prepared, err := simulation.PrepareCheckpoint(snapshot)
	if err != nil {
		return err
	}
	atlas, moments := r.Kind == "micro_atlas", r.Kind == "micro_moments"
	if !atlas && !moments && r.Kind != "micro_probe" && r.Kind != "micro_extend" {
		return fmt.Errorf("unknown micro operation")
	}
	wanted := map[int]bool{}
	for _, step := range r.Observe {
		if step < 0 || step > r.Steps {
			return fmt.Errorf("observation outside horizon")
		}
		wanted[step] = true
	}
	if atlas && len(r.Seeds) != 1 {
		return fmt.Errorf("one atlas path per task required")
	}
	for _, s := range r.Seeds {
		m, err := loadMicro(r, s, prepared, snapshot)
		if err != nil {
			return err
		}
		initial, err := simulation.InspectFloat64Checkpoint(snapshot)
		if err != nil {
			return err
		}
		out.SourcePhysicalHash = physicalHash(initial.Opinions, initial.Graph)
		var before []float64
		if moments {
			a, b, err := microProjection(m, r.Model.OpinionBins)
			if err != nil {
				return err
			}
			before = append(a, b...)
		}
		base := m.Model.GetCurStep() - 1
		firstHit := -1
		var classification terminal.Result
		elapsed := 0
		for h := 0; h <= r.Steps; h++ {
			classification, err = classifyMicro(m, r.Model.Dynamics.Tolerance)
			if err != nil {
				return err
			}
			absorbed := classification.Status == terminal.StatusAbsorbed
			if absorbed && firstHit < 0 {
				firstHit = base + h
			}
			if atlas && wanted[h] {
				path, sha, err := saveMicro(m, r.Output, fmt.Sprintf("state-%09d", base+h))
				if err != nil {
					return err
				}
				snapshot, err := m.Snapshot()
				if err != nil {
					return err
				}
				physical, err := simulation.InspectFloat64Checkpoint(snapshot)
				if err != nil {
					return err
				}
				out.States = append(out.States, stateRef{base + h, physicalHash(physical.Opinions, physical.Graph), path, sha,
					label(classification.KAll, absorbed), classification.KAll})
			}
			if h == r.Steps || (!atlas && !moments && absorbed) {
				break
			}
			m.Step()
			elapsed++
		}
		p := pathResult{Replicate: s.Replicate, Step: base + elapsed, KAll: classification.KAll,
			Category: label(classification.KAll, classification.Status == terminal.StatusAbsorbed), StopReason: "horizon"}
		if classification.Status == terminal.StatusAbsorbed {
			p.StopReason = "geometric_terminal"
		}
		if atlas {
			p.StopReason = "atlas_window_complete"
			if firstHit >= 0 {
				p.FirstHitStep = &firstHit
			}
		}
		if moments {
			a, b, err := microProjection(m, r.Model.OpinionBins)
			if err != nil {
				return err
			}
			p.Delta = append(a, b...)
			for i := range p.Delta {
				p.Delta[i] -= before[i]
				if math.Abs(p.Delta[i]) < 1e-18 {
					p.Delta[i] = 0
				}
			}
		} else if !atlas && p.Category == "unresolved" {
			p.Checkpoint, p.CheckpointSHA, err = saveMicro(m, r.Output, fmt.Sprintf("tail-%06d", s.Replicate))
			if err != nil {
				return err
			}
		}
		out.ExecutedSteps += elapsed
		out.Paths = append(out.Paths, p)
	}
	return nil
}

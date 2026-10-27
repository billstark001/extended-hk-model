package main

import (
	"fmt"
	"github.com/vmihailenco/msgpack/v5"
	"math/rand/v2"
	"os"
	"path/filepath"
	"smp-meso/config"
	"smp-meso/kinetic"
	"smp-meso/lifted"
	"smp-meso/protocol"
	"smp-meso/terminal"
	"smp/simulation"
)

type mesoCheckpoint struct {
	Step             int
	Rho, Edge, Score []float64
	RNG              []byte
	ModelHash        string
	Variant          string
}

func modelHash(r request) string {
	return physicalHash(r.Model.Population, r.Model.OpinionBins, r.Model.OutDegree,
		r.Model.RecommendationCount, r.Model.Layer, r.Model.Dynamics, r.Model.Recommender, r.Model.Resolution, r.Model.Closure)
}
func encodeProjection(a, b []float64, bins int) (protocol.NodeEdgeProjection, error) {
	return protocol.EncodeNodeEdgeProjection(a, b, bins)
}
func saveMeso(state *lifted.State, rng *rand.PCG, step int, r request, name string) (string, string, error) {
	random, err := rng.MarshalBinary()
	if err != nil {
		return "", "", err
	}
	data, err := msgpack.Marshal(mesoCheckpoint{step, state.Rho, state.Edge, state.Score, random, modelHash(r), r.Variant})
	if err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(r.Output, 0755); err != nil {
		return "", "", err
	}
	path := filepath.Join(r.Output, name+".msgpack")
	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", "", err
	}
	sha, err := fileHash(path)
	return path, sha, err
}
func mesoSource(r request) ([]float64, []float64, []float64, []byte, int, error) {
	if sha, err := fileHash(r.Source); err != nil || sha != r.SourceSHA {
		return nil, nil, nil, nil, 0, fmt.Errorf("source checkpoint hash differs: %v", err)
	}
	if r.SourceKind == "meso" {
		data, err := os.ReadFile(r.Source)
		if err != nil {
			return nil, nil, nil, nil, 0, err
		}
		var cp mesoCheckpoint
		if err := msgpack.Unmarshal(data, &cp); err != nil {
			return nil, nil, nil, nil, 0, err
		}
		if cp.ModelHash != modelHash(r) {
			return nil, nil, nil, nil, 0, fmt.Errorf("meso checkpoint physical/numerical protocol differs")
		}
		if r.Mode == "resume" && cp.Variant != r.Variant {
			return nil, nil, nil, nil, 0, fmt.Errorf("resume variant differs")
		}
		return cp.Rho, cp.Edge, cp.Score, cp.RNG, cp.Step, nil
	}
	snapshot, _, err := simulation.ReadCheckpoint(r.Source)
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}
	prepared, err := simulation.PrepareCheckpoint(snapshot)
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}
	source := r
	source.Mode, source.Kind, source.BridgeAgents = "resume", "micro_probe", 0
	m, err := loadMicro(source, r.Seeds[0], prepared, snapshot)
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}
	a, b, err := microProjection(m, r.Model.OpinionBins)
	return a, b, make([]float64, len(b)), nil, snapshot.CompletedStep, err
}
func runMeso(r request, out *result) error {
	a, b, score, random, base, err := mesoSource(r)
	if err != nil {
		return err
	}
	out.SourcePhysicalHash = physicalHash(a, b)
	projection, err := encodeProjection(a, b, r.Model.OpinionBins)
	if err != nil {
		return err
	}
	if r.Kind == "measure" {
		trace, err := kinetic.TraceFromState(r.Measure, kinetic.TraceOptions{InitialState: &projection, PhysicalStep: 1, SnapshotSteps: r.Observe})
		if err != nil {
			return err
		}
		out.Trace = &trace
		out.ExecutedSteps = r.Measure.Steps
		return nil
	}
	atlas, moments := r.Kind == "meso_atlas", r.Kind == "meso_moments"
	if !atlas && !moments && r.Kind != "meso_probe" && r.Kind != "meso_extend" {
		return fmt.Errorf("unknown meso operation")
	}
	wanted := map[int]bool{}
	for _, step := range r.Observe {
		if step < 0 || step > r.Steps {
			return fmt.Errorf("observation outside horizon")
		}
		wanted[step] = true
	}
	if atlas && len(r.Seeds) != 1 {
		return fmt.Errorf("one atlas path required")
	}
	layer, err := config.ParseLayer(r.Model.Layer)
	if err != nil {
		return err
	}
	for _, s := range r.Seeds {
		state, err := lifted.FromProjected(r.Model, layer, projection, score, lifted.ClosureProfile{})
		if err != nil {
			return err
		}
		pcg := rand.NewPCG(s.Value, s.Value^0xbeef)
		if r.Mode == "resume" && len(random) > 0 {
			if err := pcg.UnmarshalBinary(random); err != nil {
				return err
			}
		}
		rng := rand.New(pcg)
		var classification terminal.Result
		firstHit, elapsed := -1, 0
		for h := 0; h <= r.Steps; h++ {
			classification, err = terminal.Classify(state.Axis, state.Rho, terminal.Options{Epsilon: r.Model.Dynamics.Tolerance,
				OccupiedMass: 0, MajorMass: .5 / float64(r.Model.Population)})
			if err != nil {
				return err
			}
			absorbed := classification.Status == terminal.StatusAbsorbed
			if absorbed && firstHit < 0 {
				firstHit = base + h
			}
			if atlas && wanted[h] {
				path, sha, err := saveMeso(state, pcg, base+h, r, fmt.Sprintf("state-%09d", base+h))
				if err != nil {
					return err
				}
				out.States = append(out.States, stateRef{base + h, physicalHash(state.Rho, state.Edge), path, sha, label(classification.KAll, absorbed), classification.KAll})
			}
			if h == r.Steps || (!atlas && !moments && r.Variant == "stochastic" && absorbed) {
				break
			}
			rewire, transport := r.Variant == "stochastic" || r.Variant == "rewire_only", r.Variant == "stochastic" || r.Variant == "transport_only"
			if r.Variant != "stochastic" && r.Variant != "deterministic" && r.Variant != "rewire_only" && r.Variant != "transport_only" {
				return fmt.Errorf("unknown variant")
			}
			if _, err := lifted.HybridStep(state, r.Model, lifted.ClosureProfile{}, rng, rewire, transport); err != nil {
				return err
			}
			elapsed++
		}
		p := pathResult{Replicate: s.Replicate, Step: base + elapsed, KAll: classification.KAll,
			Category: label(classification.KAll, classification.Status == terminal.StatusAbsorbed), StopReason: "horizon"}
		if classification.Status == terminal.StatusAbsorbed {
			p.StopReason = "geometric_terminal"
		}
		if !atlas && !moments && r.Variant != "stochastic" {
			p.StopReason = "diagnostic_window_complete"
		}
		if atlas {
			p.StopReason = "atlas_window_complete"
			if firstHit >= 0 {
				p.FirstHitStep = &firstHit
			}
		}
		if moments {
			p.Delta = append(append([]float64{}, state.Rho...), state.Edge...)
			before := append(append([]float64{}, a...), b...)
			for i := range p.Delta {
				p.Delta[i] -= before[i]
			}
		} else if !atlas && p.Category == "unresolved" {
			p.Checkpoint, p.CheckpointSHA, err = saveMeso(state, pcg, base+elapsed, r, fmt.Sprintf("tail-%06d", s.Replicate))
			if err != nil {
				return err
			}
		}
		final := kinetic.DescribeState(state.Axis, state.Rho, state.Edge, r.Model.Dynamics.Tolerance, r.Model.OutDegree)
		final.Step, final.Time = p.Step, float64(p.Step)
		p.Final = &final
		out.Paths = append(out.Paths, p)
		out.ExecutedSteps += elapsed
	}
	return nil
}

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/graph/simple"
	"math"
	"smp-meso/kinetic"
	"smp-meso/protocol"
	"smp/dynamics"
	"smp/model"
	"smp/observables"
	smprng "smp/rng"
	"smp/simulation"
	"sort"
)

// Model construction, stepping and full-state restoration belong to smp.
func microMetadata(t Task, seed uint64) (*simulation.ScenarioMetadata, error) {
	r := t.Request.Dynamics
	rec := t.Request.Recommender
	factories := map[string]string{
		"random": "Random", "opinion_random": "OpinionRandom", "structure_random_l0": "StructureRandom",
	}
	if t.Selection == "ranked" {
		factories["opinion_random"] = "Opinion"
		factories["structure_random_l0"] = "Structure"
	}
	factory, ok := factories[rec.Type]
	if !ok {
		return nil, fmt.Errorf("unsupported micro recommender %q", rec.Type)
	}
	metadata := &simulation.ScenarioMetadata{
		UniqueName: "paper_theory_micro", NetworkType: "Explicit",
		NodeCount: t.Request.Population, NodeFollowCount: t.Request.OutDegree,
		MaxSimulationStep: t.Request.Steps, RecsysFactoryType: factory,
		RNG: smprng.FixedSpec(seed, seed^0xbeef),
		SMPModelPureParams: model.SMPModelPureParams{
			RecsysCount: t.Request.RecommendationCount, PostRetainCount: t.MicroOptions.PostRetainCount,
		},
		CollectItemOptions: model.CollectItemOptions{AgentNumber: true},
		RecSysParams: map[string]any{
			"NoiseStd":         t.MicroOptions.RecommendationNoiseStd,
			"OpRandomNoiseStd": t.MicroOptions.RecommendationNoiseStd,
			"Tolerance":        rec.OpinionTolerance, "Steepness": rec.Steepness,
			"RandomRatio": rec.RandomRatio, "UseCache": true, "LogRecommendations": false,
		},
	}
	switch r.Type {
	case "hk":
		metadata.DynamicsType = simulation.DynamicsTypeHK
		metadata.HKParams = dynamics.HKParams{Tolerance: r.Tolerance, Influence: r.Influence, RewiringRate: r.RewiringRate, RepostRate: t.MicroOptions.RepostRate}
	case "deffuant":
		metadata.DynamicsType = simulation.DynamicsTypeDeffuant
		metadata.DeffuantParams = dynamics.DeffuantParams{Tolerance: r.Tolerance, Influence: r.Influence, RewiringRate: r.RewiringRate, RepostRate: t.MicroOptions.RepostRate}
	default:
		return nil, fmt.Errorf("unsupported micro dynamics %q", r.Type)
	}
	return metadata, nil
}

func makeMicro(t Task, ops []float64, graph *simple.DirectedGraph, seed uint64) (*simulation.Scenario, error) {
	metadata, err := microMetadata(t, seed)
	if err != nil {
		return nil, err
	}
	scenario := simulation.NewScenarioWithOptions("", metadata, simulation.ScenarioOptions{Quiet: true})
	if err := scenario.InitFromState(&simulation.InitialState{Graph: graph, Opinions: ops}); err != nil {
		return nil, err
	}
	return scenario, nil
}

// project bins node mass and directed-edge mass, both weighted by 1/N.
// Edge entries sum to the actual mean degree; no cell width is applied.
func project(ops []float64, g *simple.DirectedGraph, b int) ([]float64, []float64) {
	rho, edge := make([]float64, b), make([]float64, b*b)
	bins := make([]int, len(ops))
	inv := 1 / float64(len(ops))
	for i, x := range ops {
		bins[i] = max(0, min(b-1, int((x+1)*float64(b)/2)))
		rho[bins[i]] += inv
	}
	for u := range ops {
		it := g.From(int64(u))
		for it.Next() {
			v := int(it.Node().ID())
			edge[bins[u]*b+bins[v]] += inv
		}
	}
	return rho, edge
}

// feasible enforces the pair closure's row sums k*rho without changing rho.
// Empty source rows use random mixing. The returned L1 change is audit evidence.
func feasible(rho, edge []float64, k int) ([]float64, float64) {
	b := len(rho)
	out := append([]float64(nil), edge...)
	change := 0.0
	for i := 0; i < b; i++ {
		sum := 0.0
		for j := 0; j < b; j++ {
			sum += edge[i*b+j]
		}
		for j := 0; j < b; j++ {
			x := float64(k) * rho[i] * rho[j]
			if sum > 0 {
				x = float64(k) * rho[i] * edge[i*b+j] / sum
			}
			change += math.Abs(x - edge[i*b+j])
			out[i*b+j] = x
		}
	}
	return out, change
}

// MicroFrame joins binned pair observables with continuous microscopic energy.
// Time is physical update time; exact edge normalization follows the task policy.
type MicroFrame struct {
	kinetic.InspectionFrame
	ExactNodeEnergy float64 `json:"exact_node_energy"`
	ExactEdgeEnergy float64 `json:"exact_edge_energy"`
	Rewires         int     `json:"rewires"`
	MeanDegree      float64 `json:"mean_degree"`
	ExactVariance   float64 `json:"exact_variance"`
	ExactDiscordant float64 `json:"exact_discordant"`
	Projected       bool    `json:"projected"`
}

// MicroResult pairs one physical microscopic start with its projected trace.
// InitialHash identifies opinions and graph edges; ProjectionL1 reports closure
// adjustment. Audits are conditional process branches, not independent starts.
type MicroResult struct {
	InitialHash  string              `json:"initial_hash"`
	ProjectionL1 float64             `json:"projection_l1"`
	Frames       []MicroFrame        `json:"frames"`
	Pair         kinetic.TraceResult `json:"pair"`
	Audits       []MicroAudit        `json:"audits"`
}

func runMicro(t Task) (MicroResult, error) {
	var out MicroResult
	if t.InitialRNG != "shared" && t.InitialRNG != "split" {
		return out, fmt.Errorf("initial_rng must be shared or split")
	}
	if t.Projection != "fixed_degree_rows" && t.Projection != "actual" {
		return out, fmt.Errorf("unknown projection policy")
	}
	if t.RecordPolicy != "audit" && t.RecordPolicy != "population" {
		return out, fmt.Errorf("unknown recording policy")
	}
	if t.EnergyNormalization != "nominal" && t.EnergyNormalization != "actual" {
		return out, fmt.Errorf("unknown energy normalization")
	}
	if t.MicroOptions == nil || t.MicroOptions.PostRetainCount != 1 || t.MicroOptions.RepostRate != 0 || t.MicroOptions.RecommendationNoiseStd != 0 {
		return out, fmt.Errorf("this micro audit requires explicit no-history, no-repost, no-noise options")
	}
	if t.Selection != "weighted" && t.Selection != "ranked" {
		return out, fmt.Errorf("invalid recommendation selection")
	}
	if err := t.Request.Validate(); err != nil {
		return out, err
	}
	if t.Request.Dt != 1 || t.PhysicalDt != 1 || t.Request.Dynamics.OpinionMethod != "measure" {
		return out, fmt.Errorf("micro comparison requires measure and unit physical dt")
	}
	ops, g, err := initialize(t)
	if err != nil {
		return out, err
	}
	pairs := [][2]int{}
	for u := range ops {
		it := g.From(int64(u))
		for it.Next() {
			pairs = append(pairs, [2]int{u, int(it.Node().ID())})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i][0] != pairs[j][0] {
			return pairs[i][0] < pairs[j][0]
		}
		return pairs[i][1] < pairs[j][1]
	})
	bytes, _ := json.Marshal(struct {
		Opinions []float64
		Edges    [][2]int
	}{ops, pairs})
	out.InitialHash = fmt.Sprintf("%x", sha256.Sum256(bytes))
	rho, edge := project(ops, g, t.Request.OpinionBins)
	pe, adj := edge, 0.0
	if t.Projection == "fixed_degree_rows" {
		pe, adj = feasible(rho, edge, t.Request.OutDegree)
	}
	out.ProjectionL1 = adj
	projection, e := protocol.EncodeNodeEdgeProjection(rho, pe, t.Request.OpinionBins)
	if e != nil {
		return out, e
	}
	if t.Kind != "micro_only" {
		out.Pair, err = kinetic.TraceFromState(t.Request, kinetic.TraceOptions{InitialState: &projection, PhysicalStep: 1, SnapshotSteps: t.SnapshotSteps, EnergyBudgets: t.Budgets})
		if err != nil {
			return out, err
		}
	}
	wanted := map[int]bool{}
	for _, k := range t.SnapshotSteps {
		wanted[k] = true
	}
	if t.RecordPolicy == "population" {
		for i := range out.Pair.Frames {
			f := &out.Pair.Frames[i]
			if !wanted[f.Step] && f.Step != t.Request.Steps {
				f.Edge = nil
				if t.CompactRecords {
					f.Rho = nil
				}
			}
			f.Velocity, f.NodePotential, f.EdgePotential = nil, nil, nil
		}
	}
	if t.Kind == "pair_only" {
		return out, nil
	}
	m, err := makeMicro(t, ops, g, t.ProcessSeed)
	if err != nil {
		return out, err
	}
	audits := map[int]bool{}
	for _, k := range t.AuditSteps {
		audits[k] = true
	}
	axis := out.Pair.Axis
	if t.Kind == "micro_only" {
		axis = make([]float64, t.Request.OpinionBins)
		dx := (t.Request.Initial.OpinionMax - t.Request.Initial.OpinionMin) / float64(len(axis))
		for i := range axis {
			axis[i] = t.Request.Initial.OpinionMin + (float64(i)+0.5)*dx
		}
	}
	rewires := 0
	for step := 0; step <= t.Request.Steps; step++ {
		if audits[step] && t.Branches > 0 {
			a, err := audit(t, m)
			if err != nil {
				return out, err
			}
			out.Audits = append(out.Audits, a)
		}
		if wanted[step] || step == 0 || step%t.Request.RecordEvery == 0 || step == t.Request.Steps {
			projected := !t.CompactRecords || wanted[step] || step == 0 || step == t.Request.Steps
			var edge []float64
			f := kinetic.InspectionFrame{Time: float64(step), Step: step}
			if projected {
				var rho []float64
				rho, edge = project(m.Model.GetOpinions(), m.Model.GetGraph(), t.Request.OpinionBins)
				f = kinetic.DescribeState(axis, rho, edge, t.Request.Dynamics.Tolerance, t.Request.OutDegree)
				f.Time, f.Step = float64(step), step
				if wanted[step] || step == t.Request.Steps || t.RecordPolicy == "population" {
					f.Rho, f.Edge = rho, edge
				}
			}
			energy, err := observables.MeasureEnergy(m.Model.GetOpinions(), m.Model.GetGraph(), t.Request.Dynamics.Tolerance, 1)
			if err != nil {
				return out, err
			}
			// The manuscript normalizes by N*k, including ER graphs. The
			// shared observable normalizes by the actual number of edges.
			u := energy.Population
			e := energy.Edge * float64(energy.Edges) / float64(energy.Nodes*t.Request.OutDegree)
			if t.EnergyNormalization == "actual" {
				if t.Network == "fixed" && energy.Edges != energy.Nodes*t.Request.OutDegree {
					return out, fmt.Errorf("fixed degree was not conserved")
				}
				e = energy.Edge
			}
			variance, discordant := 0.0, 0.0
			if t.RecordPolicy == "population" {
				mean := 0.0
				ops := m.Model.GetOpinions()
				for _, x := range ops {
					mean += x / float64(len(ops))
				}
				for u, x := range ops {
					variance += (x - mean) * (x - mean) / float64(len(ops))
					it := m.Model.GetGraph().From(int64(u))
					for it.Next() {
						if math.Abs(x-ops[int(it.Node().ID())]) > t.Request.Dynamics.Tolerance {
							discordant += 1 / float64(energy.Nodes*t.Request.OutDegree)
						}
					}
				}
				if !wanted[step] && step != t.Request.Steps {
					f.Edge = nil
				}
			}
			sum := float64(energy.Edges) / float64(energy.Nodes)
			if projected {
				sum = 0
				for _, x := range edge {
					sum += x
				}
			}
			out.Frames = append(out.Frames, MicroFrame{InspectionFrame: f, ExactNodeEnergy: u, ExactEdgeEnergy: e, Rewires: rewires, MeanDegree: sum, ExactVariance: variance, ExactDiscordant: discordant, Projected: projected})
		}
		if step < t.Request.Steps {
			rw, _ := m.Step()
			rewires += rw
		}
	}
	return out, nil
}

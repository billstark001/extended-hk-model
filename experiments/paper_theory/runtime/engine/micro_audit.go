package main

import (
	"fmt"
	"math"
	"smp-meso/kinetic"
	"smp-meso/protocol"
	smprng "smp/rng"
	"smp/simulation"
)

// MicroAudit records one-step moments conditional on a complete saved state.
// Node/edge deltas are cell masses; opinion increments are unscaled physical
// updates. All branches share the state and vary only their process seeds.
type MicroAudit struct {
	Step            int                  `json:"step"`
	Branches        int                  `json:"branches"`
	MeanRho         []float64            `json:"mean_delta_rho"`
	VarianceRho     []float64            `json:"variance_delta_rho"`
	MeanEdge        []float64            `json:"mean_delta_edge"`
	VarianceEdge    []float64            `json:"variance_delta_edge"`
	MeanIncrement   []float64            `json:"mean_increment"`
	SecondIncrement []float64            `json:"second_increment"`
	SourceCounts    []int                `json:"source_counts"`
	MeanCross       float64              `json:"mean_endpoint_product"`
	CrossVariance   float64              `json:"variance_endpoint_product"`
	RewireMean      float64              `json:"rewire_mean"`
	RewireVariance  float64              `json:"rewire_variance"`
	EligibilityMean float64              `json:"eligibility_mean"`
	EmptyFeedMean   float64              `json:"empty_feed_mean"`
	ProjectionL1    float64              `json:"projection_l1"`
	Pair            kinetic.TraceResult  `json:"pair_one_step"`
	Frozen          kinetic.FrozenFields `json:"frozen"`
}

func audit(t Task, parent *simulation.Scenario) (MicroAudit, error) {
	step := parent.Model.GetCurStep() - 1
	ops, g := parent.Model.GetOpinions(), parent.Model.GetGraph()
	snapshot, err := parent.Snapshot()
	if err != nil {
		return MicroAudit{}, err
	}
	prepared, err := simulation.PrepareCheckpoint(snapshot)
	if err != nil {
		return MicroAudit{}, err
	}
	b := t.Request.OpinionBins
	n := len(ops)
	rho, edge := project(ops, g, b)
	pe, adj := feasible(rho, edge, t.Request.OutDegree)
	out := MicroAudit{Step: step, Branches: t.Branches, MeanRho: make([]float64, b), VarianceRho: make([]float64, b), MeanEdge: make([]float64, b*b), VarianceEdge: make([]float64, b*b), MeanIncrement: make([]float64, b), SecondIncrement: make([]float64, b), SourceCounts: make([]int, b), ProjectionL1: adj}
	seeds := t.BranchSeeds[step]
	if len(seeds) != t.Branches {
		return out, fmt.Errorf("branch seed count differs from requested branches at step %d", step)
	}
	for _, x := range ops {
		out.SourceCounts[max(0, min(b-1, int((x+1)*float64(b)/2)))]++
	}
	for branch := 0; branch < t.Branches; branch++ {
		metadata := *parent.Metadata
		metadata.RNG = smprng.FixedSpec(seeds[branch], seeds[branch]^0xbeef)
		m := simulation.NewScenarioWithOptions("", &metadata, simulation.ScenarioOptions{Quiet: true})
		if err := m.InitFromPreparedCheckpoint(prepared, true); err != nil {
			return out, err
		}
		rw, _ := m.Step()
		after := m.Model.GetOpinions()
		ar, ae := project(after, m.Model.GetGraph(), b)
		for i, x := range ar {
			d := x - rho[i]
			out.MeanRho[i] += d
			out.VarianceRho[i] += d * d
		}
		for i, x := range ae {
			d := x - edge[i]
			out.MeanEdge[i] += d
			out.VarianceEdge[i] += d * d
		}
		delta := make([]float64, n)
		for i, x := range ops {
			bin := max(0, min(b-1, int((x+1)*float64(b)/2)))
			delta[i] = after[i] - x
			out.MeanIncrement[bin] += delta[i]
			out.SecondIncrement[bin] += delta[i] * delta[i]
		}
		cross, total := 0.0, 0.0
		for u := range ops {
			it := g.From(int64(u))
			for it.Next() {
				cross += delta[u] * delta[it.Node().ID()]
				total++
			}
		}
		if total > 0 {
			cross /= total
		}
		out.MeanCross += cross
		out.CrossVariance += cross * cross
		out.RewireMean += float64(rw) / float64(n)
		out.RewireVariance += math.Pow(float64(rw)/float64(n), 2)
		counts, err := simulation.ObserveAgentNumbers(m.Model)
		if err != nil {
			return out, err
		}
		for _, c := range counts {
			if c[2] > 0 && c[1] > 0 {
				out.EligibilityMean += 1 / float64(n)
			}
			if c[0]+c[1] == 0 {
				out.EmptyFeedMean += 1 / float64(n)
			}
		}
	}
	f := float64(t.Branches)
	finish := func(mean, variance []float64) {
		for i := range mean {
			mean[i] /= f
			variance[i] = math.Max(0, (variance[i]-f*mean[i]*mean[i])/math.Max(1, f-1))
		}
	}
	finish(out.MeanRho, out.VarianceRho)
	finish(out.MeanEdge, out.VarianceEdge)
	for i, c := range out.SourceCounts {
		if c > 0 {
			out.MeanIncrement[i] /= f * float64(c)
			out.SecondIncrement[i] /= f * float64(c)
		}
	}
	out.MeanCross /= f
	out.CrossVariance = math.Max(0, (out.CrossVariance-f*out.MeanCross*out.MeanCross)/math.Max(1, f-1))
	out.RewireMean /= f
	out.RewireVariance = math.Max(0, (out.RewireVariance-f*out.RewireMean*out.RewireMean)/math.Max(1, f-1))
	out.EligibilityMean /= f
	out.EmptyFeedMean /= f
	r := t.Request
	r.Steps = 1
	r.RecordEvery = 1
	r.Stopping.Mode = "fixed_steps"
	r.Stopping.MinimumSteps = 0
	r.Stopping.CheckEvery = 1
	r.Stopping.PatienceSteps = 1
	projection, e := protocol.EncodeNodeEdgeProjection(rho, pe, b)
	if e != nil {
		return out, e
	}
	out.Pair, err = kinetic.TraceFromState(r, kinetic.TraceOptions{InitialState: &projection, PhysicalStep: 1, SnapshotSteps: []int{0, 1}, EnergyBudgets: true})
	if err != nil {
		return out, err
	}
	out.Frozen, err = kinetic.InspectFrozenState(r, &projection)
	return out, err
}

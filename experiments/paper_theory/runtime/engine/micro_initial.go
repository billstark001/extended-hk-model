package main

import (
	"fmt"
	"gonum.org/v1/gonum/graph/simple"
	"math"
	"math/rand/v2"
	"smp/utils"
)

// Initial-law parameters come from the experiment parameter generator.
// This adapter samples the supplied law; it does not define experiment points.
type normalComponent struct {
	Weight float64 `json:"weight"`
	Mean   float64 `json:"mean"`
	Std    float64 `json:"std"`
}
type microscopicInitial struct {
	Type        string            `json:"type"`
	Minimum     float64           `json:"minimum"`
	Maximum     float64           `json:"maximum"`
	ClipMinimum float64           `json:"clip_minimum"`
	ClipMaximum float64           `json:"clip_maximum"`
	Components  []normalComponent `json:"components"`
}

type microOptions struct {
	PostRetainCount        int     `json:"post_retain_count"`
	RepostRate             float64 `json:"repost_rate"`
	RecommendationNoiseStd float64 `json:"recommendation_noise_std"`
}

func (s *microscopicInitial) validate() error {
	if s == nil {
		return fmt.Errorf("microscopic_initial is required")
	}
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if !finite(s.Minimum) || !finite(s.Maximum) || !finite(s.ClipMinimum) || !finite(s.ClipMaximum) || s.Minimum >= s.Maximum || s.ClipMinimum > s.ClipMaximum || s.ClipMinimum < s.Minimum || s.ClipMaximum > s.Maximum {
		return fmt.Errorf("invalid microscopic initial bounds")
	}
	switch s.Type {
	case "uniform":
		if len(s.Components) != 0 {
			return fmt.Errorf("uniform law has mixture components")
		}
	case "gaussian_mixture":
		weight := 0.0
		for _, c := range s.Components {
			if !finite(c.Weight) || !finite(c.Mean) || !finite(c.Std) || c.Weight <= 0 || c.Std <= 0 {
				return fmt.Errorf("invalid normal component")
			}
			weight += c.Weight
		}
		if math.Abs(weight-1) > 1e-12 {
			return fmt.Errorf("normal weights must sum to one")
		}
	default:
		return fmt.Errorf("unknown microscopic initial type %q", s.Type)
	}
	return nil
}

func initialize(t Task) ([]float64, *simple.DirectedGraph, error) {
	n, k := t.Request.Population, t.Request.OutDegree
	if n <= k+1 || t.Network != "fixed" && t.Network != "er" {
		return nil, nil, fmt.Errorf("invalid network/population")
	}
	if err := t.MicroscopicInitial.validate(); err != nil {
		return nil, nil, err
	}
	law := t.MicroscopicInitial
	rng := rand.New(rand.NewPCG(t.InitSeed, t.InitSeed^0xcafe))
	graphRNG := rng
	if t.InitialRNG == "split" {
		// Keep graph draws independent of how many draws the opinion law consumes.
		graphRNG = rand.New(rand.NewPCG(t.InitSeed^0xface, t.InitSeed^0xbeef))
	}
	ops := make([]float64, n)
	for i := range ops {
		switch law.Type {
		case "uniform":
			ops[i] = law.Minimum + (law.Maximum-law.Minimum)*rng.Float64()
		case "gaussian_mixture":
			draw := rng.Float64()
			component := law.Components[len(law.Components)-1]
			for _, c := range law.Components {
				draw -= c.Weight
				if draw < 0 {
					component = c
					break
				}
			}
			ops[i] = component.Mean + component.Std*rng.NormFloat64()
		}
		ops[i] = math.Max(law.ClipMinimum, math.Min(law.ClipMaximum, ops[i]))
	}
	if t.Network == "er" {
		return ops, utils.CreateRandomNetwork(n, float64(k)/float64(n-1), graphRNG), nil
	}
	g := simple.NewDirectedGraph()
	for i := 0; i < n; i++ {
		g.AddNode(simple.Node(i))
	}
	for i := 0; i < n; i++ {
		seen := map[int]bool{i: true}
		for len(seen) < k+1 {
			j := graphRNG.IntN(n)
			if !seen[j] {
				seen[j] = true
				g.SetEdge(g.NewEdge(g.Node(int64(i)), g.Node(int64(j))))
			}
		}
	}
	return ops, g, nil
}

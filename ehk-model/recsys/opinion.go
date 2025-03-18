package recsys

import (
	"math"
	"math/rand"
	"sort"

	"ehk-model/model"
)

// Opinion implements a recommendation system based on opinion similarity
type Opinion struct {
	model.BaseRecommendationSystem
	Model        *model.HKModel
	NoiseStd     float64
	NumNodes     int
	Epsilon      []float64
	Agents       []*model.HKAgent
	AgentIndices map[int64]int
}

// NewOpinion creates a new opinion-based recommendation system
func NewOpinion(model *model.HKModel, noiseStd float64) *Opinion {
	return &Opinion{
		Model:    model,
		NoiseStd: noiseStd,
	}
}

// PostInit implements model.HKModelRecommendationSystem
func (o *Opinion) PostInit(dumpData any) {
	o.NumNodes = o.Model.Graph.Nodes().Len()
	o.Agents = o.Model.Schedule.Agents
	o.AgentIndices = make(map[int64]int, o.NumNodes)
	o.Epsilon = make([]float64, o.NumNodes)
}

// PreStep implements model.HKModelRecommendationSystem
func (o *Opinion) PreStep() {
	// Sort agents by current opinion
	sort.Slice(o.Agents, func(i, j int) bool {
		return o.Agents[i].CurOpinion < o.Agents[j].CurOpinion
	})

	// Update agent indices map
	for i, a := range o.Agents {
		o.AgentIndices[a.ID] = i
	}

	// Generate random noise
	for i := range o.Epsilon {
		o.Epsilon[i] = rand.NormFloat64() * o.NoiseStd
	}
}

// Recommend implements model.HKModelRecommendationSystem
func (o *Opinion) Recommend(agent *model.HKAgent, neighbors []*model.HKAgent, count int) []*model.TweetRecord {
	// Create set of neighbor IDs
	neighborIDs := make(map[int64]bool)
	neighborIDs[agent.ID] = true
	for _, n := range neighbors {
		neighborIDs[n.ID] = true
	}

	// Get adjusted opinion with noise
	opinionWithNoise := agent.CurOpinion + o.Epsilon[agent.ID]

	// Start indices for searching closest agents by opinion
	iPre := o.AgentIndices[agent.ID] - 1
	iPost := o.AgentIndices[agent.ID] + 1

	ret := make([]*model.TweetRecord, 0, count)

	// Find closest agents by opinion difference
	for len(ret) < count {
		noPre := iPre < 0
		noPost := iPost >= len(o.Agents)
		if noPre && noPost {
			break
		}

		// Determine whether to use predecessor or successor
		usePre := noPost || (!noPre &&
			math.Abs(opinionWithNoise-o.Agents[iPre].CurOpinion) <
				math.Abs(o.Agents[iPost].CurOpinion-opinionWithNoise))

		var a *model.HKAgent
		if usePre {
			a = o.Agents[iPre]
			iPre--
		} else {
			a = o.Agents[iPost]
			iPost++
		}

		// Skip existing neighbors
		if !neighborIDs[a.ID] {
			// Get the latest tweet from the agent
			if a.CurTweet != nil && a.CurTweet.AgentID != agent.ID {
				ret = append(ret, a.CurTweet)
			}
		}
	}

	return ret
}

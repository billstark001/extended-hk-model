package recsys

import (
	"math"
	"math/rand"

	"ehk-model/model"
)

// OpinionRandom implements a recommendation system with random opinion preferences
type OpinionRandom struct {
	model.BaseRecommendationSystem
	Model       *model.HKModel
	Tolerance   float64
	Steepness   float64
	NoiseStd    float64
	RandomRatio float64

	NumNodes   int
	Agents     []*model.HKAgent
	AllIndices []int
	RateMat    [][]float64
}

// NewOpinionRandom creates a new random opinion-based recommendation system
func NewOpinionRandom(model *model.HKModel, tolerance, steepness, noiseStd, randomRatio float64) *OpinionRandom {
	return &OpinionRandom{
		Model:       model,
		Tolerance:   tolerance,
		Steepness:   steepness,
		NoiseStd:    noiseStd,
		RandomRatio: randomRatio,
	}
}

// PostInit implements model.HKModelRecommendationSystem
func (o *OpinionRandom) PostInit(dumpData []byte) {
	o.NumNodes = o.Model.Graph.Nodes().Len()
	o.AllIndices = make([]int, o.NumNodes)
	for i := range o.NumNodes {
		o.AllIndices[i] = i
	}

	// Cache agents
	o.Agents = o.Model.Schedule.Agents

	// Initialize rate matrix
	o.RateMat = makeRawMat[float64](o.NumNodes, o.NumNodes)
}

// PreStep implements model.HKModelRecommendationSystem
func (o *OpinionRandom) PreStep() {

	// if o.Model.CurStep > 5 && o.Model.CurStep%5 != 0 {
	// 	return
	// }

	// Get all opinions
	opinions := make([]float64, o.NumNodes)
	for i, agent := range o.Agents {
		opinions[i] = agent.CurOpinion
	}

	// Calculate opinion difference matrix and rate matrix
	rawRateMat := makeRawMat[float64](o.NumNodes, o.NumNodes)

	// Calculate raw rate matrix based on opinion differences
	for i := range o.NumNodes {
		for j := i + 1; j < o.NumNodes; j++ {

			// Calculate rate based on opinion difference
			diff := math.Abs(opinions[i] - opinions[j])
			rate := max(1.0-diff/o.Tolerance, 0)

			// Add noise if needed
			if o.NoiseStd > 0 {
				noise := rand.NormFloat64() * o.NoiseStd
				rate = max(rate*(1-2*noise)+noise, 0)
			}

			// Apply steepness
			if o.Steepness != 1 {
				rate = math.Pow(rate, o.Steepness)
			}

			rawRateMat[i][j] = rate
			rawRateMat[j][i] = rate
		}
	}

	// Normalize rate matrix
	for i := range o.NumNodes {
		sum := 0.0
		for j := range o.NumNodes {
			sum += rawRateMat[i][j]
		}

		if sum > 0 {
			for j := range o.NumNodes {
				// Normalize and add random component if needed
				if o.RandomRatio > 0 {
					o.RateMat[i][j] = (1-o.RandomRatio)*rawRateMat[i][j]/sum +
						o.RandomRatio/(float64(o.NumNodes)-1)
				} else {
					o.RateMat[i][j] = rawRateMat[i][j] / sum
				}
			}
		}
		o.RateMat[i][i] = 0 // No self-recommendation
	}
}

// Recommend implements model.HKModelRecommendationSystem
func (o *OpinionRandom) Recommend(
	agent *model.HKAgent,
	neighbors []*model.HKAgent,
	count int,
) []*model.TweetRecord {
	// Create set of neighbor IDs
	neighborIDs := make(map[int64]bool)
	neighborIDs[agent.ID] = true
	for _, n := range neighbors {
		neighborIDs[n.ID] = true
	}

	// Create a copy of the rate vector
	rateVec := make([]float64, o.NumNodes)
	copy(rateVec, o.RateMat[agent.ID])

	// Set rate to 0 for neighbors
	sum := 0.0
	for id := range neighborIDs {
		rateVec[id] = 0
	}

	// Renormalize
	for i := range rateVec {
		sum += rateVec[i]
	}
	if sum > 0 {
		for i := range rateVec {
			rateVec[i] /= sum
		}
	}

	// Sample agents based on probability
	candidates := sampleWithoutReplacement(o.AllIndices, count, rateVec)

	// Collect tweets from selected agents
	ret := make([]*model.TweetRecord, 0, len(candidates))
	for _, idx := range candidates {
		a := o.Agents[idx]
		if a.CurTweet != nil && a.CurTweet.AgentID != agent.ID {
			ret = append(ret, a.CurTweet)
		}
	}

	return ret
}

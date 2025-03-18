package recsys

import (
	"ehk-model/model"
	"math"
)

// StructureRandom implements a weighted random recommendation based on network structure
type StructureRandom struct {
	Structure
	Steepness   float64
	RandomRatio float64
}

// NewStructureRandom creates a new structure-based random recommendation system
func NewStructureRandom(model *model.HKModel, steepness, noiseStd, randomRatio float64, matrixInit bool, logFunc func(string)) *StructureRandom {
	return &StructureRandom{
		Structure: Structure{
			Model:      model,
			NoiseStd:   noiseStd,
			MatrixInit: matrixInit,
			LogFunc:    logFunc,
		},
		Steepness:   steepness,
		RandomRatio: randomRatio,
	}
}

// PreStep implements model.HKModelRecommendationSystem
func (s *StructureRandom) PreStep() {
	// Call parent PreStep to create raw rate matrix
	s.Structure.PreStep()

	// Apply steepness
	if s.Steepness != 1 {
		for i := range s.NumNodes {
			for j := range s.NumNodes {
				s.RateMat[i][j] = math.Pow(s.RateMat[i][j], s.Steepness)
			}
		}
	}

	// Normalize rate matrix
	for i := range s.NumNodes {
		sum := 0.0
		for j := range s.NumNodes {
			sum += s.RateMat[i][j]
		}

		if sum > 0 {
			for j := range s.NumNodes {
				// Normalize and add random component if needed
				if s.RandomRatio > 0 {
					s.RateMat[i][j] = (1-s.RandomRatio)*s.RateMat[i][j]/sum +
						s.RandomRatio/(float64(s.NumNodes)-1)
				} else {
					s.RateMat[i][j] = s.RateMat[i][j] / sum
				}
			}
		}
		s.RateMat[i][i] = 0 // No self-recommendation
	}
}

// Recommend implements model.HKModelRecommendationSystem
func (s *StructureRandom) Recommend(agent *model.HKAgent, neighbors []*model.HKAgent, count int) []*model.TweetRecord {
	// Create set of neighbor IDs
	neighborIDs := make(map[int64]bool)
	neighborIDs[agent.ID] = true
	for _, n := range neighbors {
		neighborIDs[n.ID] = true
	}

	// Create a copy of the rate vector
	rateVec := make([]float64, s.NumNodes)
	copy(rateVec, s.RateMat[agent.ID])

	// Set rate to 0 for neighbors
	sum := 0.0
	rateVec[agent.ID] = 0
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
	candidates := sampleWithoutReplacement(s.AllIndices, count+4, rateVec)

	// Collect tweets from selected agents
	ret := make([]*model.TweetRecord, 0, len(candidates))
	for _, idx := range candidates {
		if a, ok := s.AgentMap[int64(idx)]; ok && a.CurTweet != nil && a.CurTweet.AgentID != agent.ID {
			ret = append(ret, a.CurTweet)
			if len(ret) >= len(candidates) {
				break
			}
		}
	}

	return ret
}

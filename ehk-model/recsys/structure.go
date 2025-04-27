package recsys

import (
	"fmt"
	"log"
	"math/rand"
	"sort"
	"time"

	"ehk-model/model"
	"ehk-model/utils"

	"github.com/vmihailenco/msgpack/v5"
)

// Structure implements a recommendation system based on network structure
type Structure struct {
	model.BaseRecommendationSystem
	Model      *model.HKModel
	NoiseStd   float64
	MatrixInit bool
	LogFunc    func(string)

	NumNodes   int
	ConnMat    [][]int
	RateMat    [][]float64
	AllIndices []int
	AgentMap   map[int64]*model.HKAgent

	topKFinder *utils.TopKFinder
}

// NewStructure creates a structure-based recommendation system
func NewStructure(model *model.HKModel, noiseStd float64, matrixInit bool, logFunc func(string)) *Structure {
	ret := &Structure{
		Model:      model,
		NoiseStd:   noiseStd,
		MatrixInit: matrixInit,
		LogFunc:    logFunc,
	}
	// init top k finder
	// TODO add to model parameters
	ret.topKFinder = utils.NewTopKFinder(50)

	return ret
}

func (s *Structure) calcConnMatMult() {
	s.ConnMat = makeRawMat[int](s.NumNodes, s.NumNodes)

	adjMat := makeRawMat[int](s.NumNodes, s.NumNodes)

	for i := range s.NumNodes {
		nodes := s.Model.Graph.From(int64(i))
		for nodes.Next() {
			j := int(nodes.Node().ID())
			adjMat[i][j] = 1
			adjMat[j][i] = 1 // Symmetric
		}
	}

	// Calculate conn_mat = adjMat * adjMat
	for i := range s.NumNodes {
		for j := range s.NumNodes {
			for k := range s.NumNodes {
				s.ConnMat[i][j] += adjMat[i][k] * adjMat[k][j]
			}
		}
	}

	for i := range s.NumNodes {
		for j := 0; j <= i; j++ {
			s.ConnMat[i][j] = 0
		}
	}
}

func (s *Structure) calcConnMatGraph() {
	s.ConnMat = makeRawMat[int](s.NumNodes, s.NumNodes)

	for u := range s.NumNodes {
		for v := u + 1; v < s.NumNodes; v++ { // v > u
			s.ConnMat[u][v] = commonNeighborsCount(s.Model.Graph, u, v)
		}
	}
}

// PostInit implements model.HKModelRecommendationSystem
func (s *Structure) PostInit(dumpData []byte) {

	s.NumNodes = s.Model.Graph.Nodes().Len()
	s.AllIndices = make([]int, s.NumNodes)
	for i := range s.NumNodes {
		s.AllIndices[i] = i
	}

	// Build agent map
	s.AgentMap = make(map[int64]*model.HKAgent, s.NumNodes)
	for _, a := range s.Model.Schedule.Agents {
		s.AgentMap[a.ID] = a
	}

	// Load connection matrix if dumped
	if dumpData != nil {
		s.ConnMat = make([][]int, 0)
		err := msgpack.Unmarshal(dumpData, &s.ConnMat)
		if err == nil {
			if s.LogFunc != nil {
				s.LogFunc("Connection matrix loaded from dump data.")
			}
			return
		} else {
			panic("test")
		}
	}

	// Calculate full connection matrix
	tstart := time.Now()
	if s.MatrixInit {
		s.calcConnMatMult()
	} else {
		s.calcConnMatGraph()
	}
	tend := time.Now()
	if s.LogFunc != nil {
		s.LogFunc(fmt.Sprintf("Connection matrix generation costs %v", tend.Sub(tstart)))
	}

	// Initialize rate matrix
	s.RateMat = makeRawMat[float64](s.NumNodes, s.NumNodes)
}

// PreStep implements model.HKModelRecommendationSystem
func (s *Structure) PreStep() {

	// if s.Model.CurStep > 5 && s.Model.CurStep%5 != 0 {
	// 	return
	// }

	// Create raw rate matrix from connection matrix
	rawRateMat := make([][]float64, s.NumNodes)
	for i := range rawRateMat {
		rawRateMat[i] = make([]float64, s.NumNodes)
		for j := range s.NumNodes {
			rawRateMat[i][j] = float64(s.ConnMat[i][j] + s.ConnMat[j][i])

			// Add noise if needed
			if s.NoiseStd > 0 {
				noise := rand.NormFloat64() * s.NoiseStd
				rawRateMat[i][j] = max(rawRateMat[i][j]*(1-2*noise)+noise, 0)
			}
		}
		rawRateMat[i][i] = 0 // No self-loops
	}

	// We don't apply steepness here - it's handled in derived classes
	s.RateMat = rawRateMat
}

// PostStep implements model.HKModelRecommendationSystem
func (s *Structure) PostStep(changed []*model.RewiringEventBody) {
	// Update connection matrix for changed nodes
	if len(changed) == 0 {
		return
	}

	// Get unique node IDs that were involved in rewiring
	changedNodes := make(map[int64]bool)
	for _, event := range changed {
		changedNodes[event.Unfollow] = true
		changedNodes[event.Follow] = true
	}

	// Convert to sorted slice
	nodeSlice := make([]int, 0, len(changedNodes))
	for node := range changedNodes {
		nodeSlice = append(nodeSlice, int(node))
	}
	sort.Ints(nodeSlice)

	// Update connection matrix for pairs of changed nodes
	for i := range nodeSlice {
		for j := i + 1; j < len(nodeSlice); j++ {
			u := nodeSlice[i]
			v := nodeSlice[j]
			if u < v {
				s.ConnMat[u][v] = commonNeighborsCount(s.Model.Graph, u, v)
			} else {
				s.ConnMat[v][u] = commonNeighborsCount(s.Model.Graph, v, u)
			}
		}
	}
}

// Recommend implements model.HKModelRecommendationSystem
func (s *Structure) Recommend(agent *model.HKAgent, neighbors []*model.HKAgent, count int) []*model.TweetRecord {
	// Create set of neighbor IDs
	neighborIDs := make(map[int64]bool)
	neighborIDs[agent.ID] = true
	for _, n := range neighbors {
		neighborIDs[n.ID] = true
	}

	// Get rate vector for agent
	rateVec := s.RateMat[agent.ID]

	// find largest elements
	candidates := s.topKFinder.FindTopK(rateVec, len(neighbors)+count+1)

	// Take top count
	numToTake := min(count, len(candidates))

	// Get tweets from selected agents
	ret := make([]*model.TweetRecord, 0, numToTake)
	for i := range numToTake {
		if a, ok := s.AgentMap[int64(candidates[i])]; ok {
			if a.CurTweet != nil && a.CurTweet.AgentID != agent.ID {
				ret = append(ret, a.CurTweet)
			}
		}
	}

	return ret
}

// Dump implements model.HKModelRecommendationSystem
func (s *Structure) Dump() []byte {
	data, err := msgpack.Marshal(s.ConnMat)
	if err != nil {
		log.Fatalf("Failed to marshal structure recsys dump data: %v", err)
	}
	return data
}

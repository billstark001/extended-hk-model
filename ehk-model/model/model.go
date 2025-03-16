package model

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/graph/simple"
)

// HKModelParams contains configuration parameters for the HK model
type HKModelParams struct {
	Tolerance        float64
	Decay            float64
	RewiringRate     float64
	RetweetRate      float64
	RecsysCount      int
	TweetRetainCount int
	RecsysFactory    func(*HKModel) HKModelRecommendationSystem
}

// DefaultHKModelParams creates a new parameters struct with default values
func DefaultHKModelParams() *HKModelParams {
	return &HKModelParams{
		Tolerance:        0.25,
		Decay:            1.0,
		RewiringRate:     0.1,
		RetweetRate:      0.3,
		RecsysCount:      10,
		TweetRetainCount: 3,
	}
}

// ToMap converts the parameters to a map
func (p *HKModelParams) ToMap() map[string]any {
	return map[string]any{
		"tolerance":          p.Tolerance,
		"decay":              p.Decay,
		"rewiring_rate":      p.RewiringRate,
		"retweet_rate":       p.RetweetRate,
		"recsys_count":       p.RecsysCount,
		"tweet_retain_count": p.TweetRetainCount,
	}
}

// HKModel represents the Hegselmann-Krause model
type HKModel struct {
	Graph        *simple.DirectedGraph
	Params       *HKModelParams
	Recsys       HKModelRecommendationSystem
	CollectItems map[string]bool
	EventLogger  func(map[string]any)
	CurStep      int
	Grid         *NetworkGrid
	Schedule     *RandomActivation
}

// NewHKModel creates a new HK model
func NewHKModel(
	g *simple.DirectedGraph,
	opinions []float64,
	params *HKModelParams,
	collectItems []string,
	eventLogger func(map[string]any),
	dumpData any,
) *HKModel {
	// Use default params if none provided
	if params == nil {
		params = DefaultHKModelParams()
	}

	// Initialize collection items
	collectMap := make(map[string]bool)
	for _, item := range collectItems {
		collectMap[item] = true
	}

	model := &HKModel{
		Graph:        g,
		Params:       params,
		CollectItems: collectMap,
		EventLogger:  eventLogger,
		CurStep:      0,
	}

	// Initialize grid and scheduler
	model.Grid = NewNetworkGrid(g)
	model.Schedule = NewRandomActivation(model)

	// Initialize recommendation system if factory is provided
	if params.RecsysFactory != nil {
		model.Recsys = params.RecsysFactory(model)
	}

	// Initialize agents
	nodes := g.Nodes()
	i := 0
	for nodes.Next() {
		nodeID := nodes.Node().ID()
		var opinion float64
		if i < len(opinions) {
			opinion = opinions[i]
		} else {
			opinion = rand.Float64()*2 - 1 // Random between -1 and 1
		}

		agent := NewHKAgent(int(nodeID), model, &opinion)
		model.Grid.PlaceAgent(agent, nodeID)
		model.Schedule.AddAgent(agent)
		i++
	}

	// Post-initialization for recommendation system
	if model.Recsys != nil {
		model.Recsys.PostInit(dumpData)
	}

	return model
}

// Dump returns data from the recommendation system
func (m *HKModel) Dump() any {
	if m.Recsys != nil {
		return m.Recsys.Dump()
	}
	return nil
}

// HasCollectionItem checks if an item is in the collection list
func (m *HKModel) HasCollectionItem(item string) bool {
	return m.CollectItems[item]
}

// Step advances the model by one time step
func (m *HKModel) Step() (int, float64) {
	// Pre-step actions for recommendation system
	if m.Recsys != nil {
		m.Recsys.PreStep()
	}

	// Execute agent steps
	m.Schedule.Step()

	// Pre-commit actions for recommendation system
	if m.Recsys != nil {
		m.Recsys.PreCommit()
	}

	// Collect changed nodes
	changed := make([]int, 0)
	changedCount := 0
	changedOpinionMax := 0.0

	// Apply changes from agents
	for _, agent := range m.Schedule.Agents {
		// Opinion change
		changedOpinion := agent.NextOpinion - agent.CurOpinion
		agent.CurOpinion = agent.NextOpinion
		changedOpinionMax = math.Max(changedOpinionMax, math.Abs(changedOpinion))

		// Add tweet if there is one
		if agent.NextTweet != nil {
			m.Grid.AddTweet(int64(agent.UniqueID), *agent.NextTweet, m.Params.TweetRetainCount)
			agent.CurTweet = agent.NextTweet
		}

		// Rewiring
		if agent.NextFollow != nil {
			m.Graph.RemoveEdge(int64(agent.UniqueID), int64(agent.NextFollow.Unfollow))
			m.Graph.SetEdge(m.Graph.NewEdge(
				m.Graph.Node(int64(agent.UniqueID)),
				m.Graph.Node(int64(agent.NextFollow.Follow)),
			))
			changed = append(changed,
				agent.UniqueID,
				agent.NextFollow.Unfollow,
				agent.NextFollow.Follow,
			)
			changedCount++
		}
	}

	// Post-step actions for recommendation system
	if m.Recsys != nil {
		m.Recsys.PostStep(changed)
	}

	// Increment step counter
	m.CurStep++

	return changedCount, changedOpinionMax
}

// GetRecommendation gets recommendations for an agent
func (m *HKModel) GetRecommendation(agent *HKAgent, neighbors []TweetRecord) []TweetRecord {
	if m.Recsys == nil {
		return []TweetRecord{}
	}

	return m.Recsys.Recommend(agent, neighbors, m.Params.RecsysCount)
}

// CollectOpinions collects all agent opinions
func (m *HKModel) CollectOpinions() map[int]float64 {
	opinions := make(map[int]float64)
	for _, agent := range m.Schedule.Agents {
		opinions[agent.UniqueID] = agent.CurOpinion
	}
	return opinions
}

// CollectTweets collects all current tweets
func (m *HKModel) CollectTweets() []TweetRecord {
	var tweets []TweetRecord
	for _, agent := range m.Schedule.Agents {
		if agent.CurTweet != nil {
			tweets = append(tweets, *agent.CurTweet)
		}
	}
	return tweets
}

// CollectGraph collects the current graph structure
func (m *HKModel) CollectGraph() *simple.DirectedGraph {
	return m.Graph
}

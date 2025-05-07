package model

import (
	"math"

	"gonum.org/v1/gonum/graph/simple"
)

type HKModelPureParams struct {
	RecsysCount      int
	TweetRetainCount int
}

type RecsysFactory func(*HKModel) HKModelRecommendationSystem

type HKModelParams struct {
	HKModelPureParams
	RecsysFactory RecsysFactory
}

type CollectItemOptions struct {
	AgentNumber     bool
	OpinionSum      bool
	RewiringEvent   bool
	ViewTweetsEvent bool
	TweetEvent      bool
}

func DefaultHKModelParams() *HKModelParams {
	ret := &HKModelParams{
		HKModelPureParams: HKModelPureParams{
			RecsysCount:      10,
			TweetRetainCount: 3,
		},
	}
	ret.RecsysCount = 10
	ret.TweetRetainCount = 3
	return ret
}

// HKModel represents the Hegselmann-Krause model
type HKModel struct {
	// params
	AgentParams  *HKAgentParams
	ModelParams  *HKModelParams
	CollectItems *CollectItemOptions
	// state
	Graph *simple.DirectedGraph
	Grid  *NetworkGrid

	// the smallest step that the data is not intact
	// equates the current step during step
	// will be incremented after the simulation step ends
	// starts from 1
	CurStep int

	// utils(?)
	Recsys      HKModelRecommendationSystem
	Schedule    *RandomActivation
	EventLogger func(*EventRecord)
}

// NewHKModel creates a new HK model
func NewHKModel(
	graph *simple.DirectedGraph,
	opinions *[]float64,
	modelParams *HKModelParams,
	agentParams *HKAgentParams,
	collectItems *CollectItemOptions,
	eventLogger func(*EventRecord),
) *HKModel {
	// Use default params if none provided
	if modelParams == nil {
		modelParams = DefaultHKModelParams()
	}
	if agentParams == nil {
		agentParams = DefaultHKAgentParams()
	}

	// Initialize struct
	model := &HKModel{
		Graph:        graph,
		ModelParams:  modelParams,
		AgentParams:  agentParams,
		CollectItems: collectItems,
		EventLogger:  eventLogger,
		CurStep:      0,
	}

	// Initialize grid and scheduler
	model.Grid = NewNetworkGrid(graph)
	model.Schedule = NewRandomActivation(model)

	// Initialize recommendation system if factory is provided
	if modelParams.RecsysFactory != nil {
		model.Recsys = modelParams.RecsysFactory(model)
	}

	// Initialize agents
	nodes := graph.Nodes()
	opinionsVal := make([]float64, 0)
	if opinions != nil {
		opinionsVal = *opinions
	}
	i := 0
	for nodes.Next() {
		nodeID := nodes.Node().ID()

		var opinion *float64
		if i < len(opinionsVal) {
			i2 := opinionsVal[int64(i)]
			opinion = &i2
		}

		agent := NewHKAgent(nodeID, model, opinion)
		model.Grid.PlaceAgent(agent, nodeID)
		model.Schedule.AddAgent(agent)
		i++
	}

	// Post-initialization for recommendation system
	return model
}

func (m *HKModel) SetAgentCurTweets() {
	cntTweetRetain := max(m.ModelParams.TweetRetainCount, 1)
	for aid, a := range m.Grid.AgentMap {
		if a.CurTweet == nil {
			l := len(m.Grid.TweetMap[aid])
			if l == 0 {
				// if no existent tweets, create one
				m.Grid.AddTweet(aid, &TweetRecord{
					AgentID: aid,
					Opinion: a.CurOpinion,
					Step:    -1,
				}, cntTweetRetain)
				l = 1
			}
			// apply the latest one
			a.CurTweet = m.Grid.TweetMap[aid][l-1]
		}
	}
}

// Step advances the model by one time step
func (m *HKModel) Step(doIncrementCurStep bool) (int, float64) {
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
	changed := make([]*RewiringEventBody, 0)
	changedCount := 0
	changedOpinionMax := 0.0

	// Apply changes from agents
	cntTweetRetain := max(m.ModelParams.TweetRetainCount, 1)
	for _, a := range m.Schedule.Agents {
		// Opinion change
		changedOpinion := a.NextOpinion - a.CurOpinion
		a.CurOpinion = a.NextOpinion
		changedOpinionMax = math.Max(changedOpinionMax, math.Abs(changedOpinion))

		// Add tweet if there is one
		if a.NextTweet != nil {
			m.Grid.AddTweet(a.ID, a.NextTweet, cntTweetRetain)
			a.CurTweet = a.NextTweet
		}

		// Rewiring
		if a.NextFollow != nil {
			if !m.Graph.HasEdgeFromTo(a.ID, a.NextFollow.Unfollow) {
				panic("BUG: wrong rewiring parameters")
			}
			// only rewires if the edge is inexistent
			if !m.Graph.HasEdgeFromTo(a.ID, a.NextFollow.Follow) {
				m.Graph.RemoveEdge(a.ID, a.NextFollow.Unfollow)
				m.Graph.SetEdge(m.Graph.NewEdge(
					m.Graph.Node(a.ID),
					m.Graph.Node(a.NextFollow.Follow),
				))
				changed = append(changed, a.NextFollow)
				changedCount++
			}
		}
	}

	// Post-step actions for recommendation system
	if m.Recsys != nil {
		m.Recsys.PostStep(changed)
	}

	// lastly, increment step counter
	if doIncrementCurStep {
		m.CurStep++
	}

	return changedCount, changedOpinionMax
}

// GetRecommendation gets recommendations for an agent
func (m *HKModel) GetRecommendation(agent *HKAgent, neighbors []*HKAgent) []*TweetRecord {
	if m.Recsys == nil {
		return []*TweetRecord{}
	}

	return m.Recsys.Recommend(agent, neighbors, m.ModelParams.RecsysCount)
}

func (m *HKModel) CollectOpinions() []float64 {
	ret := make([]float64, len(m.Schedule.Agents))
	for _, agent := range m.Schedule.Agents {
		ret[agent.ID] = agent.CurOpinion
	}
	return ret
}

func (m *HKModel) CollectAgentNumbers() []AgentNumberRecord {
	ret := make([]AgentNumberRecord, len(m.Schedule.Agents))
	for _, agent := range m.Schedule.Agents {
		ret[agent.ID] = agent.AgentNumber
	}
	return ret
}

func (m *HKModel) CollectAgentOpinions() []AgentOpinionSumRecord {
	ret := make([]AgentOpinionSumRecord, len(m.Schedule.Agents))
	for _, agent := range m.Schedule.Agents {
		ret[agent.ID] = agent.OpinionSum
	}
	return ret
}

// CollectTweets collects all current tweets
func (m *HKModel) CollectTweets() map[int64][]TweetRecord {
	tweets := make(map[int64][]TweetRecord)
	for agent, value := range m.Grid.TweetMap {
		tweets[agent] = []TweetRecord{}
		for _, ptr := range value {
			tweets[agent] = append(tweets[agent], *ptr)
		}
	}
	return tweets
}

package model

import utils "ehk-model/utils"

type HKModelDumpData struct {
	CurStep          int
	Graph            utils.NetworkXGraph
	Opinions         []float64
	AgentNumbers     []AgentNumberRecord
	AgentOpinionSums []AgentOpinionSumRecord
	Tweets           map[int64][]TweetRecord
	RecsysDumpData   any // no pointer
}

func (m *HKModel) Dump() *HKModelDumpData {
	ret := &HKModelDumpData{
		CurStep:          m.CurStep,
		Graph:            *utils.SerializeGraph(m.Graph),
		Opinions:         m.CollectOpinions(),
		AgentNumbers:     m.CollectAgentNumbers(),
		AgentOpinionSums: m.CollectAgentOpinions(),
		Tweets:           m.CollectTweets(),
	}
	if m.Recsys != nil {
		ret.RecsysDumpData = m.Recsys.Dump()
	}
	return ret
}

func (d *HKModelDumpData) Load(
	modelParams *HKModelParams,
	agentParams *HKAgentParams,
	collectItems *CollectItemOptions,
	eventLogger func(*EventRecord),
) *HKModel {
	model := NewHKModel(
		utils.DeserializeGraph(&d.Graph),
		&d.Opinions,
		modelParams,
		agentParams,
		collectItems,
		eventLogger,
	)

	// recover agent numbers and opinion sums
	for i, agent := range model.Schedule.Agents {
		agent.AgentNumber = d.AgentNumbers[i]
		agent.OpinionSum = d.AgentOpinionSums[i]
	}

	// recover step
	model.CurStep = d.CurStep

	// recover tweets
	g := model.Grid
	for agent, value := range d.Tweets {
		g.TweetMap[agent] = []*TweetRecord{}
		for _, ptr := range value {
			g.TweetMap[agent] = append(g.TweetMap[agent], &ptr)
		}
	}

	// recover tweets
	model.SetAgentCurTweets()

	// recover dump data
	if model.Recsys != nil {
		// pointer is passed
		model.Recsys.PostInit(&d.RecsysDumpData)
	}

	return model
}

package simulation

import (
	model "ehk-model/model"
)

type AccumulativeModelState struct {
	// (step, agent)
	Opinions [][]float64
	// (step, agent, type)
	AgentNumbers [][][4]int
	// (step, agent, type)
	AgentOpinionSums [][][4]float64

	UnsafeTweetEvent int
}

func NewAccumulativeModelState() *AccumulativeModelState {
	return &AccumulativeModelState{
		Opinions:         make([][]float64, 0),
		AgentNumbers:     make([][][4]int, 0),
		AgentOpinionSums: make([][][4]float64, 0),
	}
}

func (s *AccumulativeModelState) accumulate(model model.HKModel) {
	s.Opinions = append(s.Opinions, model.CollectOpinions())
	s.AgentNumbers = append(s.AgentNumbers, model.CollectAgentNumbers())
	s.AgentOpinionSums = append(s.AgentOpinionSums, model.CollectAgentOpinions())
}

func (s *AccumulativeModelState) validate(model model.HKModel) bool {
	st := model.CurStep
	return len(s.Opinions) == st &&
		len(s.AgentNumbers) == st &&
		len(s.AgentOpinionSums) == st
}

package recsys

import (
	"ehk-model/model"
	"math/rand"
)

type Random struct {
	model.BaseRecommendationSystem
	model      *model.HKModel
	agentCount int
}

func NewRandom(
	model *model.HKModel,
) *Random {
	return &Random{
		model:      model,
		agentCount: model.Graph.Nodes().Len(),
	}
}

func (r *Random) Recommend(
	agent *model.HKAgent,
	neighbors []*model.HKAgent,
	count int,
) []*model.TweetRecord {

	generated := make(map[int]bool)
	generated[int(agent.ID)] = true
	for _, a := range neighbors {
		generated[int(a.ID)] = true
	}

	// collect results that are not in neighbors
	result := make([]*model.TweetRecord, 0)
	i := 0
	for len(result) < count {
		// avoid dead loop
		if i > count*10 {
			break
		}
		num := rand.Intn(r.agentCount)
		if !generated[num] {
			generated[num] = true
			el := r.model.Grid.AgentMap[int64(num)].CurTweet
			if el != nil && el.AgentID != agent.ID {
				result = append(result, el)
			}
		}
		i++
	}

	return result
}

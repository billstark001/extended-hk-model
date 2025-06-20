package recsys

import (
	"ehk-model/model"
	"math/rand"
)

type Random struct {
	model.BaseRecommendationSystem
	Model                *model.HKModel
	HistoricalTweetCount int
	AgentCount           int
}

func NewRandom(
	model *model.HKModel,
	historicalTweetCount *int,
) *Random {
	h := model.ModelParams.TweetRetainCount
	if historicalTweetCount != nil {
		h = *historicalTweetCount
	}
	return &Random{
		Model:                model,
		AgentCount:           model.Graph.Nodes().Len(),
		HistoricalTweetCount: h,
	}
}

func (r *Random) Recommend(
	agent *model.HKAgent,
	neighbors []*model.HKAgent,
	count int,
) []*model.TweetRecord {

	generated := make(map[int64]bool)
	generated[agent.ID] = true
	for _, a := range neighbors {
		generated[a.ID] = true
	}
	neighborIDs := make(map[int64]bool)
	neighborIDs[agent.ID] = true
	for _, n := range neighbors {
		neighborIDs[n.ID] = true
	}

	visibleTweets := r.Model.Grid.TweetMap

	// collect results that are not in neighbors
	result := make([]*model.TweetRecord, 0)
	i := 0
	for len(result) < count {
		// avoid dead loop
		if i > count*10 {
			break
		}
		agentPickedId := int64(rand.Intn(r.AgentCount))
		if !generated[agentPickedId] {
			// do not replace
			generated[agentPickedId] = true
			tweetPickedIndex := -1 // 0: newest
			if r.HistoricalTweetCount > 0 {
				tweetPickedIndex = rand.Intn(r.HistoricalTweetCount)
			}
			var el *model.TweetRecord
			if tweetPickedIndex != -1 && tweetPickedIndex < len(visibleTweets[agentPickedId]) {
				// since visibleTweets is declared as -1: newest, revert it
				el = visibleTweets[agentPickedId][len(visibleTweets[agentPickedId])-tweetPickedIndex-1]
			} else {
				el = r.Model.Grid.AgentMap[agentPickedId].CurTweet
			}
			// skip:
			cond := el != nil && // nil
				el.AgentID != agent.ID && // itself
				!neighborIDs[el.AgentID] // its neighbor
			if cond {
				result = append(result, el)
			}
		}
		i++
	}

	return result
}

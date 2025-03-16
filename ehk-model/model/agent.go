package model

import (
	"math"
	"math/rand"
)

// FollowEventRecord represents an unfollow and follow event
type FollowEventRecord struct {
	Unfollow int
	Follow   int
}

// HKAgent represents an agent in the Hegselmann-Krause model
type HKAgent struct {
	UniqueID    int
	Model       *HKModel
	CurOpinion  float64
	CurTweet    *TweetRecord
	NextOpinion float64
	NextTweet   *TweetRecord
	NextFollow  *FollowEventRecord
	NrAgents    [4]int
	OpSumAgents [4]float64
	StepDict    map[string]any
}

// NewHKAgent creates a new HKAgent
func NewHKAgent(uniqueID int, model *HKModel, opinion *float64) *HKAgent {
	// Initialize opinion if not provided
	curOpinion := rand.Float64()*2 - 1 // Random between -1 and 1
	if opinion != nil {
		curOpinion = *opinion
	}

	agent := &HKAgent{
		UniqueID:    uniqueID,
		Model:       model,
		CurOpinion:  curOpinion,
		NextOpinion: curOpinion,
	}

	// Setup reporting flags
	hasEventLogger := model.EventLogger != nil
	reportViewTweets := hasEventLogger && model.HasCollectionItem("e:view_tweets")
	reportRewiring := hasEventLogger && model.HasCollectionItem("e:rewiring")
	reportRetweet := hasEventLogger && model.HasCollectionItem("e:retweet")

	// Initialize step configuration
	agent.StepDict = map[string]any{
		"decay":                 model.Params.Decay,
		"gamma":                 model.Params.RewiringRate,
		"tolerance":             model.Params.Tolerance,
		"r_retweet":             model.Params.RetweetRate,
		"collect_nr_agents":     model.HasCollectionItem("nr_agents"),
		"collect_op_sum_agents": model.HasCollectionItem("op_sum_agents"),
		"report_view_tweets":    reportViewTweets,
		"report_retweet":        reportRetweet,
		"report_rewiring":       reportRewiring,
	}

	return agent
}

// PartitionTweets divides tweets into concordant and discordant groups
func PartitionTweets(
	opinion float64,
	neighbors []TweetRecord,
	recommended []TweetRecord,
	epsilon float64,
) (
	concordantNeighbor []TweetRecord,
	concordantRecommended []TweetRecord,
	discordantNeighbor []TweetRecord,
	discordantRecommended []TweetRecord,
	sumN float64,
	sumR float64,
	sumND float64,
	sumRD float64,
) {
	// Prepare return values
	concordantNeighbor = []TweetRecord{}
	concordantRecommended = []TweetRecord{}
	discordantNeighbor = []TweetRecord{}
	discordantRecommended = []TweetRecord{}

	// Process neighbors
	for _, a := range neighbors {
		o := a.Opinion
		if math.Abs(opinion-o) <= epsilon {
			concordantNeighbor = append(concordantNeighbor, a)
			sumN += o - opinion
		} else {
			discordantNeighbor = append(discordantNeighbor, a)
			sumND += o - opinion
		}
	}
	if neighbors != nil {
	}

	// Process recommended
	for _, a := range recommended {
		o := a.Opinion
		if math.Abs(opinion-o) <= epsilon {
			concordantRecommended = append(concordantRecommended, a)
			sumR += o - opinion
		} else {
			discordantRecommended = append(discordantRecommended, a)
			sumRD += o - opinion
		}
	}

	return
}

// HKAgentStep implements the agent's step function
func HKAgentStep(
	uid int,
	opinion float64,
	curStep int,
	decay float64,
	gamma float64,
	tolerance float64,
	rRetweet float64,
	neighbors []TweetRecord,
	recommended []TweetRecord,
	collectNrAgents bool,
	collectOpSumAgents bool,
	reportViewTweets bool,
	reportRetweet bool,
	reportRewiring bool,
) (
	nextOpinion float64,
	nextTweet *TweetRecord,
	nextFollow *FollowEventRecord,
	nrAgents [4]int,
	opSumAgents [4]float64,
	eViewTweets [][]TweetRecord,
	eRetweet *TweetRecord,
	eRewiring *FollowEventRecord,
) {
	// Default return values
	nextOpinion = opinion
	var zeroAgents [4]int
	var zeroOpSum [4]float64
	nrAgents = zeroAgents
	opSumAgents = zeroOpSum

	// Calculate tweet sets
	concordantNeighbor, concordantRecommended,
		discordantNeighbor, discordantRecommended,
		sumN, sumR, sumND, sumRD := PartitionTweets(
		opinion, neighbors, recommended, tolerance,
	)

	nNeighbor := len(concordantNeighbor)
	nRecommended := len(concordantRecommended)
	nConcordant := nNeighbor + nRecommended

	// Collect agent counts if requested
	if collectNrAgents {
		nrAgents = [4]int{
			nNeighbor,
			nRecommended,
			len(discordantNeighbor),
			len(discordantRecommended),
		}
	}

	// Record viewed tweets if requested
	if reportViewTweets {
		eViewTweets = [][]TweetRecord{
			concordantNeighbor,
			concordantRecommended,
			discordantNeighbor,
			discordantRecommended,
		}
	}

	// Calculate influence
	if nConcordant > 0 {
		nextOpinion += ((sumR + sumN) / float64(nConcordant)) * decay
	}

	// Collect opinion sums if requested
	if collectOpSumAgents {
		opSumAgents = [4]float64{sumN, sumR, sumND, sumRD}
	}

	// Generate random numbers for retweet and rewiring decisions
	rndRetweet := rand.Float64()
	rndRewiring := rand.Float64()

	// Handle tweet or retweet
	if nNeighbor > 0 && rndRetweet < rRetweet { // Retweet
		retweetIndex := int(float64(nConcordant)*rndRetweet/rRetweet) % nConcordant
		var retweetRecord TweetRecord
		if retweetIndex < nNeighbor {
			retweetRecord = concordantNeighbor[retweetIndex]
		} else {
			retweetRecord = concordantRecommended[retweetIndex-nNeighbor]
		}
		nextTweet = &retweetRecord

		if reportRetweet {
			eRetweet = nextTweet
		}
	} else { // Post new tweet
		tweetRecord := TweetRecord{uid, curStep, nextOpinion}
		nextTweet = &tweetRecord
	}

	// Handle rewiring
	if gamma > 0 &&
		len(discordantNeighbor) > 0 && len(concordantRecommended) > 0 &&
		rndRewiring < gamma {
		idx1 := rand.Intn(len(concordantRecommended))
		idx2 := rand.Intn(len(discordantNeighbor))
		follow := concordantRecommended[idx1].UserID
		unfollow := discordantNeighbor[idx2].UserID

		nextFollow = &FollowEventRecord{
			Unfollow: unfollow,
			Follow:   follow,
		}

		if reportRewiring {
			eRewiring = nextFollow
		}
	}

	return
}

// Step performs a single step for this agent
func (a *HKAgent) Step() {
	// Get the neighbors
	neighbors := a.Model.Grid.GetNeighbors(a.UniqueID, false)
	recommended := a.Model.GetRecommendation(a, neighbors)

	// Call agent step function
	nextOpinion, nextTweet, nextFollow, nrAgents, opSumAgents,
		eViewTweets, eRetweet, eRewiring := HKAgentStep(
		a.UniqueID,
		a.CurOpinion,
		a.Model.CurStep,
		a.StepDict["decay"].(float64),
		a.StepDict["gamma"].(float64),
		a.StepDict["tolerance"].(float64),
		a.StepDict["r_retweet"].(float64),
		neighbors,
		recommended,
		a.StepDict["collect_nr_agents"].(bool),
		a.StepDict["collect_op_sum_agents"].(bool),
		a.StepDict["report_view_tweets"].(bool),
		a.StepDict["report_retweet"].(bool),
		a.StepDict["report_rewiring"].(bool),
	)

	// Update agent state
	a.NextOpinion = nextOpinion
	a.NextTweet = nextTweet
	a.NextFollow = nextFollow
	a.NrAgents = nrAgents
	a.OpSumAgents = opSumAgents

	// Report events if required
	if a.Model.EventLogger != nil {
		if eViewTweets != nil {
			a.Model.EventLogger(map[string]any{
				"name": "view_tweets",
				"uid":  a.UniqueID,
				"step": a.Model.CurStep,
				"body": eViewTweets,
			})
		}
		if eRetweet != nil {
			a.Model.EventLogger(map[string]any{
				"name": "retweet",
				"uid":  a.UniqueID,
				"step": a.Model.CurStep,
				"body": eRetweet,
			})
		}
		if eRewiring != nil {
			a.Model.EventLogger(map[string]any{
				"name": "rewiring",
				"uid":  a.UniqueID,
				"step": a.Model.CurStep,
				"body": eRewiring,
			})
		}
	}
}

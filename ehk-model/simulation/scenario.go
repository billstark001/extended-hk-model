package simulation

import (
	"ehk-model/model"
	"ehk-model/utils"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/schollz/progressbar/v3"
)

type Scenario struct {
	dir        string
	metadata   *ScenarioMetadata
	model      *model.HKModel
	acc        *AccumulativeModelState
	serializer *SimulationSerializer
	db         *EventDB
}

func NewScenario(dir string, metadata *ScenarioMetadata) *Scenario {
	return &Scenario{
		dir:        dir,
		metadata:   metadata,
		serializer: NewSimulationSerializer(dir, metadata.UniqueName, 3),
	}
}

var RECSYS_FACTORY = GetDefaultRecsysFactoryDefs()

const MAX_TWEET_EVENT_INTERVAL = 500

func (s *Scenario) Init() {

	// TODO create parametrized graph initializer
	nodeCount := 500
	graph := utils.CreateRandomNetwork(
		nodeCount,
		15./(float64(nodeCount)-1),
	)

	// initialize model

	factory := RECSYS_FACTORY[s.metadata.RecsysFactoryType]
	modelParams := model.HKModelParams{
		HKModelPureParams: s.metadata.HKModelPureParams,
		RecsysFactory:     factory,
	}

	m := model.NewHKModel(
		graph,
		nil,
		&modelParams,
		&s.metadata.HKAgentParams,
		&s.metadata.CollectItemOptions,
		s.logEvent,
	)
	m.SetAgentCurTweets()
	if m.Recsys != nil {
		m.Recsys.PostInit(nil)
	}

	s.model = m

	// initialize accumulative record

	s.acc = NewAccumulativeModelState()

	db, err := OpenEventDB(filepath.Join(s.dir, s.metadata.UniqueName, "events.db"))
	if err != nil {
		log.Fatalf("Failed to create event db logger: %v", err)
	}

	s.db = db

}

func (s *Scenario) Load() bool {

	// initialize event db
	// TODO ensure dirs
	// TODO optimize acc state storage

	db, err := OpenEventDB(filepath.Join(s.dir, s.metadata.UniqueName, "events.db"))
	if err != nil {
		log.Fatalf("Failed to create event db logger: %v", err)
	}

	s.db = db

	// initialize model

	modelDump, err := s.serializer.GetLatestSnapshot()
	if err != nil {
		log.Fatalf("Failed to load model dump: %v", err)
		return false
	}

	if modelDump == nil {
		return false
	}

	factory := RECSYS_FACTORY[s.metadata.RecsysFactoryType]
	modelParams := model.HKModelParams{
		HKModelPureParams: s.metadata.HKModelPureParams,
		RecsysFactory:     factory,
	}
	s.model = modelDump.Load(
		&modelParams,
		&s.metadata.HKAgentParams,
		&s.metadata.CollectItemOptions,
		s.logEvent,
	)

	// initialize accumulative record

	acc, err := s.serializer.GetLatestAccumulativeState()
	if err != nil {
		log.Fatalf("Failed to load accumulative state: %v", err)
		return false
	} else {
		validated := acc.validate((*s.model))
		if !validated {
			log.Fatalf("Accumulative state validation failed")
			return false
		}
	}

	s.acc = acc

	return true
}

func (s *Scenario) Dump() {
	s.serializer.SaveSnapshot(s.model.Dump())
	s.serializer.SaveAccumulativeState(s.acc)
}

func (s *Scenario) Step() (int, float64) {
	changedCount, maxOpinionChange := s.model.Step()

	// event is naturally logged

	// log accumulative state
	s.acc.accumulate(*s.model)
	s.acc.UnsafeTweetEvent += changedCount

	// log graph if necessary
	if s.acc.UnsafeTweetEvent > MAX_TWEET_EVENT_INTERVAL {
		s.serializer.SaveGraph(utils.SerializeGraph(s.model.Graph), s.model.CurStep)
		s.acc.UnsafeTweetEvent = 0
	}

	return changedCount, maxOpinionChange
}

const MAX_SIM_COUNT = 15000
const NETWORK_CHANGE_THRESHOLD = 1
const OPINION_CHANGE_THRESHOLD = 1e-6
const STOP_SIM_STEPS = 60
const SAVE_INTERVAL = 300 // seconds

func (s *Scenario) StepTillEnd() {
	bar := progressbar.Default(MAX_SIM_COUNT)
	bar.Set(s.model.CurStep)

	lastSaveTime := time.Now()
	successiveThresholdMet := 0

	for s.model.CurStep < MAX_SIM_COUNT {

		// step
		nwChange, opChange := s.Step()
		bar.Set(s.model.CurStep)

		// if threshold is met, end in prior
		thresholdMet := nwChange < NETWORK_CHANGE_THRESHOLD &&
			opChange < OPINION_CHANGE_THRESHOLD
		if thresholdMet {
			successiveThresholdMet++
		} else {
			successiveThresholdMet = 0
		}
		if successiveThresholdMet > STOP_SIM_STEPS {
			break
		}

		// save at fixed interval
		timeInterval := time.Since(lastSaveTime)
		if timeInterval.Seconds() >= SAVE_INTERVAL {
			lastSaveTime = time.Now()
			s.Dump()
		}

	}

	// finally save everything
	s.Dump()
	s.serializer.SaveGraph(utils.SerializeGraph(s.model.Graph), s.model.CurStep)
}

func (s *Scenario) logEvent(event *model.EventRecord) {

	// TODO add to database
	switch event.Type {

	case "Tweet1":
		body := event.Body.(model.TweetEventBody)
		if body.IsRetweet {
			fmt.Printf(
				"Agent %d retweeted (tweet from Agent %d at step %d) at step %d\n",
				event.AgentID, body.Record.AgentID, body.Record.Step, event.Step,
			)
		} else {
			fmt.Printf(
				"Agent %d tweeted at step %d\n",
				event.AgentID, event.Step,
			)
		}

	case "Rewiring1":
		body := event.Body.(model.RewiringEventBody)
		fmt.Printf(
			"Agent %d unfollows %d and follows %d at step %d\n",
			event.AgentID, body.Unfollow, body.Follow, event.Step,
		)
	}

}

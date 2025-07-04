package simulation

import (
	"context"
	"ehk-model/model"
	"ehk-model/utils"
	"fmt"
	"log"
	"os"
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
		serializer: NewSimulationSerializer(dir, metadata.UniqueName, 2),
	}
}

var RECSYS_FACTORY = GetDefaultRecsysFactoryDefs()

const MAX_TWEET_EVENT_INTERVAL = 500
const DB_CACHE_SIZE = 40000

func (s *Scenario) Init() {

	nodeCount := max(s.metadata.NodeCount, 1)
	edgeCount := max(s.metadata.NodeFollowCount, 1)
	graph := utils.CreateRandomNetwork(
		nodeCount,
		float64(edgeCount)/(float64(nodeCount)-1),
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

	// create model record dump
	err := os.MkdirAll(
		filepath.Join(s.dir, s.metadata.UniqueName),
		0755,
	)
	if err != nil {
		log.Fatalf("Failed to create scenario dump folder: %v", err)
	}

	// initialize accumulative record

	s.acc = NewAccumulativeModelState()

	db, err := OpenEventDB(filepath.Join(s.dir, s.metadata.UniqueName, "events.db"), DB_CACHE_SIZE)
	if err != nil {
		log.Fatalf("Failed to create event db logger: %v", err)
	}

	s.db = db

	// write initial record
	s.serializer.SaveGraph(utils.SerializeGraph(s.model.Graph), s.model.CurStep)
	s.acc.accumulate(*s.model)
	s.model.CurStep = 1

	s.sanitize()
}

func (s *Scenario) Load() bool {

	// initialize event db
	dbPath := filepath.Join(s.dir, s.metadata.UniqueName, "events.db")
	_, err := os.Stat(dbPath)
	if os.IsNotExist(err) {
		// db inexistent
		return false
	}
	db, err := OpenEventDB(dbPath, DB_CACHE_SIZE)
	if err != nil {
		log.Printf("Failed to create event db logger: %v", err)
		return false
	}

	s.db = db

	// initialize model

	modelDump, err := s.serializer.GetLatestSnapshot()
	if err != nil {
		log.Printf("Failed to load model dump: %v", err)
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
		log.Printf("Failed to load accumulative state: %v", err)
		return false
	} else {
		validated := acc.validate((*s.model))
		if !validated {
			log.Printf("Accumulative state validation failed")
			return false
		}
	}

	s.acc = acc

	s.sanitize()

	return true
}

func (s *Scenario) sanitize() {
	// delete potentially dirty data
	// 'after': >=
	s.db.DeleteEventsAfterStep(s.model.CurStep)
	s.serializer.DeleteGraphsAfterStep(s.model.CurStep, false)
}

func (s *Scenario) Dump() {
	s.db.Flush()
	s.serializer.SaveSnapshot(s.model.Dump())
	s.serializer.SaveAccumulativeState(s.acc)
}

func (s *Scenario) Step() (int, float64) {
	changedCount, maxOpinionChange := s.model.Step(false)

	// event is naturally logged

	// log accumulative state
	s.acc.accumulate(*s.model)
	s.acc.UnsafeTweetEvent += changedCount

	// log graph if necessary
	if s.acc.UnsafeTweetEvent > MAX_TWEET_EVENT_INTERVAL {
		s.serializer.SaveGraph(utils.SerializeGraph(s.model.Graph), s.model.CurStep)
		s.acc.UnsafeTweetEvent = 0
	}

	// increase the counter manually
	// to ensure the graph records' step numbers stay consistent
	s.model.CurStep++

	return changedCount, maxOpinionChange
}

func (s *Scenario) IsFinished() bool {
	finished, _ := s.serializer.IsFinished()
	return finished
}

const NETWORK_CHANGE_THRESHOLD = 1
const OPINION_CHANGE_THRESHOLD = 1e-7
const STOP_SIM_STEPS = 60
const SAVE_INTERVAL = 300 // seconds

func (s *Scenario) StepTillEnd(ctx context.Context) {

	maxSimCount := s.metadata.MaxSimulationStep
	if maxSimCount < 0 {
		maxSimCount = 1
	}

	// if finished, jump this simulation
	if s.IsFinished() {
		return
	}

	bar := progressbar.Default(int64(maxSimCount))
	bar.Set(s.model.CurStep)

	lastSaveTime := time.Now()
	successiveThresholdMet := 0

	unitStep := func() (bool, bool) {

		didDump := false

		// step
		bar.Set(s.model.CurStep)
		nwChange, opChange := s.Step()

		// if threshold is met, end in prior
		thresholdMet := nwChange < NETWORK_CHANGE_THRESHOLD &&
			opChange < OPINION_CHANGE_THRESHOLD
		if thresholdMet {
			successiveThresholdMet++
		} else {
			successiveThresholdMet = 0
		}
		if successiveThresholdMet > STOP_SIM_STEPS {
			return false, didDump
		}

		// save at fixed interval
		timeInterval := time.Since(lastSaveTime)
		if timeInterval.Seconds() >= SAVE_INTERVAL {
			lastSaveTime = time.Now()
			s.Dump()
			didDump = true
		}

		return true, didDump

	}

	isCtxDone := false
	isShouldNotContinue := false
	didDump := false

iterLoop:
	for s.model.CurStep <= maxSimCount {
		select {
		case <-ctx.Done():
			isCtxDone = true
			break iterLoop

		default:
			didDump = false
			shouldContinue, _didDump := unitStep()
			didDump = _didDump

			if shouldContinue {
				// do nothing
			} else {
				isShouldNotContinue = true
				break iterLoop
			}
		}
	}

	// bar.Close()
	if s.model.CurStep <= maxSimCount {
		fmt.Println("")
	}

	if !didDump {
		s.Dump()
	}

	// st is the last step that has full simulation record
	st := s.model.CurStep - 1

	if isCtxDone {
		log.Printf("Simulation ended (`ctx.Done()` received, step: %d)", st)
		// the simulation is halted
		// do nothing
	} else {
		if !isShouldNotContinue {
			log.Printf("Simulation ended (max iteration reached), step: %d", st)
		} else {
			log.Printf("Simulation ended (shouldContinue == false, step: %d)", st)
		}
		// the simulation is finished
		s.serializer.MarkFinished()
		s.serializer.SaveGraph(utils.SerializeGraph(s.model.Graph), st)
	}

}

func (s *Scenario) logEvent(event *model.EventRecord) {

	// add to database when necessary

	switch event.Type {

	case "Tweet":
		body := event.Body.(model.TweetEventBody)
		if body.IsRetweet && s.metadata.CollectItemOptions.TweetEvent {
			s.db.StoreEvent(event)
		} else {
			// do nothing
		}

	case "Rewiring":
		if s.metadata.CollectItemOptions.RewiringEvent {
			s.db.StoreEvent(event)
		}

	case "ViewTweets":
		if s.metadata.CollectItemOptions.ViewTweetsEvent {
			s.db.StoreEvent(event)
		}

	}

}

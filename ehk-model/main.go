package main

import (
	"ehk-model/model"
	"ehk-model/simulation"
	"encoding/json"
	"log"
	"os"
)

func main() {
	metadata := &simulation.ScenarioMetadata{

		HKAgentParams: model.HKAgentParams{

			Decay:        0.01,
			Tolerance:    0.45,
			RewiringRate: 0.05,
			RetweetRate:  0.3,
		},

		HKModelPureParams: model.HKModelPureParams{

			TweetRetainCount: 3,
			RecsysCount:      10,
		},

		CollectItemOptions: model.CollectItemOptions{

			OpinionSum:    true,
			RewiringEvent: true,
			TweetEvent:    true,
		},

		RecsysFactoryType: "Random",
		NetworkType:       "Random",
		NodeCount:         500,
		NodeFollowCount:   15,

		UniqueName: "test",
	}

	args := os.Args
	basePath := args[1]
	metadataPath := args[2]
	metadataJson, err := os.ReadFile(metadataPath)
	if err != nil {
		log.Fatalf("Failed to load metadata file: %v", err)
	}

	// basePath := "./run"
	// metadataJson := []byte(`{}`)

	err = json.Unmarshal(metadataJson, metadata)
	if err != nil {
		log.Fatalf("Failed to unmarshal metadata file: %v", err)
	}

	scenario := simulation.NewScenario(basePath, metadata)

	if !scenario.Load() {
		scenario.Init()
	}

	scenario.StepTillEnd()
}

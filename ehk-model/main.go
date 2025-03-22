package main

import (
	"ehk-model/model"
	"ehk-model/simulation"
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

		UniqueName: "test",
	}

	scenario := simulation.NewScenario("./run", metadata)

	if !scenario.Load() {
		scenario.Init()
	}

	scenario.StepTillEnd()
}

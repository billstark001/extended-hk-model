package simulation

import (
	"ehk-model/model"
	"ehk-model/recsys"
	"fmt"
)

type ScenarioMetadata struct {
	UniqueName string

	model.HKAgentParams
	model.HKModelPureParams
	model.CollectItemOptions

	RecsysFactoryType string
	NetworkType       string // currently useless
	NodeCount         int
	NodeFollowCount   int
}

func GetDefaultRecsysFactoryDefs() map[string]model.RecsysFactory {
	ret := map[string]model.RecsysFactory{

		"Random": func(h *model.HKModel) model.HKModelRecommendationSystem {
			return recsys.NewRandom(h)
		},

		"Opinion": func(h *model.HKModel) model.HKModelRecommendationSystem {
			return recsys.NewOpinion(h, 0.1)
		},

		"Structure": func(h *model.HKModel) model.HKModelRecommendationSystem {
			return recsys.NewStructure(h, 0.1, true, func(s string) {
				fmt.Println(s)
			})
		},

		"OpinionRandom": func(h *model.HKModel) model.HKModelRecommendationSystem {
			return recsys.NewOpinionRandom(h, 0.4, 1, 2, 0)
		},

		"StructureRandom": func(h *model.HKModel) model.HKModelRecommendationSystem {
			return recsys.NewStructureRandom(h, 1, 0.1, 0, true, func(s string) {
				fmt.Println(s)
			})
		},
	}
	return ret
}

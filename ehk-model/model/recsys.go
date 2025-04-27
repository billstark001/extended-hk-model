package model

// HKModelRecommendationSystem defines the interface for recommendation systems
type HKModelRecommendationSystem interface {
	// PostInit is called after the model is initialized
	// dumpData is passed as pointer
	PostInit(dumpData []byte)

	// PreStep is called before each model step
	PreStep()

	// PreCommit is called before changes are committed
	PreCommit()

	// PostStep is called after each model step
	PostStep(changed []*RewiringEventBody)

	// Recommend returns recommendations for an agent
	Recommend(agent *HKAgent, neighbors []*HKAgent, count int) []*TweetRecord

	// Dump returns internal data for debugging/analysis
	Dump() []byte
}

type BaseRecommendationSystem struct {
	// do nothing, provide default empty methods
}

// for type check
func _() HKModelRecommendationSystem {
	return &BaseRecommendationSystem{}
}

func (rs *BaseRecommendationSystem) PostInit(dumpData []byte) {
}

func (rs *BaseRecommendationSystem) PreStep() {
}

func (rs *BaseRecommendationSystem) PreCommit() {
}

func (rs *BaseRecommendationSystem) PostStep(changed []*RewiringEventBody) {
}

func (rs *BaseRecommendationSystem) Recommend(agent *HKAgent, neighbors []*HKAgent, count int) []*TweetRecord {
	return []*TweetRecord{}
}

func (rs *BaseRecommendationSystem) Dump() []byte {
	return nil
}

package model

// HKModelRecommendationSystem defines the interface for recommendation systems
type HKModelRecommendationSystem interface {
	// PostInit is called after the model is initialized
	PostInit(dumpData any)

	// PreStep is called before each model step
	PreStep()

	// PreCommit is called before changes are committed
	PreCommit()

	// PostStep is called after each model step
	PostStep(changed []int)

	// Recommend returns recommendations for an agent
	Recommend(agent *HKAgent, neighbors []TweetRecord, count int) []TweetRecord

	// Dump returns internal data for debugging/analysis
	Dump() any
}

// SimpleRandomRecommendationSystem implements a basic recommendation system
type SimpleRandomRecommendationSystem struct {
	Model *HKModel
}

// NewSimpleRandomRecommendationSystem creates a new simple recommendation system
func NewSimpleRandomRecommendationSystem(model *HKModel) HKModelRecommendationSystem {
	return &SimpleRandomRecommendationSystem{
		Model: model,
	}
}

// PostInit implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) PostInit(dumpData any) {
	// Nothing to do for simple random recommender
}

// PreStep implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) PreStep() {
	// Nothing to do
}

// PreCommit implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) PreCommit() {
	// Nothing to do
}

// PostStep implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) PostStep(changed []int) {
	// Nothing to do
}

// Recommend implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) Recommend(agent *HKAgent, neighbors []TweetRecord, count int) []TweetRecord {
	// This is a placeholder for more sophisticated recommendation systems
	// In a real system, you would implement recommendation algorithms here
	return []TweetRecord{}
}

// Dump implements HKModelRecommendationSystem
func (rs *SimpleRandomRecommendationSystem) Dump() any {
	return map[string]string{
		"type": "SimpleRandomRecommendationSystem",
	}
}

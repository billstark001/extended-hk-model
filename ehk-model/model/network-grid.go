package model

import (
	"sync"

	"gonum.org/v1/gonum/graph/simple"
)

// NetworkGrid represents a network structure for placing agents
type NetworkGrid struct {
	Graph    *simple.DirectedGraph
	AgentMap map[int64]*HKAgent
	TweetMap map[int64][]TweetRecord
	mu       sync.RWMutex
}

// NewNetworkGrid creates a new network grid
func NewNetworkGrid(g *simple.DirectedGraph) *NetworkGrid {
	return &NetworkGrid{
		Graph:    g,
		AgentMap: make(map[int64]*HKAgent),
		TweetMap: make(map[int64][]TweetRecord),
	}
}

// PlaceAgent places an agent on the grid
func (ng *NetworkGrid) PlaceAgent(agent *HKAgent, nodeID int64) {
	ng.mu.Lock()
	defer ng.mu.Unlock()
	ng.AgentMap[nodeID] = agent
}

// GetAgent returns the agent at the specified node
func (ng *NetworkGrid) GetAgent(nodeID int64) *HKAgent {
	ng.mu.RLock()
	defer ng.mu.RUnlock()
	return ng.AgentMap[nodeID]
}

// AddTweet adds a tweet to the specified node
func (ng *NetworkGrid) AddTweet(nodeID int64, tweet TweetRecord, maxTweets int) {
	ng.mu.Lock()
	defer ng.mu.Unlock()
	tweets := ng.TweetMap[nodeID]

	// Add the new tweet
	tweets = append(tweets, tweet)

	// Limit the number of tweets per node
	if len(tweets) > maxTweets {
		tweets = tweets[len(tweets)-maxTweets:]
	}

	ng.TweetMap[nodeID] = tweets
}

// GetNeighbors returns the tweets from the neighbors of a node
func (ng *NetworkGrid) GetNeighbors(nodeID int, includeCenter bool) []TweetRecord {
	ng.mu.RLock()
	defer ng.mu.RUnlock()

	var result []TweetRecord

	// Get neighbor nodes
	neighbors := ng.Graph.From(int64(nodeID))
	for neighbors.Next() {
		neighborID := neighbors.Node().ID()
		if tweets, ok := ng.TweetMap[neighborID]; ok {
			result = append(result, tweets...)
		}
	}

	// Include center node's tweets if requested
	if includeCenter {
		if tweets, ok := ng.TweetMap[int64(nodeID)]; ok {
			result = append(result, tweets...)
		}
	}

	return result
}

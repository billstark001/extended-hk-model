package recsys

import (
	"math/rand"

	"gonum.org/v1/gonum/graph"
)

func makeRawMat[T any](h int, w int) [][]T {
	m := make([][]T, h)
	for i := range m {
		m[i] = make([]T, w)
	}
	return m
}

// sampleWithoutReplacement samples n items from population without replacement
// using the given probabilities
func sampleWithoutReplacement(population []int, n int, probabilities []float64) []int {
	if n >= len(population) {
		result := make([]int, len(population))
		copy(result, population)
		return result
	}

	// Make a weighted sampling using rejection method
	selected := make(map[int]bool)
	result := make([]int, 0, n)

	for len(result) < n {
		// Find a suitable index
		r := rand.Float64()
		cumSum := 0.0

		for i, p := range probabilities {
			cumSum += p
			if r < cumSum {
				if !selected[i] {
					selected[i] = true
					result = append(result, population[i])
				}
				break
			}
		}
	}

	return result
}

// commonNeighborsCount calculates the number of common neighbors between two nodes
func commonNeighborsCount(g graph.Directed, u, v int) int {
	uPred := nodesSet(g.To(int64(u)))
	uSucc := nodesSet(g.From(int64(u)))
	vPred := nodesSet(g.To(int64(v)))
	vSucc := nodesSet(g.From(int64(v)))

	count := 0

	// Count u's predecessors that are also v's predecessors or successors
	for w := range uPred {
		if vPred[w] || vSucc[w] {
			count++
		}
	}

	// Count u's successors that are also v's predecessors or successors
	for w := range uSucc {
		if vPred[w] || vSucc[w] {
			count++
		}
	}

	return count
}

// nodesSet converts a nodes iterator to a set (map)
func nodesSet(iter graph.Nodes) map[int64]bool {
	result := make(map[int64]bool)
	for iter.Next() {
		result[iter.Node().ID()] = true
	}
	return result
}

package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/graph/simple"

	ehk "ehk-model/model"
	"ehk-model/recsys"
	"ehk-model/utils"
)

func eventHandler(event *ehk.EventRecord) {

	switch event.Type {

	case "Tweet1":
		body := event.Body.(ehk.TweetEventBody)
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
		body := event.Body.(ehk.RewiringEventBody)
		fmt.Printf(
			"Agent %d unfollows %d and follows %d at step %d\n",
			event.AgentID, body.Unfollow, body.Follow, event.Step,
		)
	}

}

// printSimulationResults 打印仿真结果
func printSimulationResults(opinionHistory map[int64][]float64, tweetHistory [][]ehk.TweetRecord, graphHistory []*simple.DirectedGraph) {
	// 打印观点演化
	fmt.Println("\n=== Opinion Evolution ===")
	fmt.Println("Final opinions:")
	for agentID, opinions := range opinionHistory {
		if len(opinions) > 0 {
			initialOpinion := opinions[0]
			finalOpinion := opinions[len(opinions)-1]
			fmt.Printf("Agent %d: %.2f -> %.2f (change: %.2f)\n",
				agentID, initialOpinion, finalOpinion, finalOpinion-initialOpinion)
		}
	}

	// 计算观点极化程度
	fmt.Println("\n=== Opinion Polarization ===")
	if len(opinionHistory) > 0 {
		var firstAgentID int64 = 0
		for id := range opinionHistory {
			firstAgentID = id
			break
		}

		lastStep := len(opinionHistory[firstAgentID]) - 1
		if lastStep >= 0 {
			opinions := make([]float64, 0, len(opinionHistory))
			for _, agentOpinions := range opinionHistory {
				opinions = append(opinions, agentOpinions[lastStep])
			}

			// 计算最大和最小观点
			min, max := opinions[0], opinions[0]
			for _, op := range opinions {
				if op < min {
					min = op
				}
				if op > max {
					max = op
				}
			}

			fmt.Printf("Opinion range: %.2f to %.2f (spread: %.2f)\n", min, max, max-min)
		}
	}

	// 打印推文统计
	fmt.Println("\n=== Tweet Statistics ===")
	totalTweets := 0
	for _, tweets := range tweetHistory {
		totalTweets += len(tweets)
	}
	fmt.Printf("Total tweets: %d (avg %.2f per step)\n",
		totalTweets, float64(totalTweets)/float64(len(tweetHistory)))

	// 打印网络统计
	fmt.Println("\n=== Network Evolution ===")
	if len(graphHistory) > 0 {
		initialGraph := graphHistory[0]
		finalGraph := graphHistory[len(graphHistory)-1]

		initialEdges := initialGraph.Edges()
		initialEdgeCount := 0
		for initialEdges.Next() {
			initialEdgeCount++
		}

		finalEdges := finalGraph.Edges()
		finalEdgeCount := 0
		for finalEdges.Next() {
			finalEdgeCount++
		}

		fmt.Printf("Initial edge count: %d\n", initialEdgeCount)
		fmt.Printf("Final edge count: %d\n", finalEdgeCount)
		fmt.Printf("Edge change: %d\n", finalEdgeCount-initialEdgeCount)
	}
}

func main() {
	// 设置随机种子
	rand.Seed(time.Now().UnixNano())

	// 仿真参数
	nodeCount := 500
	simulationSteps := 5000
	// graphCollectionInterval := 10

	// 创建事件记录器
	eventLogger, err := ehk.NewEventLogger("simulation_events.msgpack", 1000)
	if err != nil {
		log.Fatalf("Failed to create event logger: %v", err)
	}
	defer eventLogger.Stop()

	// 创建网络
	fmt.Println("Creating small world network...")
	graph := utils.CreateRandomNetwork(
		nodeCount,
		15./(float64(nodeCount)-1),
	)

	// 生成初始观点
	initialOpinions := make([]float64, nodeCount)
	for i := range nodeCount {
		// 使用两种不同的观点群体，模拟极化的初始状态
		if i < nodeCount/2 {
			initialOpinions[i] = rand.Float64()*0.5 - 1.0 // -1.0 到 -0.5 之间
		} else {
			initialOpinions[i] = rand.Float64()*0.5 + 0.5 // 0.5 到 1.0 之间
		}
	}

	// 创建模型参数
	params := ehk.DefaultHKAgentParams()
	params.Decay = 0.01
	params.Tolerance = 0.45
	params.RewiringRate = 0.05
	params.RetweetRate = 0.3

	// 自定义推荐系统工厂
	modelParams := ehk.DefaultHKModelParams()
	modelParams.TweetRetainCount = 3
	modelParams.RecsysCount = 10
	modelParams.RecsysFactory = func(model *ehk.HKModel) ehk.HKModelRecommendationSystem {
		// return &ehk.BaseRecommendationSystem{}
		return recsys.NewRandom(model)
	}

	// 设置收集项
	collectItems := &ehk.CollectItemOptions{
		OpinionSum:    true,
		RewiringEvent: true,
		TweetEvent:    true,
	}

	// 创建事件处理函数
	logEvent := func(event *ehk.EventRecord) {
		eventLogger.LogEvent(event)
		eventHandler(event)
	}

	// 创建模型
	fmt.Println("Initializing HK model...")
	model := ehk.NewHKModel(
		graph,
		nil,
		modelParams,
		params,
		collectItems,
		logEvent,
	)
	model.SetAgentCurTweets()

	// 准备数据收集结构
	opinionHistory := make(map[int64][]float64)
	var tweetHistory [][]ehk.TweetRecord
	var graphHistory []*simple.DirectedGraph

	// 运行仿真
	fmt.Println("Running simulation...")
	for i := range simulationSteps {
		// 模型步进
		changedCount, maxOpinionChange := model.Step()

		// 进度报告
		fmt.Printf("Step %d/%d: %d agents changed connections, max opinion change: %.4f\n",
			i+1, simulationSteps, changedCount, maxOpinionChange)
	}

	// 打印结果
	printSimulationResults(opinionHistory, tweetHistory, graphHistory)

	// 打印推荐系统数据（如果适用）
	if model.Recsys != nil {
		fmt.Println("\n=== Recommendation System Data ===")
		fmt.Printf("%v\n", model.Dump())
	}

	fmt.Println("\nSimulation complete! Events saved to simulation_events.msgpack")
}

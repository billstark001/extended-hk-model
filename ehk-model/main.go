package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/graph/simple"

	ehk "ehk-model/model"
	utils "ehk-model/utils"
)

// eventHandler 处理模型生成的事件
func eventHandler(event map[string]interface{}) {
	// 在实际应用中，可能会将事件写入数据库或文件
	// 这里只是简单地记录事件名称和步骤
	eventName := event["name"].(string)
	step := event["step"].(int)
	uid := event["uid"].(int)

	switch eventName {
	case "view_tweets":
		// 处理查看推文事件
		// body 是 [][]TweetRecord
		fmt.Printf("Agent %d viewed tweets at step %d\n", uid, step)
	case "retweet":
		// 处理转发事件
		// body 是 TweetRecord
		fmt.Printf("Agent %d retweeted at step %d\n", uid, step)
	case "rewiring":
		// 处理重连事件
		// body 是 FollowEventRecord
		fmt.Printf("Agent %d changed following at step %d\n", uid, step)
	}
}

// opinionCollector 收集每个时间步的观点数据
func opinionCollector(model *ehk.HKModel, opinionHistory map[int][]float64) {
	opinions := model.CollectOpinions()
	for agentID, opinion := range opinions {
		opinionHistory[agentID] = append(opinionHistory[agentID], opinion)
	}
}

// tweetCollector 收集每个时间步的推文数据
func tweetCollector(model *ehk.HKModel, tweetHistory *[][]ehk.TweetRecord) {
	tweets := model.CollectTweets()
	*tweetHistory = append(*tweetHistory, tweets)
}

// graphCollector 每k步收集一次图结构
func graphCollector(model *ehk.HKModel, graphHistory *[]*simple.DirectedGraph, step int, interval int) {
	if step%interval == 0 {
		// 克隆当前图结构
		g := simple.NewDirectedGraph()
		nodes := model.Graph.Nodes()
		for nodes.Next() {
			node := nodes.Node()
			g.AddNode(simple.Node(node.ID()))
		}

		edges := model.Graph.Edges()
		for edges.Next() {
			edge := edges.Edge()
			g.SetEdge(g.NewEdge(
				simple.Node(edge.From().ID()),
				simple.Node(edge.To().ID()),
			))
		}

		*graphHistory = append(*graphHistory, g)
	}
}

// printSimulationResults 打印仿真结果
func printSimulationResults(opinionHistory map[int][]float64, tweetHistory [][]ehk.TweetRecord, graphHistory []*simple.DirectedGraph) {
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
		firstAgentID := 0
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
	nodeCount := 100
	simulationSteps := 50
	graphCollectionInterval := 10

	// 创建事件记录器
	eventLogger, err := ehk.NewEventLogger("simulation_events.msgpack", 1000)
	if err != nil {
		log.Fatalf("Failed to create event logger: %v", err)
	}
	defer eventLogger.Stop()

	// 创建网络
	fmt.Println("Creating small world network...")
	graph := utils.CreateSmallWorldNetwork(nodeCount, 4, 0.1)

	// 生成初始观点
	initialOpinions := make([]float64, nodeCount)
	for i := 0; i < nodeCount; i++ {
		// 使用两种不同的观点群体，模拟极化的初始状态
		if i < nodeCount/2 {
			initialOpinions[i] = rand.Float64()*0.5 - 1.0 // -1.0 到 -0.5 之间
		} else {
			initialOpinions[i] = rand.Float64()*0.5 + 0.5 // 0.5 到 1.0 之间
		}
	}

	// 创建模型参数
	params := ehk.DefaultHKModelParams()
	params.Tolerance = 0.3
	params.RewiringRate = 0.2
	params.RetweetRate = 0.4

	// 自定义推荐系统工厂
	params.RecsysFactory = func(model *ehk.HKModel) ehk.HKModelRecommendationSystem {
		return ehk.NewSimpleRandomRecommendationSystem(model)
	}

	// 设置收集项
	collectItems := []string{
		"nr_agents",
		"op_sum_agents",
		"e:view_tweets",
		"e:retweet",
		"e:rewiring",
	}

	// 创建事件处理函数
	logEvent := func(event map[string]interface{}) {
		eventLogger.LogEvent(event)
		eventHandler(event)
	}

	// 创建模型
	fmt.Println("Initializing HK model...")
	model := ehk.NewHKModel(
		graph,
		initialOpinions,
		params,
		collectItems,
		logEvent,
		nil,
	)

	// 准备数据收集结构
	opinionHistory := make(map[int][]float64)
	var tweetHistory [][]ehk.TweetRecord
	var graphHistory []*simple.DirectedGraph

	// 初始状态收集
	opinionCollector(model, opinionHistory)
	tweetCollector(model, &tweetHistory)
	graphCollector(model, &graphHistory, 0, graphCollectionInterval)

	// 运行仿真
	fmt.Println("Running simulation...")
	for i := 0; i < simulationSteps; i++ {
		// 模型步进
		changedCount, maxOpinionChange := model.Step()

		// 收集数据
		opinionCollector(model, opinionHistory)
		tweetCollector(model, &tweetHistory)
		graphCollector(model, &graphHistory, i+1, graphCollectionInterval)

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

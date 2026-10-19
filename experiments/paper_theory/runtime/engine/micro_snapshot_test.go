package main

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vmihailenco/msgpack/v5"
	"gonum.org/v1/gonum/graph/simple"
	"smp/simulation"
)

func TestFullSnapshotPreservesNativeStateAndFuture(t *testing.T) {
	task := Task{Selection: "weighted", MicroOptions: &microOptions{PostRetainCount: 1}}
	task.Request.Population, task.Request.OutDegree, task.Request.RecommendationCount = 20, 3, 4
	task.Request.Steps = 20
	task.Request.Dynamics.Type = "hk"
	task.Request.Dynamics.Tolerance, task.Request.Dynamics.Influence, task.Request.Dynamics.RewiringRate = .4, .02, .04
	task.Request.Recommender.Type = "opinion_random"
	task.Request.Recommender.OpinionTolerance, task.Request.Recommender.Steepness = .4, 4
	makeScenario := func() *simulation.Scenario {
		graph := simple.NewDirectedGraph()
		opinions := make([]float64, 20)
		for i := range opinions {
			opinions[i] = -0.95 + float64(i)*.1
			graph.AddNode(simple.Node(i))
		}
		for i := range opinions {
			for j := 1; j <= 3; j++ {
				graph.SetEdge(graph.NewEdge(graph.Node(int64(i)), graph.Node(int64((i+j)%20))))
			}
		}
		scenario, err := makeMicro(task, opinions, graph, 24000)
		if err != nil {
			t.Fatal(err)
		}
		return scenario
	}
	saved, control := makeScenario(), makeScenario()
	for range 5 {
		saved.Step()
		control.Step()
	}
	path := filepath.Join(t.TempDir(), "state.msgpack.lz4")
	if err := saveFullCheckpoint(saved, path); err != nil {
		t.Fatal(err)
	}
	snapshot, _, err := simulation.ReadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	state, err := simulation.InspectFloat64Checkpoint(snapshot)
	if err != nil || state.CompletedStep != 5 || !reflect.DeepEqual(state.Opinions, saved.Model.GetOpinions()) {
		t.Fatalf("physical checkpoint changed: %v", err)
	}
	var native struct{ Opinions msgpack.RawMessage }
	if err := msgpack.Unmarshal(snapshot.Data, &native); err != nil {
		t.Fatal(err)
	}
	if len(native.Opinions) == 0 || native.Opinions[0] != 0xdc {
		t.Fatal("checkpoint uses a transport extension instead of native SMP arrays")
	}
	restored := simulation.NewScenarioWithOptions("", snapshot.Metadata, simulation.ScenarioOptions{Quiet: true})
	if err := restored.InitFromCheckpoint(snapshot, false); err != nil {
		t.Fatal(err)
	}
	for range 5 {
		saved.Step()
		control.Step()
		restored.Step()
	}
	for _, scenario := range []*simulation.Scenario{saved, restored} {
		if !reflect.DeepEqual(scenario.Model.GetOpinions(), control.Model.GetOpinions()) {
			t.Fatal("checkpoint write or restore changed the future opinions")
		}
		a, _ := scenario.Snapshot()
		b, _ := control.Snapshot()
		x, _ := simulation.InspectFloat64Checkpoint(a)
		y, _ := simulation.InspectFloat64Checkpoint(b)
		if !reflect.DeepEqual(x.Graph, y.Graph) {
			t.Fatal("checkpoint write or restore changed future adjacency")
		}
	}
}

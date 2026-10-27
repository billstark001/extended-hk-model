package main

import (
	"encoding/json"
	"github.com/vmihailenco/msgpack/v5"
	"gonum.org/v1/gonum/graph/simple"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/protocol"
	"smp/dynamics"
	"smp/model"
	smprng "smp/rng"
	"smp/simulation"
	"testing"
)

func fixture(t *testing.T) request {
	t.Helper()
	dir := t.TempDir()
	metadata := simulation.DefaultScenarioMetadata()
	metadata.UniqueName = "lifted-continuation-test"
	metadata.NodeCount, metadata.NodeFollowCount = 40, 15
	metadata.NetworkType, metadata.RecsysFactoryType = "Explicit", "Random"
	metadata.DynamicsType = simulation.DynamicsTypeHK
	metadata.HKParams = dynamics.HKParams{Tolerance: .4, Influence: .02, RewiringRate: .04}
	metadata.SMPModelPureParams = model.SMPModelPureParams{RecsysCount: 10, PostRetainCount: 1}
	metadata.RNG = smprng.FixedSpec(1234, 5678)
	g := simple.NewDirectedGraph()
	ops := make([]float64, 40)
	for i := range ops {
		g.AddNode(simple.Node(i))
		ops[i] = -1 + 2*float64(i)/39
	}
	for i := range ops {
		for j := 1; j <= 15; j++ {
			g.SetEdge(g.NewEdge(g.Node(int64(i)), g.Node(int64((i+j)%40))))
		}
	}
	scenario := simulation.NewScenarioWithOptions("", metadata, simulation.ScenarioOptions{Quiet: true})
	if err := scenario.InitFromState(&simulation.InitialState{Graph: g, Opinions: ops}); err != nil {
		t.Fatal(err)
	}
	source, sha, err := saveMicro(scenario, dir, "root")
	if err != nil {
		t.Fatal(err)
	}
	return request{ID: "test", Kind: "micro_probe", Source: source, SourceSHA: sha, SourceKind: "micro", Mode: "branch",
		Steps: 4, Seeds: []seed{{0, 555}}, Output: filepath.Join(dir, "out"), Variant: "stochastic",
		Model: config.RunRequest{RequestID: "test", Layer: "naive", Population: 40, OpinionBins: 15, OutDegree: 15,
			RecommendationCount: 10, MaxSteps: 4, Paths: 1, IntervalPaths: 1, AmbiguitySamples: 1, ConfidenceLevel: .95, Workers: 1,
			MajorClusterMass: .0125, Dynamics: config.DynamicsConfig{Type: "hk", Tolerance: .4, Influence: .02, RewiringRate: .04},
			Recommender: config.RecommenderConfig{Type: "random", Steepness: 1, OpinionTolerance: .4, NoiseQuadraturePoints: 9},
			Initial:     config.InitialConfig{Type: "uniform", OpinionMin: -1, OpinionMax: 1},
			Resolution:  config.ResolutionConfig{ScoreMax: 64, AvailabilityBins: 2, ComponentSizeBins: 2, OpinionQuadrature: 33, OpinionQuadratureRule: "gauss_hermite"},
			FastSlow:    config.FastSlowConfig{Mode: "unsplit", MaxSubsteps: 1, ZeroEventBatches: 1}}}
}

func TestControlsUseFixedWindowsEvenAtTerminalBoundary(t *testing.T) {
	r := fixture(t)
	r.Kind, r.SourceKind, r.Steps = "meso_probe", "meso", 5
	a, b := make([]float64, 15), make([]float64, 225)
	a[7], b[7*15+7] = 1, 15
	projection, err := protocol.EncodeNodeEdgeProjection(a, b, 15)
	if err != nil {
		t.Fatal(err)
	}
	state, err := lifted.FromProjected(r.Model, lifted.LayerNaive, projection, make([]float64, 225), lifted.ClosureProfile{})
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []string{"stochastic", "deterministic", "rewire_only", "transport_only"} {
		r.Variant = variant
		r.Source, r.SourceSHA, err = saveMeso(state, rand.NewPCG(9, 10), 0, r, "boundary-"+variant)
		if err != nil {
			t.Fatal(err)
		}
		out, err := run(r)
		if err != nil {
			t.Fatal(err)
		}
		expected := 5
		if variant == "stochastic" {
			expected = 0
		}
		if out.ExecutedSteps != expected || out.Paths[0].Final == nil || out.Paths[0].Final.Variance != 0 {
			t.Fatalf("%s window or endpoint differs", variant)
		}
	}
}

func TestProjectedBoundaryCanDifferFromMicroscopicBoundary(t *testing.T) {
	r := fixture(t)
	snapshot, _, err := simulation.ReadCheckpoint(r.Source)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := simulation.PrepareCheckpoint(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	m, err := loadMicro(r, r.Seeds[0], prepared, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	ops := make([]float64, 40)
	for i := 20; i < 40; i++ {
		ops[i] = .401
	}
	if err := m.InitFromState(&simulation.InitialState{Graph: m.Model.GetGraph(), Opinions: ops}); err != nil {
		t.Fatal(err)
	}
	r.Source, r.SourceSHA, err = saveMicro(m, t.TempDir(), "two-groups")
	if err != nil {
		t.Fatal(err)
	}
	micro, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Kind = "meso_probe"
	coarse, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if micro.Paths[0].Category != "k2" || coarse.Paths[0].Category != "k1" {
		t.Fatal("boundary test failed to distinguish continuous and projected geometry")
	}
}
func rawMicro(t *testing.T, path string) model.SMPModelDumpData[float64, dynamics.HKParams] {
	t.Helper()
	snap, _, err := simulation.ReadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	var value model.SMPModelDumpData[float64, dynamics.HKParams]
	if err := msgpack.Unmarshal(snap.Data, &value); err != nil {
		t.Fatal(err)
	}
	return value
}
func TestMicroResumeAndIndependentBranches(t *testing.T) {
	r := fixture(t)
	first, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if first.Paths[0].Category != "unresolved" {
		t.Fatal("fixture absorbed too early")
	}
	direct := rawMicro(t, first.Paths[0].Checkpoint)
	r.Steps = 1
	partial, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Source, r.SourceSHA = partial.Paths[0].Checkpoint, partial.Paths[0].CheckpointSHA
	r.Mode, r.Kind, r.Steps = "resume", "micro_extend", 3
	resumed, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(direct, rawMicro(t, resumed.Paths[0].Checkpoint)) {
		t.Fatal("resume changed physical or RNG state")
	}
	r = fixture(t)
	r.Seeds = []seed{{0, 555}, {1, 556}}
	split, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(rawMicro(t, split.Paths[0].Checkpoint), rawMicro(t, split.Paths[1].Checkpoint)) {
		t.Fatal("branches share RNG")
	}
	r.SourceSHA = "wrong"
	if _, err := run(r); err == nil {
		t.Fatal("accepted changed checkpoint")
	}
}
func TestMesoResumeAndRandomLayerEquivalence(t *testing.T) {
	r := fixture(t)
	r.Kind = "meso_probe"
	first, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if first.Paths[0].Category != "unresolved" {
		t.Fatal("fixture absorbed too early")
	}
	read := func(path string) mesoCheckpoint {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var c mesoCheckpoint
		if err := msgpack.Unmarshal(data, &c); err != nil {
			t.Fatal(err)
		}
		return c
	}
	direct := read(first.Paths[0].Checkpoint)
	r.Steps = 1
	partial, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Source, r.SourceSHA = partial.Paths[0].Checkpoint, partial.Paths[0].CheckpointSHA
	r.SourceKind, r.Mode, r.Kind, r.Steps = "meso", "resume", "meso_extend", 3
	resumed, err := run(r)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(direct, read(resumed.Paths[0].Checkpoint)) {
		t.Fatal("meso resume changed state or RNG")
	}
	r = fixture(t)
	g := rand.New(rand.NewPCG(11, 12))
	naive, err := lifted.InitialState(r.Model, lifted.LayerNaive, lifted.ClosureProfile{}, g)
	if err != nil {
		t.Fatal(err)
	}
	g = rand.New(rand.NewPCG(11, 12))
	base, err := lifted.InitialState(r.Model, lifted.LayerBase, lifted.ClosureProfile{}, g)
	if err != nil {
		t.Fatal(err)
	}
	a, b := rand.New(rand.NewPCG(13, 14)), rand.New(rand.NewPCG(13, 14))
	for step := 0; step < 30; step++ {
		if _, err := lifted.Step(naive, r.Model, lifted.ClosureProfile{}, a); err != nil {
			t.Fatal(err)
		}
		if _, err := lifted.Step(base, r.Model, lifted.ClosureProfile{}, b); err != nil {
			t.Fatal(err)
		}
		x, _ := json.Marshal([][]float64{naive.Rho, naive.Edge})
		y, _ := json.Marshal([][]float64{base.Rho, base.Edge})
		if string(x) != string(y) {
			t.Fatalf("Random naive/base physical paths differ at %d", step)
		}
	}
}

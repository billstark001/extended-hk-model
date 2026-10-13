package main

import (
	"bytes"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/vmihailenco/msgpack/v5"
	"smp-meso/kinetic"
)

func TestBinaryFloat64Bits(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, 1.0 / 3}
	var buffer bytes.Buffer
	if err := encodeBinary(&buffer, values); err != nil {
		t.Fatal(err)
	}
	var decoded []float64
	if err := msgpack.NewDecoder(&buffer).Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	for i := range values {
		if math.Float64bits(values[i]) != math.Float64bits(decoded[i]) {
			t.Fatal("float64 changed")
		}
	}
}

func TestBinaryCompactFrameIsExplicit(t *testing.T) {
	var buffer bytes.Buffer
	frame := MicroFrame{InspectionFrame: kinetic.InspectionFrame{Step: 3, Time: 3}, ExactNodeEnergy: .3, ExactEdgeEnergy: .2}
	if err := encodeBinary(&buffer, frame); err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := msgpack.NewDecoder(&buffer).Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	if _, exists := decoded["node_energy"]; exists {
		t.Fatal("invented binned energy")
	}
	if _, exists := decoded["exact_node_energy"]; !exists {
		t.Fatal("missing exact energy")
	}
	if _, exists := decoded["InspectionFrame"]; exists {
		t.Fatal("embedded frame not flattened")
	}
	if !reflect.DeepEqual(decoded["projected"], false) {
		t.Fatal("projection flag")
	}
}

// Emits synthetic transport data for the Python interoperability test. No
// initialization, model update, kinetic solver or scientific simulation runs.
func TestBinaryPythonFixture(t *testing.T) {
	path := os.Getenv("EHK_CODEC_FIXTURE")
	if path == "" {
		return
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	value := map[string]any{"id": "fixture", "status": "ok", "result": MicroResult{
		InitialHash: "synthetic", Frames: []MicroFrame{{Projected: true,
			InspectionFrame: kinetic.InspectionFrame{Rho: []float64{0, 1}, Edge: []float64{0, 0, 0, 1}},
			ExactNodeEnergy: 1.0 / 3}},
	}}
	if err := encodeBinary(file, value); err != nil {
		t.Fatal(err)
	}
}

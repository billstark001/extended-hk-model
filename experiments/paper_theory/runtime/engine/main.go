// Experiment adapter: both public runtime repositories own model stepping.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"smp-meso/kinetic"
)

// run dispatches instrumentation and comparisons through public solver APIs.
func run(t Task) (any, error) {
	initial := t.InitialState
	var result any
	var err error
	switch t.Kind {
	case "trace":
		result, err = kinetic.TraceFromState(t.Request, kinetic.TraceOptions{InitialState: initial, PhysicalStep: t.PhysicalDt, SnapshotSteps: t.SnapshotSteps, EnergyBudgets: t.Budgets})
	case "frozen":
		result, err = kinetic.InspectFrozenState(t.Request, initial)
	case "micro":
		result, err = runMicro(t)
	case "parity":
		var reference kinetic.Result
		reference, err = kinetic.Run(t.Request)
		if err == nil {
			var hooked kinetic.TraceResult
			hooked, err = kinetic.TraceFromState(t.Request, kinetic.TraceOptions{PhysicalStep: t.PhysicalDt, SnapshotSteps: t.SnapshotSteps, EnergyBudgets: t.Budgets})
			result = map[string]any{"reference": reference, "hooked": hooked}
		}
	default:
		err = fmt.Errorf("unknown task kind %q", t.Kind)
	}

	return result, err
}

// main accepts exactly one strict JSON object per line. Stdout is reserved for
// identified ok/error responses; transport failures are reported on stderr.
func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 65536), 64<<20)
	encoder := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var t Task
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&t)
		if err == nil {
			var extra any
			if e := decoder.Decode(&extra); e != io.EOF {
				err = fmt.Errorf("exactly one JSON object required per line")
			}
		}
		var result any
		if err == nil && t.ID == "" {
			err = fmt.Errorf("task id is required")
		}
		if err == nil {
			result, err = run(t)
		}
		response := map[string]any{"id": t.ID, "status": "ok", "result": result}
		if err != nil {
			response = map[string]any{"id": t.ID, "status": "error", "error": err.Error()}
		}
		if err := encoder.Encode(response); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

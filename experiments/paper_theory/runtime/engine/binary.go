package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"reflect"

	"github.com/vmihailenco/msgpack/v5"
)

// Float64 vectors are typed binary payloads, not arrays of boxed numbers.
// This extension is shared with common.io.records.decode_array in Python.
func init() {
	msgpack.Register([]float64(nil), func(enc *msgpack.Encoder, value reflect.Value) error {
		array := value.Interface().([]float64)
		if array == nil {
			return enc.EncodeNil()
		}
		if err := enc.EncodeExtHeader(42, len(array)*8); err != nil {
			return err
		}
		buffer := make([]byte, 8192)
		for start := 0; start < len(array); start += 1024 {
			end := min(start+1024, len(array))
			for i, number := range array[start:end] {
				binary.LittleEndian.PutUint64(buffer[i*8:], math.Float64bits(number))
			}
			if _, err := enc.Writer().Write(buffer[:(end-start)*8]); err != nil {
				return err
			}
		}
		return nil
	}, func(dec *msgpack.Decoder, value reflect.Value) error {
		code, length, err := dec.DecodeExtHeader()
		if err != nil || code != 42 || length%8 != 0 {
			return fmt.Errorf("invalid float64 vector: %v", err)
		}
		data := make([]byte, length)
		if err := dec.ReadFull(data); err != nil {
			return err
		}
		array := make([]float64, length/8)
		for i := range array {
			array[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
		}
		value.Set(reflect.ValueOf(array))
		return nil
	})
	// Explicitly flatten the embedded InspectionFrame. Compact samples only
	// contain continuous observables; absent binned energies are never set to 0.
	msgpack.Register(MicroFrame{}, func(enc *msgpack.Encoder, value reflect.Value) error {
		f := value.Interface().(MicroFrame)
		fields := map[string]any{
			"step": f.Step, "time": f.Time, "exact_node_energy": f.ExactNodeEnergy,
			"exact_edge_energy": f.ExactEdgeEnergy, "rewires": f.Rewires,
			"mean_degree": f.MeanDegree, "exact_variance": f.ExactVariance,
			"exact_discordant": f.ExactDiscordant, "projected": f.Projected,
		}
		if f.Projected {
			fields["node_energy"], fields["edge_energy"] = f.NodeEnergy, f.EdgeEnergy
			fields["reference_energy"], fields["variance"], fields["discordant"] = f.ReferenceEnergy, f.Variance, f.Discordant
			if f.Rho != nil {
				fields["rho"] = f.Rho
			}
			if f.Edge != nil {
				fields["edge"] = f.Edge
			}
		}
		return enc.Encode(fields)
	}, func(dec *msgpack.Decoder, value reflect.Value) error {
		return fmt.Errorf("MicroFrame decoding is owned by the Python record reader")
	})
}

func encodeBinary(writer io.Writer, value any) error {
	enc := msgpack.NewEncoder(writer)
	enc.SetCustomStructTag("json")
	enc.SetSortMapKeys(true)
	return enc.Encode(value)
}

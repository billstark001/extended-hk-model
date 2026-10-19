package main

import (
	"smp-meso/kinetic"
	"smp-meso/protocol"
)

// Task is one JSONL adapter request. Python freezes its ID, physical parameters
// and seeds before execution. Model stepping remains in the two public runtimes.
// SnapshotSteps and AuditSteps count updates; PhysicalDt is physical time per
// update. InitialState stores node/edge cell masses, not opinion densities.
type Task struct {
	// InitialRNG selects shared opinion/graph draws or split initialization streams.
	InitialRNG string `json:"initial_rng"`
	// Projection selects actual graph masses or rows rescaled to nominal degree.
	Projection string `json:"projection"`
	// RecordPolicy selects conditional-audit fields or compact population frames.
	RecordPolicy string `json:"record_policy"`
	// CompactRecords affects recording only; never enters frozen scientific IDs.
	CompactRecords bool `json:"compact_records"`
	// EnergyNormalization divides edge sums by N*k (nominal) or actual edge count.
	EnergyNormalization string             `json:"energy_normalization"`
	ID                  string             `json:"id"`
	Kind                string             `json:"kind"`
	Request             kinetic.RunRequest `json:"request"`
	PhysicalDt          float64            `json:"physical_dt"`
	SnapshotSteps       []int              `json:"snapshot_steps"`
	// CheckpointDir opts into SMP full-state checkpoints; ordinary batches stay in memory.
	CheckpointDir      string                       `json:"checkpoint_dir,omitempty"`
	Budgets            bool                         `json:"budgets"`
	InitialState       *protocol.NodeEdgeProjection `json:"initial_state"`
	Initial            string                       `json:"initial"`
	MicroscopicInitial *microscopicInitial          `json:"microscopic_initial"`
	MicroOptions       *microOptions                `json:"micro_options"`
	// BranchSeeds gives independent process seeds for each saved audit state.
	BranchSeeds map[int][]uint64 `json:"branch_seeds"`
	Network     string           `json:"network"`
	Replicate   int              `json:"replicate"`
	// InitSeed fixes the physical start; ProcessSeed changes only the evolution.
	InitSeed    uint64 `json:"init_seed"`
	ProcessSeed uint64 `json:"process_seed"`
	Branches    int    `json:"branches"`
	AuditSteps  []int  `json:"audit_steps"`
	Selection   string `json:"selection"`
}

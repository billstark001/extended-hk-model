# Runtime installation and data locations

The three repositories have separate responsibilities. Keep them as separate
checkouts; `extended-hk-model` owns experiment orchestration, not the solvers.

| Repository | Owns | Dependencies for this paper |
| --- | --- | --- |
| [extended-hk-model](https://github.com/billstark001/extended-hk-model) | Python theory, experiment designs, fitting, analysis, figures, one small Go experiment adapter | Python packages in `pyproject.toml`; both runtime source checkouts for compiled micro/pair experiments |
| [social-media-models](https://github.com/billstark001/social-media-models) | Public microscopic Go dynamics/recommenders; `smp`, `smp-batch`, `smp-probe`; `smp_bindings` Python reader/launcher | Its own Go and Python dependencies; it does not import this experiment repository |
| [social-media-mesoscopic-models](https://github.com/billstark001/social-media-mesoscopic-models) | Public pair-measure Go dynamics; `bin/smp-kinetic`; `smp_meso_bindings` Python launcher | Its own Go and Python dependencies; it does not import this experiment repository |

## Install

Use Python >=3.10 and a Go version satisfying both runtimes' `go.mod` and the
experiment engine (currently Go >=1.26.7). The micro command uses CGO/SQLite;
install a C compiler for those CLI builds. The default mesoscopic backend is
pure Go; Accelerate/OpenBLAS are optional and are not required here.

For sibling checkouts, run these commands from the model repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . -e ../social-media-models -e ../social-media-mesoscopic-models
python -m pip install pytest
cp .env.example .env

go -C ../social-media-models build -o smp ./cmd/smp
go -C ../social-media-models build -o smp-batch ./cmd/smp-batch
go -C ../social-media-models build -o smp-probe ./cmd/smp-probe
go -C ../social-media-mesoscopic-models build -o bin/smp-kinetic ./cmd/smp-kinetic
python -m pytest -q
```

Only install the dependencies needed by the intended workflow. Reading energy
NPZ/CSV inputs and redrawing figures require NumPy/SciPy/Matplotlib and this
project, but do not require the Go binaries or either binding. Microscopic
legacy record readers require `smp_bindings`. New measure scans require
`smp_meso_bindings` and `smp-kinetic`. The shared validation/matched-population engine
links both Go modules directly with generated `replace` entries, rather than
launching the microscopic CLI.

For checkouts in other locations, edit the ignored `.env`:

```dotenv
EHK_MICRO_SOURCE_DIR=/path/to/social-media-models
EHK_MESO_SOURCE_DIR=/path/to/social-media-mesoscopic-models
EHK_KINETIC_BINARY=/path/to/social-media-mesoscopic-models/bin/smp-kinetic
SMP_BINARY_PATH=/path/to/social-media-models/smp
SMP_PROBE_BINARY=/path/to/social-media-models/smp-probe
EHK_PAPER_THEORY_OUTPUT_DIR=/path/to/generated/theory
```

Build the public mesoscopic command or the shared experiment adapter through the unified interface:

```sh
python -m experiments.paper_theory build --suite scaling
python -m experiments.paper_theory build --suite validation
```

The first command builds `smp-kinetic` under the configured ignored output tree. The second builds the single `runtime/engine` adapter linking both runtime source modules; `validation`, `ensembles` and matched `population` use this same binary. `--micro-root`, `--meso-root`, `--backend` and `--build-dir` override the corresponding build settings. Generated module files and binaries stay under outputs. Do not edit runtime module replacements or copy solver implementations here. Changed build inputs require a new immutable build directory. See the [workflow guide](../experiments/paper_theory/README.md) for presets and pipelines.

## External data and output links

Large microscopic records and TB-scale workspaces remain external. Configure
`SIMULATION_WS_PATH` to an ignored local workspace manifest and
`SIMULATION_STAT_DIR`/`EHK_PAPER_OUTPUT_DIR` for the derived microscopic tables.
The workspace JSON maps run groups to the actual mounted data directories.
`SIMULATION_INSTANCE_NAME` must name a group present in that JSON. Missing
volumes are a data-availability issue, not a reason to place the records in Git.

For the theory workflows, configure these read-only input roots. The revision-named settings refer to historical dataset layouts; they do not select legacy code entry points:

| Setting | Expected contents |
| --- | --- |
| `EHK_RELEASE_SCAN_INPUT_DIR` | `raw_shards/*.npz`, `low_parameter_fits.csv`, `edge_timescale_fits.csv` |
| `EHK_RELEASE_CLOCK_INPUT_DIR` | `parameter_clocks/data/*.csv`, optional `raw/*.npz` refinements |
| `EHK_REVISION_INPUT_DIR` | `revision_core_v3/`, `revision_followup_v1/`, with frozen plans, bindings, status, per-family raw results and analysis |
| `EHK_REVISION_REVIEW_INPUT_DIR` | `review_20261007/`, `review_followup_20261007/` tables |
| `EHK_MANUSCRIPT_INPUT_DIR` | Existing `provenance.json`, `rewiring_sign_audit.json`, `tables/` for the historical claim audit |

Input defaults are under `outputs/inputs/paper_theory`, separate from generated
`outputs/experiments/paper_theory`; no paper checkout is a runtime default. These inputs can remain on external storage, or be exposed
through links within ignored `outputs/`. For example, with existing datasets
at `/path/to/data/expanded` and `/path/to/data/clocks`:

```sh
mkdir -p outputs/inputs/paper_theory/energy_scan
ln -s /path/to/data/expanded outputs/inputs/paper_theory/energy_scan/expanded
ln -s /path/to/data/clocks outputs/inputs/paper_theory/release_clocks
```

Create links only where no directory/link already exists. A linked historical
input is read-only by convention: use a separate `--output-root`
for generation or reanalysis. Alternatively use the input settings directly;
links are not required. Relative configured paths resolve against the model
repository root. The local `.env`, workspace manifests, `outputs/` directories
and links, builds, raw records, fit tables, reports and figures remain ignored.
Only code, fixed experiment definitions and documentation are versioned.

Publication is separate from Git. An optional data archive should include the
selected files, recorded protocols, checksums, directory mapping and retrieval
instructions. A DOI/archive service is not required to execute these workflows.
No upload, download of bulk data, or archival publication is automatic.

The CAS 2026 study serves as a literature citation and motivates the question.
It is not an input dataset or a rerun target of the theory workflow. Existing
`experiments/paper` remains available for the original microscopic paper.

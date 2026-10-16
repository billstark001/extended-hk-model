# Runtime installation and data locations

The three repositories have separate responsibilities. Keep them as separate
checkouts; `extended-hk-model` owns experiment orchestration, not the solvers.

| Repository                                                                                       | Owns                                                                                                                | Dependencies for this paper                                                                            |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [extended-hk-model](https://github.com/billstark001/extended-hk-model)                           | Python theory, experiment designs, fitting, analysis, figures, one small Go experiment adapter                      | Python packages in `pyproject.toml`; both runtime source checkouts for compiled micro/pair experiments |
| [social-media-models](https://github.com/billstark001/social-media-models)                       | Public microscopic Go dynamics/recommenders; `smp`, `smp-batch`, `smp-probe`; `smp_bindings` Python reader/launcher | Its own Go and Python dependencies; it does not import this experiment repository                      |
| [social-media-mesoscopic-models](https://github.com/billstark001/social-media-mesoscopic-models) | Public pair-measure Go dynamics; `bin/smp-kinetic`; `smp_meso_bindings` Python launcher                             | Its own Go and Python dependencies; it does not import this experiment repository                      |

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
SMP_BINARY_PATH=/path/to/social-media-models/smp
SMP_PROBE_BINARY=/path/to/social-media-models/smp-probe
EHK_PAPER_THEORY_OUTPUT_DIR=/path/to/generated/theory
```

The main `run` command builds both the public mesoscopic executable and the
small experiment adapter; `resume` restores the original verified executable
copies. The active CLI supports named research topics; `main`/`main_smoke` select all formal/software topics.

```sh
python -m experiments.paper_theory plan --profile main --work-dir /path/to/new-run
python -m experiments.paper_theory run --profile main --work-dir /path/to/new-run --publish-dir /path/to/export
python -m experiments.paper_theory plot --run-dir /path/to/export --tables-only
```

Keep `.venv`, control logs/caches and pending result commits on local APFS.
Bulk records may remain external. `--control-dir`, `--micro-root`, `--meso-root`
and `--backend` configure generation. Builds retain immutable source/module and
binary hashes; changed build inputs require a new run. See the
[workflow guide](../experiments/paper_theory/README.md).

## External data and output links

Large microscopic records and TB-scale workspaces remain external. Configure
`SIMULATION_WS_PATH` to an ignored local workspace manifest and
`SIMULATION_STAT_DIR`/`EHK_PAPER_OUTPUT_DIR` for the derived microscopic tables.
The workspace JSON maps run groups to the actual mounted data directories.
`SIMULATION_INSTANCE_NAME` must name a group present in that JSON. Missing
volumes are a data-availability issue, not a reason to place the records in Git.

Theory offline commands take an explicit frozen dataset directory using
`--run-dir`; it contains `plan.json`, bindings, `measure/raw`, and/or
`micro/raw` plus shared `micro/pairs`. CSV-only exports need the frozen plan,
report and tables. Historical audit libraries take explicit dataset roots.
Energy-slot fitting/plotting takes an explicit baseline/expanded NPZ archive
or shard directory through `--run-dir`, with `--dynamics`, `--epsilon`,
`--alpha` and `--ratio`. The old scan/fit input settings and symlink layouts
are retired; these offline commands do not need runtime executables.

Generated outputs, local environment/workspace manifests, links, builds and
raw records remain ignored. Only code, fixed study definitions and
documentation are versioned.

Publication is separate from Git. An optional data archive should include the
selected files, recorded protocols, checksums, directory mapping and retrieval
instructions. A DOI/archive service is not required to execute these workflows.
No upload, download of bulk data, or archival publication is automatic.

The CAS 2026 study serves as a literature citation and motivates the question.
It is not an input dataset or a rerun target of the theory workflow. Existing
`experiments/paper` remains available for the original microscopic paper.

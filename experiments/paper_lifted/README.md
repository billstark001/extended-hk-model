# HK + Random commitment experiments

`src/lifted` owns conditional outcome probabilities, all-agent terminal-label
handling, commitment on an observation grid, and joint increment diagnostics.
This study owns sampling, frozen plans, resource allocation and offline figures.
`common.experiments` owns task selection, execution, journals and recovery.
Model stepping and recommendation remain in the public Go runtimes.

See [CONTINUATION.md](CONTINUATION.md) for current global progress, balanced
scheduling, worker-independent resume and explicit cumulative-budget extensions.
The [module audit](../../docs/lifted-architecture-audit.md) documents shared-service
extraction, responsibility boundaries and deferred migrations.

See [RECHECK.md](RECHECK.md) for the current recovery/phase checks and frozen
six-to-eight-hour plan. [VALIDATION.md](VALIDATION.md) retains the initial snapshot.

## Validation and plan freezing

Run from the model repository with the installed runtime bindings. Each new
validation run needs a new output directory. The default backend is Accelerate;
pass `--backend purego` when that backend is unavailable.

```sh
python -m experiments.paper_lifted smoke --work-dir /absolute/new-smoke --workers 4
python -m experiments.paper_lifted pilot --work-dir /absolute/new-pilot --workers 4
python -m experiments.paper_lifted freeze --pilot-dir /absolute/new-pilot --work-dir /absolute/new-formal --workers 4 --budget-hours 7
python -m experiments.paper_lifted status --work-dir /absolute/new-formal
```

`smoke` exercises the entire public runtime/record/analysis path at N=40, B=15
and short horizons. `pilot` uses the theory physical settings at N=500,
epsilon=.4, alpha=.02, fixed out-degree 15 and ten recommendations. It measures
all three core ratios .25/2/10, B=81/161, Q=33, actual long terminal continuations
from early/intermediate/late states (tau=0/4/30),
single-step moments, the four noise controls, and short N=250/1000/2000 and B=241
cost checks. Both are software/cost evidence, not formal scientific results.

`freeze` reads and authenticates the complete pilot. It never calls a build,
solver or executor. It selects replication from measured costs, binds the pilot
records, code and runtime hashes, and writes only `plan.json` and `budget.json`.
The central estimate lies between six hours and the requested target (six to
eight hours). An eight-hour persistent execution budget is independent of that
estimate. Estimates extrapolate unsampled path lengths and are not guarantees
of complete coverage by the deadline.

Formal execution is a separate, explicit action:

```sh
python -m experiments.paper_lifted run --work-dir /absolute/new-formal --workers 4
python -m experiments.paper_lifted resume --work-dir /absolute/new-formal --workers 4
python -m experiments.paper_lifted analyze --work-dir /absolute/new-formal
python -m experiments.paper_lifted plot --work-dir /absolute/new-formal
```

The three phases are named **run / analyze / plot**, matching `paper_theory`.
By default `run` and `resume` perform all three. Select execution alone with
`--phase run`; repeat `--phase` to select several phases in that order:

```sh
python -m experiments.paper_lifted run --work-dir /absolute/new-formal --workers 4 --phase run
python -m experiments.paper_lifted resume --work-dir /absolute/new-formal --workers 4 --phase run
python -m experiments.paper_lifted analyze --work-dir /absolute/new-formal
python -m experiments.paper_lifted plot --work-dir /absolute/new-formal
```

`analyze` works on an incomplete run, including a plan with no completed tasks.
It fixes one authenticated record snapshot for all tables, retains every planned
initial state and probability sample in the denominator, and separates pending
work from physically unresolved paths. Records committed during analysis enter
the next export. Analysis does not start, build or restore a model. Plotting uses
four portable CSV files; it needs no raw records, checkpoints or runtime checkout.
`resume --phase analyze --phase plot` also bypasses model/source startup checks.

Analysis and plots have separate manifests. `status` reports `analysis_current`
and `plots_current`; new records invalidate both until re-exported. Analysis
never includes old figure files in its manifest. Plotting checks CSV hashes when
an analysis manifest is present and skips entirely pending, unclassified queries.
Concurrent analysis/plot writers use a shared output lock while execution may
continue writing independent, atomic raw records.

Backend must match the frozen cost calibration. Workers are a runtime option;
new plans retain only a pilot reference count. Scientific source or
runtime changes require a new run. Formal execution copies and authenticates the
pilot's measured executables and module manifests, so later Git metadata alone
does not cause a rebuild. Numerical source hashes must still match. Budget
selection, cost estimation, CLI dispatch, offline exports and executable staging
may be revised before freezing;
the receipt records those revisions separately from the original pilot binding.
Execution-only source revisions are audited separately while preserving the
original scientific binding. Resume skips authenticated completed work,
retains the remaining wall budget and reuses exact checkpoint inputs. Ordinary
cleanly paused time is not charged. An unfinalized killed attempt is charged
conservatively at recovery. Pending tasks and process failures are operational
missing work; they never become physical terminal labels.
An interrupted in-flight task restarts deterministically from its committed input
checkpoint with the same task seed. Completed chunks are authenticated and
skipped. Physical long-tail continuation instead restores that path's saved
checkpoint and RNG; neither kind of recovery creates an additional sample.

## Scientific design

The primary population design uses iid uniform opinions, a directed fixed-out-
degree graph, HK, Random, epsilon=.4, alpha=.02, dt=1, D0=0, no reposts and
one current post per sender. Reported time is tau=alpha*t. The three core
q/alpha ratios are .25/2/10. Mother snapshots are at tau=0/1/4/16/30/120.
The initial-state and future-process streams are separated, matching theory's
population initialization. Same-initial-state mothers share their t=0 query.

The pilot chooses 4/8/12/16/24/32 initial states per condition, one or two mothers,
64/128/256 primary microscopic continuations, and 16--2048 coarse continuations
per selected state. The selection prefers at least eight initial states and
128 microscopic paths whenever that fits, then prioritizes independent initial
coverage and mothers before inner replication. It uses costs, not formal labels.

Prespecified extensions are tagged `mechanism` and excluded from population
timing: the full eleven-point ratio grid at B=81, size comparisons, and paired
two-group initial states with 1/4/16 bridging agents. The bridge cases reuse the
same graph family and initialization seed while changing the exact bridging
count. Their group centers are +/-0.35 with narrow deterministic within-group
offsets. These are a separate initial law.

Four coarse controls retain the same lifted kernel: stochastic, deterministic,
rewire-only noise and transport-only noise. The three channel controls use fixed
tau=120 diagnostic windows at six states (one deterministic and eight sampled
paths per state), with endpoint energies, variance and discordant mass exported
in `noise_controls.csv`. They are not eventual-terminal-probability ensembles,
and do not enter the long-tail extension queue. The pilot times tau=30 windows.
The main coarse grid is B=161;
B=81/241 are resolution checks. Random naive/base physical-path equivalence is
checked with identical seeds, so the main study uses naive without carrying
unused score coordinates. This equality does not claim exact closure of the
microscopic process. The current theory measure map remains a separate reference:
it uses cell-averaged confidence whereas the lifted kernel uses bin centers and
candidate conditioning. Paired initial projections expose this difference.

Each mother also continues along its original random stream to supply a held-out
terminal label. These futures are independent of the probability probes given
the queried state. Conditional validation exports Brier scores and report-class
accuracy; times from one mother are correlated and remain grouped by initial
state in any subsequent uncertainty analysis.

## Outcomes and probability/time interpretation

Primary terminal classes use all confidence components. No bridge mass is
discarded: each component has diameter <=epsilon and adjacent components have
gap >epsilon. The physical class count is retained, with 4+ combined for display.
Numerical halt is not used as a terminal decision.

The initial continuation cap is relative tau=480. Physical unresolved paths
retain their exact checkpoint/RNG and can continue another 480 without becoming
new samples. Point estimates keep unknown mass; Clopper-Pearson/Bonferroni bounds
cover every allocation of that mass. Unexecuted planned paths are recorded
separately as pending and also receive unrestricted label allocations in bounds.
No completed-only renormalization is used.
The simultaneous family is fixed before execution and reserves 42 possible
follow-up queries (two nine-point micro/coarse windows and six validations).
Repeated partial analysis and adaptive selection do not change core bounds.

Queried microscopic terminal states use their exact one-hot boundary probability.
Coarse queries use the classifier of their projected state, not the original
continuous-opinion classification. This prevents grid projection from silently
changing a boundary label, and avoids sampling intervals on known boundaries.

Delta=.1 and .05 are predeclared. Output separates first estimated threshold hit,
first possible hit and first confirmed hit on the observation grid, and records
confirmed returns below the threshold. The CDF bounds describe the observed
initial-state ensemble and inner probability uncertainty; they are not confidence
bands for the entire initial-state law. No infinite-horizon mean is inferred.

After the fixed core, a separately frozen follow-up plan binds its parent record
hashes. It includes same-stream unresolved-tail extensions, independent validation
for at most six states, and at most two nine-step local windows. Window replay
checks the original mother checkpoint; new query streams remain independent.
Selected follow-ups remain mechanism evidence, not population prevalence.

## Runtime and storage

The study reuses the existing theory instrumentation adapter solely for explicit
fixed-degree initialization and native step-zero checkpoints, via the public
`theory.runtime.population` request/transport interfaces. The lifted adapter in
`src/lifted/runtime/engine` calls SMP prepared-checkpoint restore and `Step`, and
the mesoscopic `FromProjected`, `HybridStep` and `TraceFromState` APIs. It does not
contain an opinion-update, rewiring, recommendation or closure implementation.

Every raw record binds its task, executables, source record and checkpoint hashes.
Checkpoint files commit through `RunStorage` before the completion record.
Microscopic checkpoints use SMP's native format. Naive/base coarse checkpoints
retain rho/E/score, the PCG state, numerical/physical identity and evolution mode.
Joint moment records retain increment arrays; analysis exports covariance factors
rather than a dense (B+B^2)-squared matrix.

Data stays under the caller's ignored run root; control/spooling is local. A
worker runs one single-threaded Go model request; micro and meso share the same
worker limit. Follow-up work and analysis have explicit budget reserves. Stages
stop dispatching before the deadline, with remaining scientific tasks missing.

`analysis/` contains portable probability, timing, CDF, held-out validation,
closure-error, mother-path, state, moment and runtime-cost tables, covariance
factors, hashes and a completion report. The four PDF/PNG figures read CSV only.
Smoke/pilot and incomplete formal reports are diagnostic outputs.

## Checks

```sh
python -m ruff format --check src/lifted experiments/paper_lifted
python -m ruff check src/lifted experiments/paper_lifted
python -m mypy
python -m pytest -q
```

The Go continuation tests use a built adapter's authenticated module file:
run `go test -modfile /absolute/run/build/lifted/go.mod ./...` from
`src/lifted/runtime/engine`. They check exact micro/coarse checkpoint continuation,
independent branches, source-hash rejection and Random layer equivalence.

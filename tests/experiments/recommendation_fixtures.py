"""Real domain observations shared by recommendation boundary tests."""

from dataclasses import replace

from ehk.domain import (AnalysisReport, ArmObservation, Binding, BundleManifest, Endpoint, Exposure, FullResult,
                        Pathway, Plan, SlotCounts, StageResult, StagedResult, TerminalResult)
from ehk.domain import PooledCoverage
from smp_bindings.domain import TerminalMargins
from experiments.paper.recommendation import study as staged_study
from experiments.paper.recommender_variants import study as variants_study


def empty_slots():
    return SlotCounts(shown=0, cross_camp=0, cross_confident=0, cross_rewire=0)


def exposure():
    return Exposure(updates=0, requested=0, neighbor_seen=0, neighbor_concordant=0,
                    neighbor_with_posts=0, neighbor_share_sum=0., recommended=0,
                    exploration=empty_slots(), policy=empty_slots(), bridge=empty_slots(),
                    cross_camp=0, cross_confident=0, cross_rewire=0, rewires=0, absolute_shift=0.)


def point(step=0, start=0, separation=.5, excess=.05, *, counters=None):
    return Endpoint(step=step, relative_step=step-start,
                    pathway=Pathway(separation=separation, pair_concordance=.5,
                                    edge_concordance=.5+excess, excess_concordance=excess),
                    exposure=exposure() if counters is None else counters)


def terminal(status="nonterminal", k_major=0):
    return TerminalResult(status=status, category=status, k_all=k_major, k_major=k_major,
                          components=(), margins=TerminalMargins(
                              gap_to_epsilon=0., diameter_to_epsilon=0., mass_to_major_cutoff=0.))


def arm(start=0, *, endpoints=None, opinions=(-.8, -.7, .7, .8), status="nonterminal",
        k_major=0, raw_record="raw/test", binary_sha256=None):
    if endpoints is None:
        endpoints = (point(start, start), point(start+512, start,
                     counters=replace(exposure(), updates=100)))
    return ArmObservation(endpoints=tuple(endpoints), terminal=terminal(status, k_major),
                          stop_reason=status, stopped_at=endpoints[-1].step,
                          final_opinions=tuple(opinions), raw_record=raw_record,
                          binary_sha256=binary_sha256)


def staged(stages=None):
    return StagedResult(reference=arm(), stages={} if stages is None else stages)


def stage(start=0, arms=None):
    return StageResult(checkpoint=point(start, start), arms={} if arms is None else arms)


def plan(profile="staged", *, tasks=None):
    value = variants_study.make_plan() if profile == "recommender_variants" else staged_study.make_plan(profile)
    return value if tasks is None else replace(value, tasks=tuple(tasks))


def bundle_manifest(*, files=None, completed=0, planned=0, profile="staged", plan_fingerprint="fixture"):
    return BundleManifest(schema=2, exported_at_utc="2026-11-27T00:00:00+00:00",
                          source_dataset="fixture/source", observation_dataset="fixture/source",
                          plan_fingerprint=plan_fingerprint, profile=profile,
                          available_records=completed, completed=completed, planned=planned,
                          files={} if files is None else files)


def report(source_plan="fixture"):
    return AnalysisReport(study="staged", created_at_utc="2026-11-27T00:00:00+00:00",
                          source_directory="fixture", source_plan=source_plan, binding=None,
                          record_sha256={}, raw_root="fixture", planned=1, available_records=1,
                          completed=1, complete=True, missing=(), settings=(),
                          pooled_figures=PooledCoverage(rule="completed_parameter_cells_only", tasks=1, cells=1))

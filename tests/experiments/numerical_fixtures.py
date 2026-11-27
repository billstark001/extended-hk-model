"""Declared scientific fixtures for metric and numerical-pipeline tests."""
from dataclasses import replace
from ehk.domain import ScenarioMetadata, HKParams, Rng
from common.io.json import canonical
from experiments.paper.metric_comparison.contracts import (
    CacheCounts,CollectionManifest,ComparisonManifest,Coverage,EvaluationRecord,MetricPoint,PathMeasurement,
    PathMetric,PeakAudit,PreparationManifest,PreparedPath,ReportManifest,Weighting,
)
from experiments.paper.metric_comparison import raw


def metadata(name="hold"):
    return ScenarioMetadata(UniqueName=name,NodeCount=4,HKParams=HKParams(Tolerance=.45,Influence=.05,RewiringRate=.05,RepostRate=.1),
        RNG=Rng(Algorithm="PCG",Seed1="1",Seed2="2"))


def path(directory,*,profile="staged",task_id="task",stage="late",arm="hold",start=16,end=32):
    record=f"raw/{task_id}/{stage}/{arm}"
    return PreparedPath(profile,task_id,stage,arm,"plan","binary",record,str(directory.parents[3]),str(directory),
        start,end,True,"",.45,0,.05,.05,.1,"core",0,"1","2",directory.exists(),canonical(metadata(arm)),
        canonical(metadata(arm).RNG),"{}","[]","absorbed",2,
        '{"status":"absorbed","k_major":2}')


def prepared(points=2,paths=1):
    return PreparationManifest(schema=1,files={},status="pending_metrics",sources=(),points=points,paths=paths,
        available_raw_paths=paths,pending_raw_paths=0,missing_metrics=("I_p","I_s","I_h","I_w"),
        weighting=Weighting("users","edges"),path_requirement="full prefix")


def collection(**changes):
    value=CollectionManifest(schema=1,files={},preparation_fingerprint="prep",endpoint_complement=True,
        preparation_sha256="prep",status="complete",planned_paths=2,selected_paths=2,completed_paths=2,
        pending_paths=0,failed_paths=0,points=4,paths=(),cache=CacheCounts())
    return replace(value,**changes)


def comparison(**changes):
    return replace(ComparisonManifest(schema=1,files={},status="complete",coverage={"paths":Coverage(2,2,0)},
        endpoint_complement=True,preparation_sha256="prep",collection_sha256="collect"),**changes)


def report(**changes):
    return replace(ReportManifest(schema=1,files={},status="complete",paths=2,endpoint_complement=True,
        points=4,paired_interventions=1,endpoint_peak_count_audit=PeakAudit(2,0,0),preparation_sha256="prep",
        comparison_sha256="compare",report_source_sha256="source"),**changes)


def measurement(path):
    p=path.identity
    points=tuple(MetricPoint(p.profile,p.task_id,p.stage,p.arm,p.plan_fingerprint,p.binary_sha256,p.raw_record,step,.2,.3,.4)
        for step in (path.start_step,path.end_step))
    return PathMeasurement(points,PathMetric(p.profile,p.task_id,p.stage,p.arm,p.plan_fingerprint,p.binary_sha256,p.raw_record,
        .5,raw.IW_SAMPLING_METHOD,0,path.end_step,True),EvaluationRecord((0,path.end_step),True))

"""Typed preparation and alignment of the single SQLite comparison route."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from common.experiments.lifecycle import session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import atomic_json, canonical, fingerprint, sha
from common.storage.documents import DocumentWriter, encode_document
from ehk.domain import (
    ArmObservation, Binding, BundleManifest, FullResult, Plan, RecommendationProtocol,
    RecommenderParams, StagedResult, Task, parse_binding, parse_dataclass, read_plan, rows_frame,
)
from experiments.paper.recommendation_shared.bundle import read_bundle_manifest
from experiments.paper.recommendation_shared.storage import StudyStore, read_table, write_table

from .contracts import (
    ArtifactManifest, ComparisonManifest, Coverage, PathProvenance, PreparedPath, PreparedPoint,
    PreparationInput, PreparationInputs, PreparationManifest, PreparationSource,
    PreparedTaskCheckpoint, PreparedTaskRow, Weighting,
)

SCHEMA = 1
IW_SAMPLING_METHOD = "canonical_adaptive_0.01_max_interval_512"
PATH_KEYS = ["profile", "task_id", "stage", "arm"]
POINT_KEYS = [*PATH_KEYS, "step"]
PROVENANCE = ["plan_fingerprint", "binary_sha256", "raw_record"]
POINT_METRICS = ["I_p", "I_s", "I_h"]
PATH_METRICS = ["I_w", "sampling_method", "prefix_start", "trajectory_end", "endpoint_complement"]
POINT_FIELDS = [*POINT_KEYS, "relative_step", "separation", "excess_concordance", "edge_concordance",
                "pair_concordance", *PROVENANCE, "prefix_reference_raw_record"]
PATH_FIELDS = [*PATH_KEYS, *PROVENANCE, "source_dataset", "raw_directory", "start_step", "end_step",
               "full_prefix_required", "prefix_reference_raw_record", "epsilon", "history", "influence",
               "rewiring", "repost", "sensitivity", "replicate", "seed1", "seed2", "available",
               "metadata_json", "seed_json", "expected_rec_params_json", "missing_files", "terminal_status",
               "major_components", "terminal_json"]


def load_contract(path, contract):
    """Parse a JSON document immediately at the file boundary."""
    return parse_dataclass(contract, json.loads(Path(path).read_text(encoding="utf-8")))


def endpoint_mode(record) -> bool:
    value = record.endpoint_complement
    if type(value) is not bool:
        raise ValueError("endpoint_complement must be a boolean")
    return value


def validate_endpoint_column(frame, endpoint_complement):
    if "endpoint_complement" not in frame:
        if len(frame):
            raise ValueError("I_w table has no explicit endpoint_complement policy")
        frame["endpoint_complement"] = endpoint_complement
    values = frame["endpoint_complement"].astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any() or not (values == endpoint_complement).all():
        raise ValueError("I_w endpoint_complement differs from the selected policy")
    frame["endpoint_complement"] = values.astype(bool)
    return frame


def _relative(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes evidence directory: {relative}")
    return path


def _authenticate_files(root: Path, manifest: ArtifactManifest | BundleManifest):
    if manifest.schema not in (SCHEMA, 2) or not manifest.files:
        raise ValueError("unsupported or empty evidence manifest")
    if isinstance(manifest, ArtifactManifest):
        for name, expected in manifest.sqlite_tables.items():
            frame = read_table(root, name)
            # Hashing is a serialization boundary; tables themselves stay columnar.
            actual = fingerprint(encode_document({"columns": list(frame.columns), "rows": frame.to_dict("records")}))
            if actual != expected:
                raise ValueError(f"SQLite analysis table/hash mismatch: {name}")
    for relative, expected in manifest.files.items():
        path = _relative(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"evidence file/hash mismatch: {path}")


def authenticate_preparation(preparation) -> PreparationManifest:
    preparation = Path(preparation).resolve()
    manifest = load_contract(preparation / "preparation.json", PreparationManifest)
    _authenticate_files(preparation, manifest)
    return manifest


@dataclass(frozen=True)
class BundleLayout:
    bundle: Path
    data: Path
    study: Path
    manifest_path: Path
    manifest: BundleManifest
    plan: Plan[RecommendationProtocol]
    binding: Binding
    raw_root: Path


def _bundle_layout(bundle, raw_root=None) -> BundleLayout:
    bundle = Path(bundle).resolve()
    manifest = read_bundle_manifest(bundle)
    data, study = bundle / "data", bundle / "data/study"
    required = {"study/plan.json", "study/runtime/binding.json", "study/analysis.sqlite"}
    if not required <= manifest.files.keys():
        raise ValueError("portable manifest does not authenticate its plan, binding and SQLite observations")
    plan = read_plan(study / "plan.json")
    if plan.study != "paper_recommendation" or plan.profile not in ("staged", "staged_timing", "recommender_variants"):
        raise ValueError("portable bundle contains an unexpected study")
    if manifest.plan_fingerprint != plan.plan_fingerprint or manifest.planned != len(plan.tasks):
        raise ValueError("portable bundle and frozen plan identity disagree")
    binding = parse_binding(json.loads((study / "runtime/binding.json").read_text()))
    return BundleLayout(bundle, data, study, data / "manifest.json", manifest, plan, binding,
                        Path(raw_root or manifest.source_dataset).expanduser().resolve())


def _binary(binding: Binding, name: str, arm: ArmObservation) -> str:
    expected = binding.arm_binary_sha256.get(name) if binding.arm_binary_sha256 is not None else binding.binary_sha256
    actual = arm.binary_sha256 or expected
    if not expected or actual != expected:
        raise ValueError(f"arm binary provenance mismatch: {name}")
    return actual


@dataclass(frozen=True)
class ArmPath:
    stage: str
    name: str
    observation: ArmObservation
    prefix: str


def _arms(result: StagedResult | FullResult):
    if isinstance(result, StagedResult):
        yield ArmPath("reference", "reference", result.reference, "")
        for stage, block in result.stages.items():
            for name, arm in block.arms.items():
                yield ArmPath(stage, name, arm, result.reference.raw_record)
    else:
        for name, arm in result.arms.items():
            yield ArmPath("full", name, arm, "")


def _prepare_task(layout: BundleLayout, task: Task, result: StagedResult | FullResult,
                  input_fingerprint: str) -> PreparedTaskCheckpoint:
    from .raw import _required_files

    metadata, profile = task.metadata, layout.plan.profile
    task_row = PreparedTaskRow(profile, task.id, layout.plan.plan_fingerprint, fingerprint(task), canonical(metadata))
    points, paths = [], []
    for route in _arms(result):
        arm = route.observation
        identity = PathProvenance(profile, task.id, route.stage, route.name,
                                  layout.plan.plan_fingerprint, _binary(layout.binding, route.name, arm), arm.raw_record)
        raw = _relative(layout.raw_root, arm.raw_record)
        missing = _required_files(raw)
        steps = tuple(point.step for point in arm.endpoints)
        if not steps or any(type(step) is not int or step < 0 for step in steps) or list(steps) != sorted(set(steps)):
            raise ValueError(f"empty, non-unique or unordered observation steps: {identity.key.label}")
        parameters = metadata.RecSysParams or RecommenderParams()
        if profile == "recommender_variants":
            registry = layout.plan.protocol.variant_arms
            if registry is None:
                raise ValueError("variant plan lacks its declared recommendation parameters")
            parameters = parameters.with_overrides(registry[route.name].parameters)
        hk = metadata.HKParams
        if metadata.RNG is None or metadata.PostRetainCount is None or task.sensitivity is None or task.replicate is None:
            raise ValueError("prepared recommendation task lacks its actual seed or coordinates")
        paths.append(PreparedPath(
            identity.profile, identity.task_id, identity.stage, identity.arm, identity.plan_fingerprint,
            identity.binary_sha256, identity.raw_record, str(layout.raw_root), str(raw), steps[0], steps[-1],
            True, route.prefix, hk.Tolerance, metadata.PostRetainCount, hk.Influence, hk.RewiringRate,
            hk.RepostRate, task.sensitivity, task.replicate, str(metadata.RNG.Seed1), str(metadata.RNG.Seed2),
            not missing, canonical(metadata), canonical(metadata.RNG), canonical(parameters), canonical(missing),
            arm.terminal.status, arm.terminal.k_major, canonical(arm.terminal),
        ))
        for point in arm.endpoints:
            state = point.pathway
            values = (state.separation, state.excess_concordance, state.edge_concordance, state.pair_concordance)
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(f"missing/nonfinite recorded observable: {identity.key.label}/{point.step}")
            if not math.isclose(state.excess_concordance, state.edge_concordance - state.pair_concordance, abs_tol=1e-12):
                raise ValueError("recorded excess concordance disagrees with its components")
            points.append(PreparedPoint(
                identity.profile, identity.task_id, identity.stage, identity.arm, identity.plan_fingerprint,
                identity.binary_sha256, identity.raw_record, point.step, point.relative_step,
                state.separation, state.excess_concordance, state.edge_concordance, state.pair_concordance, route.prefix,
            ))
    return PreparedTaskCheckpoint(input_fingerprint, fingerprint(task), True, task_row, tuple(points), tuple(paths))


def _export_frame(output, name, frame, *, writer=None):
    write_table(output, name, frame, writer=writer)
    frame.to_csv(output / name, index=False)


def prepare(bundles, output, *, raw_roots=None) -> PreparationManifest:
    """Extract typed records into a fresh preparation without running a simulation."""
    from experiments.paper.recommendation_shared import storage
    import ehk.domain
    import smp_bindings.domain

    output = Path(output).expanduser().resolve()
    roots = {profile: str(Path(root).expanduser().resolve()) for profile, root in (raw_roots or {}).items()}
    layouts = []
    for source in bundles:
        base = _bundle_layout(source)
        layouts.append(replace(base, raw_root=Path(roots.get(base.plan.profile, str(base.raw_root)))))
    if not layouts or set(roots) - {layout.plan.profile for layout in layouts}:
        raise ValueError("at least one bundle and only selected raw-root profiles are required")
    identities = [(layout.plan.profile, task.id) for layout in layouts for task in layout.plan.tasks]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate task across bundles; choose one generation per study")
    inputs = PreparationInputs(1, tuple(PreparationInput(str(item.bundle), sha(item.manifest_path), item.plan.plan_fingerprint)
                                       for item in layouts), roots, sha(__file__),
        {str(Path(module.__file__).resolve()): sha(module.__file__) for module in (storage,ehk.domain,smp_bindings.domain)})
    if output.exists() and any(output.iterdir()):
        raise ValueError("preparation output already exists; select a new destination")
    output.mkdir(parents=True, exist_ok=True)
    all_points, all_paths, all_tasks, sources = [], [], [], []
    with session_lock(output / "runtime"), DocumentWriter(output / "analysis.sqlite") as writer:
        freeze_manifest(output / "preparation_inputs.json", inputs)
        for layout in layouts:
            available, missing = 0, []
            with StudyStore(layout.study, readonly=True) as store:
                for task in layout.plan.tasks:
                    saved = store.get_task(task, layout.plan)
                    task_row = PreparedTaskRow(layout.plan.profile, task.id, layout.plan.plan_fingerprint,
                                               fingerprint(task), canonical(task.metadata))
                    extracted = (PreparedTaskCheckpoint(fingerprint(inputs), fingerprint(task), False, task_row, (), ())
                                 if saved is None else _prepare_task(layout, task, saved.result, fingerprint(inputs)))
                    all_tasks.append(extracted.task)
                    all_points.extend(extracted.points)
                    all_paths.extend(extracted.paths)
                    available += extracted.available
                    if not extracted.available:
                        missing.append(task.id)
            sources.append(PreparationSource(str(layout.bundle), str(layout.manifest_path), sha(layout.manifest_path),
                           layout.plan.plan_fingerprint, layout.plan.profile, len(layout.plan.tasks), available,
                           tuple(missing), str(layout.raw_root), layout.binding))
        _export_frame(output, "points.csv.gz", rows_frame(all_points, PreparedPoint).reindex(columns=POINT_FIELDS), writer=writer)
        _export_frame(output, "paths.csv", rows_frame(all_paths, PreparedPath).reindex(columns=PATH_FIELDS), writer=writer)
        _export_frame(output, "tasks.csv", rows_frame(all_tasks, PreparedTaskRow), writer=writer)
    unavailable = sum(not path.available for path in all_paths)
    result = PreparationManifest(
        schema=SCHEMA, status="pending_metrics", sources=tuple(sources), points=len(all_points), paths=len(all_paths),
        available_raw_paths=len(all_paths) - unavailable, pending_raw_paths=unavailable,
        missing_metrics=(*POINT_METRICS, "I_w"),
        weighting=Weighting("mean over positive-out-degree users", "mean over directed edges"),
        path_requirement="I_w uses the canonical adaptive sampler on complete 0..end histories, not the 16-step export.",
        files={path.name: sha(path) for path in output.iterdir() if path.is_file() and path.name != "preparation.json"},
    )
    atomic_json(output / "preparation.json", result)
    return result


def _read_points(preparation):
    return read_table(preparation, "points.csv.gz")


def _validate_measurements(incoming, expected, keys, metrics, *, paths=False, endpoint_complement=None):
    if paths:
        incoming = validate_endpoint_column(incoming.copy(), endpoint_complement)
    required = [*keys, *PROVENANCE, *metrics]
    missing = set(required) - set(incoming)
    if missing:
        raise ValueError(f"canonical metric table is missing columns: {sorted(missing)}")
    if incoming.duplicated(keys).any():
        raise ValueError("duplicate canonical metric key")
    if "step" in keys:
        numeric = pd.to_numeric(incoming["step"], errors="raise")
        if not ((numeric >= 0) & (numeric % 1 == 0)).all():
            raise ValueError("measurement step must be a nonnegative integer")
        incoming["step"] = numeric.astype(int)
    joined = incoming.merge(expected[[*keys, *PROVENANCE]], on=keys, how="left", suffixes=("", "_expected"), indicator=True, validate="one_to_one")
    if (joined["_merge"] != "both").any():
        raise ValueError("canonical metric keys are outside the prepared observations")
    for name in PROVENANCE:
        if not (joined[name] == joined[name + "_expected"]).all():
            raise ValueError(f"canonical metric provenance mismatch: {name}")
    for name in (["I_w"] if paths else POINT_METRICS):
        incoming[name] = pd.to_numeric(incoming[name], errors="raise")
        if not incoming[name].map(math.isfinite).all():
            raise ValueError("nonfinite canonical metric; leave missing rows pending instead")
    if paths and len(incoming):
        if not (pd.to_numeric(incoming["prefix_start"], errors="raise") == 0).all():
            raise ValueError("I_w requires the complete prefix starting at zero")
        ends = incoming.merge(expected[[*keys, "end_step"]], on=keys, validate="one_to_one")
        if not (pd.to_numeric(ends["trajectory_end"], errors="raise") == ends["end_step"]).all():
            raise ValueError("I_w trajectory endpoint disagrees with prepared path")
        if not (incoming["sampling_method"] == IW_SAMPLING_METHOD).all():
            raise ValueError("I_w sampling method must use the canonical adaptive calculation")
    return incoming[required]


@dataclass(frozen=True)
class ComparisonInputs:
    preparation_sha256: str
    collection_sha256: str
    implementation_sha256: str


def compare(preparation, collection, output) -> ComparisonManifest:
    preparation, collection, output = (Path(path).resolve() for path in (preparation, collection, output))
    authenticate_preparation(preparation)
    from .contracts import CollectionManifest
    source = load_contract(collection / "collection.json", CollectionManifest)
    _authenticate_files(collection, source)
    if source.preparation_sha256 != sha(preparation / "preparation.json"):
        raise ValueError("canonical collection belongs to a different preparation")
    inputs = ComparisonInputs(sha(preparation / "preparation.json"), sha(collection / "collection.json"), sha(__file__))
    if output.exists() and any(output.iterdir()):
        raise ValueError("alignment output already exists; select a new destination")
    output.mkdir(parents=True, exist_ok=True)
    with session_lock(output / "runtime"):
        freeze_manifest(output / "comparison_inputs.json", inputs)
        coverage = {}
        status = "complete"
        for name, expected, keys, metrics in (
            ("points", _read_points(preparation), POINT_KEYS, POINT_METRICS),
            ("paths", read_table(preparation, "paths.csv"), PATH_KEYS, PATH_METRICS),
        ):
            table = "indices.csv.gz" if name == "points" else "pathways.csv"
            incoming = _validate_measurements(read_table(collection, table), expected, keys, metrics,
                paths=name == "paths", endpoint_complement=source.endpoint_complement)
            joined = expected.merge(incoming[[*keys, *metrics]], on=keys, how="left", validate="one_to_one", indicator="metric_coverage")
            _export_frame(output, f"{name}_aligned.csv.gz", joined)
            pending = int((joined["metric_coverage"] != "both").sum())
            coverage[name] = Coverage(len(expected), len(incoming), pending)
            if pending:
                status = "pending_metrics"
        result = ComparisonManifest(schema=SCHEMA, status=status, coverage=coverage,
            endpoint_complement=source.endpoint_complement, preparation_sha256=inputs.preparation_sha256,
            collection_sha256=inputs.collection_sha256,
            files={path.name: sha(path) for path in output.iterdir() if path.is_file()})
        atomic_json(output / "comparison.json", result)
        return result

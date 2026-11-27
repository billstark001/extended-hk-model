"""Validate binned KDE on frozen exact histories without changing model metrics."""

import hashlib
import inspect
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from common.experiments.lifecycle import interruptible, session_lock
from common.experiments.plans import freeze_manifest
from common.io.json import atomic_json, fingerprint, read_result, save_result, sha
from common.runtime.process import run_tasks
from experiments.paper.metric_comparison import core, raw
from experiments.paper.metric_comparison import report as baseline_reporting
from experiments.paper.metric_comparison.evaluation import ExactMetricCache

PATH_KEYS = ("profile", "task_id", "stage", "arm")
DEFAULT_BINS = (1021, 2041, 4081, 8161)
METRICS = ("Ip", "Is", "Ih")


def _calculator():
    from .kernel import BinnedDistanceCalculator

    return BinnedDistanceCalculator()


def _source_configuration():
    import networkx
    import smp_bindings.events_db
    import smp_bindings.model_state
    import smp_bindings.trajectory

    from ehk.micro import observables

    from . import kernel

    sources = {Path(__file__), Path(__file__).with_name("__main__.py"), Path(kernel.__file__),
               Path(inspect.getfile(observables.get_indices)), Path(inspect.getfile(ExactMetricCache)),
               Path(smp_bindings.events_db.__file__), Path(smp_bindings.model_state.__file__),
               Path(smp_bindings.trajectory.__file__)}
    return {
        "sources": {str(path.resolve()): sha(path) for path in sorted(sources)},
        "exact_protocol": raw._source_configuration(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "networkx": networkx.__version__},
        "kernel": "linear_binned_gaussian", "seed": "absolute completed step",
        "adaptive": {"tolerance": .01, "max_interval": 512, "full_prefix": True},
        "activity": {"active_threshold": .98, "min_inactive_value": .75},
        "baseline_activity_audit": "all selected paths; prepared lookup plus exact off-grid accepted steps",
        "timing": "uncached paired same-state exact/binned totals; each includes its own prepare",
    }


def _read_csv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip", low_memory=False)


def _load_baseline(preparation, baseline):
    points, paths, provenance = baseline_reporting.load_aligned(preparation, baseline / "alignment")
    manifests = {}
    policies = [core.endpoint_mode(provenance["comparison"])]
    for directory, name in ((baseline, "run_status.json"), (baseline / "collection", "collection.json"),
                            (baseline / "report", "report.json")):
        manifest = json.loads((directory / name).read_text())
        if manifest.get("state", manifest.get("status")) != "complete":
            raise ValueError(f"exact baseline is incomplete: {directory / name}")
        if name != "run_status.json":
            core._authenticate_files(directory, manifest)
        policies.append(core.endpoint_mode(manifest))
        manifests[name] = sha(directory / name)
    source = json.loads((baseline / "collection/configuration.json").read_text())
    endpoint_complement = core.endpoint_mode(source)
    if any(policy != endpoint_complement for policy in policies):
        raise ValueError("exact baseline endpoint_complement policies disagree")
    if source["preparation_fingerprint"] != fingerprint(provenance["preparation"]):
        raise ValueError("baseline collection belongs to a different preparation")
    if source["metrics"] != raw._source_configuration():
        raise ValueError("current exact metric/reader sources differ from the frozen baseline")
    return points, paths, {
        "endpoint_complement": endpoint_complement,
        "preparation_sha256": sha(preparation / "preparation.json"),
        "run_status_sha256": manifests["run_status.json"],
        "collection_sha256": manifests["collection.json"],
        "collection_configuration_fingerprint": fingerprint(source),
        "comparison_sha256": sha(baseline / "alignment/comparison.json"),
        "baseline_report": {"directory": str(baseline / "report"), "sha256": manifests["report.json"]},
    }


def _features(block):
    features = set()
    for row in block.to_dict("records"):
        profile = row["profile"]
        for field in ("profile", "stage", "arm", "epsilon", "history", "repost", "sensitivity"):
            if field in row:
                features.add((field, str((profile, row[field]))))
        features.add(("setting", str((profile, row.get("influence"), row.get("rewiring")))))
        features.add(("condition", str((profile, row.get("sensitivity"), row.get("epsilon"), row.get("history")))))
        terminal = json.loads(row.get("terminal_json", "{}"))
        features.add(("terminal", str((profile, terminal.get("status"), terminal.get("k_major")))))
    return features


def select_tasks(paths, points, max_tasks=32, *, full=False):
    """Freeze deterministic coverage of existing groups, never partial task arms."""
    groups = {tuple(key): block for key, block in paths.groupby(["profile", "task_id"], sort=True)}
    feature_sets = {key: _features(block) for key, block in groups.items()}
    # Prioritize groups near the existing activity threshold and a midpoint-error
    # proxy on the saved grid. The proxy is not a reconstructed adaptive decision.
    margins = {}
    for key, block in points.groupby(["profile", "task_id"], sort=True):
        activity, adaptive = float("inf"), float("inf")
        for _, curve in block.groupby(["stage", "arm"], sort=False):
            curve = curve.sort_values("step")
            values = curve[["I_p", "I_s", "I_h"]].to_numpy(float)
            threshold = max(.98 * values[-1, 1], .75)
            activity = min(activity, float(np.min(np.abs(values[:, 1] - threshold))))
            if len(values) > 2:
                steps = curve["step"].to_numpy(float)
                fraction = ((steps[1:-1] - steps[:-2]) / (steps[2:] - steps[:-2]))[:, None]
                residual = np.max(np.abs(values[1:-1] - values[:-2] * (1 - fraction) - values[2:] * fraction), axis=1)
                adaptive = min(adaptive, float(np.min(np.abs(residual - .01))))
        margins[tuple(key)] = activity, adaptive
    for index, field in enumerate(("near_activity", "saved_grid_midpoint_proxy")):
        ordered = sorted(groups, key=lambda key: (margins.get(key, (np.inf, np.inf))[index], key))
        for key in ordered[:min(4, len(ordered))]:
            feature_sets[key].add((field, "selected_stress_group"))
    selected, covered = [], set()
    remaining = set(groups)
    limit = len(groups) if full else min(max_tasks, len(groups))
    while remaining and len(selected) < limit:
        key = min(remaining, key=lambda item: (-len(feature_sets[item] - covered), item))
        selected.append(key)
        covered.update(feature_sets[key])
        remaining.remove(key)
    selected.sort()
    selected_keys = set(selected)
    rows = paths[[key in selected_keys for key in zip(paths["profile"], paths["task_id"])]].copy()
    selection = {
        "mode": "full" if full else "representative", "max_tasks": None if full else max_tasks,
        "available_tasks": len(groups), "selected_tasks": len(selected),
        "available_paths": len(paths), "selected_paths": len(rows),
        "task_keys": [list(key) for key in selected],
        "profile_tasks": {profile: sum(key[0] == profile for key in selected) for profile in sorted(paths["profile"].unique())},
        "path_keys": rows[list(PATH_KEYS)].to_dict("records"),
        "covered_features": [list(item) for item in sorted(covered)],
        "uncovered_features": [list(item) for item in sorted(set().union(*feature_sets.values()) - covered)],
        "selection_rule": "greedy existing-design/state coverage; deterministic lexical ties",
        "adaptive_stress_rule": "existing-grid midpoint residual near .01; proxy, not original rejected candidates",
    }
    return rows, selection


def _benchmark_keys(points, paths, count):
    """Select fixed saved observations across task groups before timing begins."""
    metadata = paths[[*PATH_KEYS, "end_step"]]
    candidates = points.merge(metadata, on=list(PATH_KEYS), validate="many_to_one")
    endings = candidates[candidates["step"] == candidates["end_step"]]
    terminal_s = endings[[*PATH_KEYS, "I_s"]].rename(columns={"I_s": "terminal_Is"})
    candidates = candidates.merge(terminal_s, on=list(PATH_KEYS), validate="many_to_one")
    thresholds = np.maximum(.98 * candidates["terminal_Is"].to_numpy(float), .75)
    candidates["activity_margin"] = np.abs(candidates["I_s"].to_numpy(float) - thresholds)

    def ordered():
        # A near-threshold state and an endpoint per initial state precede fillers.
        for _, block in candidates.groupby(["profile", "task_id"], sort=True):
            yield block.sort_values(["activity_margin", *PATH_KEYS, "step"]).iloc[0]
        for _, block in endings.groupby(["profile", "task_id"], sort=True):
            yield block.sort_values([*PATH_KEYS, "step"]).iloc[-1]
        yield from (row for _, row in candidates.sort_values([*PATH_KEYS, "step"]).iterrows())

    result, seen = [], set()
    for row in ordered():
        identity = tuple(row[name] for name in (*PATH_KEYS, "step"))
        if identity not in seen:
            seen.add(identity)
            result.append({**{name: row[name] for name in PATH_KEYS}, "step": int(row["step"])})
        if len(result) >= count:
            break
    return result


class CandidateEvaluator(ExactMetricCache):
    """Use the unchanged adaptive driver with approximate distance and exact I_h."""

    def __init__(self, bins, shared):
        super().__init__()
        self.bins, self.shared = bins, shared
        self.diagnostics = {}
        self._diagnostic_keys = {}

    def _values(self, record, step):
        key = self._key(record, int(step))
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            self.diagnostics[(record.directory, int(step))] = self._diagnostic_keys[key]
            return self._entries[key]
        self.misses += 1
        prepared, ih, input_sha = self.shared.prepare(record, int(step), key)
        started = time.perf_counter()
        ip, isub, diagnostics = self.shared.calculator.evaluate(prepared, self.bins)
        self.shared.evaluate_seconds[self.bins] += time.perf_counter() - started
        values = float(ip), float(isub), float(ih)
        if not np.isfinite(values).all():
            raise ValueError("approximate metric is nonfinite")
        self._entries[key] = values
        self.diagnostics[(record.directory, int(step))] = {**diagnostics, "same_input_sha": input_sha}
        self._diagnostic_keys[key] = self.diagnostics[(record.directory, int(step))]
        if len(self._entries) > self.max_entries:
            dropped, _ = self._entries.popitem(last=False)
            self._diagnostic_keys.pop(dropped, None)
        return values


class SharedInputs:
    def __init__(self, calculator, bins):
        self.calculator = calculator
        self.inputs = OrderedDict()
        self.prepare_seconds = 0.
        self.evaluate_seconds = dict.fromkeys(bins, 0.)
        self.keyer = ExactMetricCache()
        self.benchmarked = set()

    def prepare(self, record, step, key=None):
        from ehk.micro import observables

        key = self.keyer._key(record, step) if key is None else key
        if key in self.inputs:
            self.inputs.move_to_end(key)
            return self.inputs[key]
        started = time.perf_counter()
        prepared = self.calculator.prepare(record.get_graph(step), record.opinions[step],
                                           float(record.metadata["HKParams"]["Tolerance"]), step)
        self.prepare_seconds += time.perf_counter() - started
        value = prepared, float(observables.calc_homophily(record, step)), _input_sha(key)
        self.inputs[key] = value
        if len(self.inputs) > 16:
            self.inputs.popitem(last=False)
        return value


def _input_sha(key):
    digest = hashlib.sha256()
    for item in key:
        data = item if isinstance(item, bytes) else repr(item).encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _activity(steps, isub):
    from ehk.micro.observables import calc_active_step

    _, step, threshold, _, _ = calc_active_step(np.asarray(steps), np.asarray(isub), .98, .75)
    return int(step), float(threshold)


def _iw(p, h, *, endpoint_complement=True):
    from ehk.micro.observables import get_gradation_index_hp

    return float(get_gradation_index_hp(np.asarray(p), np.asarray(h), endpoint_complement=endpoint_complement))


def _timing(record, step, bins, repeats, calculator, identity):
    graph, opinions = record.get_graph(step), record.opinions[step]
    epsilon = float(record.metadata["HKParams"]["Tolerance"])
    key = ExactMetricCache()._key(record, step)
    digest = _input_sha(key)
    rows = []
    for repeat in range(repeats):
        for resolution in bins:
            measured = {}
            for method in (("exact", "binned") if repeat % 2 == 0 else ("binned", "exact")):
                started = time.perf_counter()
                prepared = calculator.prepare(graph, opinions, epsilon, step)
                prepared_at = time.perf_counter()
                values = (calculator.exact(prepared) if method == "exact"
                          else calculator.evaluate(prepared, resolution))
                finished_at = time.perf_counter()
                measured[method] = (values, prepared_at - started, finished_at - prepared_at, finished_at - started)
            ((exact_p, exact_s, _), exact_prepare, exact_evaluate, exact_total) = measured["exact"]
            ((ip, isub, diagnostics), prepare, evaluate, total) = measured["binned"]
            rows.append({
                **identity, "case_id": fingerprint({**identity, "step": step}), "step": step,
                "bins": resolution, "repeat": repeat, "same_input_sha": digest,
                "timing_order": "exact_first" if repeat % 2 == 0 else "binned_first",
                "exact_prepare_seconds": exact_prepare, "exact_evaluate_seconds": exact_evaluate,
                "exact_total_seconds": exact_total, "prepare_seconds": prepare,
                "evaluate_seconds": evaluate, "binned_total_seconds": total,
                "exact_Ip": float(exact_p), "exact_Is": float(exact_s),
                "approx_Ip": float(ip), "approx_Is": float(isub), **diagnostics,
            })
    return rows


def _measure_path(entry, bins, repeats, shared, evaluators, *, endpoint_complement=True):
    from smp_bindings import RawSimulationRecord

    path, baseline_points, baseline_receipt = entry["path"], entry["points"], entry["baseline"]
    if core.endpoint_mode(baseline_receipt["result"]["path"]) != endpoint_complement:
        raise ValueError("exact baseline receipt endpoint_complement differs from the selected policy")
    identity = {name: path[name] for name in PATH_KEYS}
    directory = Path(path["raw_directory"])
    metadata = raw._validate_raw(directory, path, pd.DataFrame(baseline_points))
    exact = {int(point["step"]): tuple(float(point[name]) for name in ("I_p", "I_s", "I_h"))
             for point in baseline_points}
    baseline_steps = baseline_receipt["result"]["evaluation"]["adaptive_steps"]
    if (not baseline_steps or baseline_steps[0] != 0 or baseline_steps[-1] != int(path["end_step"])
            or baseline_steps != sorted(set(baseline_steps))):
        raise ValueError("baseline receipt has no valid full adaptive grid")
    extra_calls, extra_seconds = 0, 0.
    rows, path_rows, timings = [], [], []
    with RawSimulationRecord(str(directory.parent), metadata) as record:
        for evaluator in evaluators.values():
            evaluator.diagnostics.clear()
        for step in baseline_steps:
            if step not in exact:
                started = time.perf_counter()
                prepared, ih, _ = shared.prepare(record, int(step))
                p, s, _ = shared.calculator.exact(prepared)
                extra_seconds += time.perf_counter() - started
                extra_calls += 1
                exact[step] = float(p), float(s), ih
        exact_curve = np.array([exact[step] for step in baseline_steps])
        baseline_iw = float(baseline_receipt["result"]["path"]["I_w"])
        reconstructed = _iw(exact_curve[:, 0], exact_curve[:, 2], endpoint_complement=endpoint_complement)
        if not np.isclose(reconstructed, baseline_iw, rtol=0, atol=64 * np.finfo(float).eps):
            raise ValueError("reconstructed exact accepted curve differs from its frozen I_w")
        active_step, active_threshold = _activity(baseline_steps, exact_curve[:, 1])
        for resolution, evaluator in evaluators.items():
            started_path = time.perf_counter()
            before_prepare, before_evaluate = shared.prepare_seconds, shared.evaluate_seconds[resolution]
            point_values = {}
            for grid, steps in (("prepared", [int(point["step"]) for point in baseline_points]),
                                ("baseline_adaptive", baseline_steps)):
                for step in steps:
                    values = (*evaluator.distance(record, step), evaluator.homophily(record, step))
                    point_values[step] = values
                    reference = exact[step]
                    if values[2] != reference[2]:
                        raise ValueError("KDE experiment changed exact homophily")
                    rows.append({
                        **identity, "bins": resolution, "step": step, "grid": grid,
                        **{f"exact_{name}": reference[index] for index, name in enumerate(METRICS)},
                        **{f"approx_{name}": values[index] for index, name in enumerate(METRICS)},
                        **evaluator.diagnostics.get((record.directory, step), {}),
                    })
            steps, h, p, s = evaluator.indices(record)
            for index, step in enumerate(steps):
                step = int(step)
                reference = exact.get(step, (None, None, float(h[index])))
                rows.append({
                    **identity, "bins": resolution, "step": step, "grid": "candidate_adaptive",
                    **{f"exact_{name}": reference[i] for i, name in enumerate(METRICS)},
                    "approx_Ip": float(p[index]), "approx_Is": float(s[index]), "approx_Ih": float(h[index]),
                    **evaluator.diagnostics.get((record.directory, step), {}),
                })
            fixed = np.array([point_values[step] for step in baseline_steps])
            fixed_active, fixed_threshold = _activity(baseline_steps, fixed[:, 1])
            adaptive_active, adaptive_threshold = _activity(steps, s)
            path_rows.append({
                **identity, "bins": resolution, "start_step": int(path["start_step"]), "end_step": int(path["end_step"]),
                "endpoint_complement": endpoint_complement,
                "baseline_Iw": baseline_iw,
                "fixed_Iw": _iw(fixed[:, 0], fixed[:, 2], endpoint_complement=endpoint_complement),
                "adaptive_Iw": _iw(p, h, endpoint_complement=endpoint_complement),
                "baseline_steps_json": json.dumps(baseline_steps),
                "candidate_steps_json": json.dumps([int(step) for step in steps]),
                "baseline_active_step": active_step, "fixed_active_step": fixed_active,
                "adaptive_active_step": adaptive_active, "baseline_active_threshold": active_threshold,
                "fixed_active_threshold": fixed_threshold, "adaptive_active_threshold": adaptive_threshold,
                "activity_audit_status": "measured", "extra_exact_calls": extra_calls,
                "extra_exact_seconds": extra_seconds,
                "shared_prepare_seconds": shared.prepare_seconds - before_prepare,
                "evaluate_seconds": shared.evaluate_seconds[resolution] - before_evaluate,
                "candidate_path_seconds": time.perf_counter() - started_path,
            })
        for step in entry["benchmark_steps"]:
            key = shared.keyer._key(record, int(step))
            if key not in shared.benchmarked:
                shared.benchmarked.add(key)
                timings.extend(_timing(record, int(step), bins, repeats, shared.calculator, identity))
    return {"points": rows, "paths": path_rows, "benchmarks": timings}


def _collect_group(job):
    shared = SharedInputs(_calculator(), job["bins"])
    evaluators = {bins: CandidateEvaluator(bins, shared) for bins in job["bins"]}
    statuses = []
    for entry in job["entries"]:
        identity = {name: entry["path"][name] for name in PATH_KEYS}
        receipt = Path(job["output"]) / "records" / (fingerprint(identity) + ".json.gz")
        directory = Path(entry["path"]["raw_directory"])
        if raw._required_files(directory):
            statuses.append({**identity, "status": "pending", "missing_files": raw._required_files(directory)})
            continue
        source = raw._raw_sources(directory)
        if source != entry["baseline"]["raw_sha256"]:
            raise ValueError("raw inputs differ from the frozen exact baseline")
        if receipt.is_file():
            saved = read_result(receipt)
            if (saved["configuration_fingerprint"] != job["configuration_fingerprint"] or saved["identity"] != identity
                    or saved["raw_sha256"] != source or saved["baseline_receipt_sha256"] != entry["baseline_receipt_sha256"]):
                raise ValueError("approximation receipt differs from the frozen experiment")
            statuses.append({**identity, "status": "cached"})
            continue
        try:
            result = _measure_path(entry, job["bins"], job["repeats"], shared, evaluators,
                                   endpoint_complement=job["endpoint_complement"])
            if raw._raw_sources(directory) != source:
                raise ValueError("raw inputs changed during approximation measurement")
        except (OSError, ValueError, KeyError) as error:
            statuses.append({**identity, "status": "failed", "error": str(error)})
            continue
        save_result(receipt, {"identity": identity, "configuration_fingerprint": job["configuration_fingerprint"],
                              "endpoint_complement": job["endpoint_complement"],
                              "baseline_receipt_sha256": entry["baseline_receipt_sha256"],
                              "raw_sha256": source, "result": result})
        statuses.append({**identity, "status": "complete"})
    return statuses


def collect(preparation, baseline, output, *, bins=DEFAULT_BINS, workers=4, max_tasks=32, full=False,
            benchmark_states=64, repeats=3, endpoint_complement=True):
    """Measure a frozen representative/full cohort and resume only matching receipts."""
    if (type(workers) is not int or not 1 <= workers <= 8 or type(max_tasks) is not int or max_tasks < 1
            or type(benchmark_states) is not int or benchmark_states < 1 or type(repeats) is not int or repeats < 1):
        raise ValueError("workers must be 1..8; task, benchmark and repeat counts must be positive integers")
    bins = tuple(bins)
    if not bins or len(set(bins)) != len(bins) or any(type(value) is not int or value not in DEFAULT_BINS for value in bins):
        raise ValueError("bins must be distinct supported resolutions")
    preparation, baseline, output = (Path(path).expanduser().resolve() for path in (preparation, baseline, output))
    if any(output == source or output in source.parents or source in output.parents for source in (preparation, baseline)):
        raise ValueError("approximation output must be separate from frozen evidence")
    points, paths, evidence = _load_baseline(preparation, baseline)
    if core.endpoint_mode(evidence) != endpoint_complement:
        raise ValueError("frozen exact baseline endpoint_complement differs from the selected policy; choose matching evidence")
    from ehk.metrics.protocol import metric_protocol
    selected, selection = select_tasks(paths, points, max_tasks, full=full)
    keys = {tuple(row[name] for name in PATH_KEYS) for row in selected.to_dict("records")}
    selected_points = points[points.apply(lambda row: tuple(row[name] for name in PATH_KEYS) in keys, axis=1)]
    benchmarks = _benchmark_keys(selected_points, selected, benchmark_states)
    configuration = {
        "schema": 1, "preparation": str(preparation), "baseline": str(baseline), **evidence,
        "selection": selection, "selected_paths": selection["path_keys"], "bins": list(bins),
        "benchmark_states": benchmarks, "repeats": repeats, "metrics": _source_configuration(),
        "endpoint_complement": endpoint_complement, "metric_protocol": metric_protocol(endpoint_complement=endpoint_complement),
    }
    groups = selected_points.groupby(list(PATH_KEYS), sort=False).indices
    outcomes = {}
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)

    def jobs():
        for (profile, task_id), block in selected.groupby(["profile", "task_id"], sort=True):
            entries = []
            for path in block.to_dict("records"):
                identity = {name: path[name] for name in PATH_KEYS}
                baseline_identity = {**identity, **{name: path[name] for name in raw.SOURCE_KEYS}}
                receipt = baseline / "collection/records" / (fingerprint(baseline_identity) + ".json.gz")
                saved = read_result(receipt)
                if (saved["identity"] != baseline_identity or
                        saved["configuration_fingerprint"] != evidence["collection_configuration_fingerprint"]):
                    raise ValueError("exact path receipt differs from the authenticated baseline")
                key = tuple(identity[name] for name in PATH_KEYS)
                entries.append({
                    "path": path, "points": selected_points.iloc[groups[key]].to_dict("records"),
                    "baseline": saved, "baseline_receipt_sha256": sha(receipt),
                    "benchmark_steps": [int(row["step"]) for row in benchmarks
                                        if all(row[name] == identity[name] for name in PATH_KEYS)],
                })
            yield {"id": (profile, task_id), "bins": bins, "repeats": repeats,
                   "endpoint_complement": endpoint_complement,
                   "configuration_fingerprint": fingerprint(configuration), "entries": entries, "output": str(output)}

    def progress(state):
        counts = {kind: sum(row["status"] == kind for row in outcomes.values())
                  for kind in ("complete", "cached", "pending", "failed")}
        atomic_json(output / "progress.json", {"schema": 1, "state": state, "workers": workers,
                    "planned_paths": len(selected), "completed_paths": counts["complete"] + counts["cached"],
                    "counts": counts, "elapsed_seconds": time.monotonic() - started})

    def finished(job, statuses):
        for row in statuses:
            outcomes[tuple(row[name] for name in PATH_KEYS)] = row
        progress("running")

    failure = None
    with session_lock(output / "runtime"), interruptible():
        freeze_manifest(output / "configuration.json", configuration)
        progress("running")
        try:
            if workers == 1:
                for job in jobs():
                    finished(job, _collect_group(job))
            else:
                run_tasks(_collect_group, jobs(), workers, executor="process", on_result=finished)
        except BaseException as error:
            failure = error
        rows = {"points": [], "paths": [], "benchmarks": []}
        statuses = []
        for identity in selection["path_keys"]:
            key = tuple(identity[name] for name in PATH_KEYS)
            receipt = output / "records" / (fingerprint(identity) + ".json.gz")
            state = outcomes.get(key, {**identity, "status": "pending"})
            if receipt.is_file():
                saved = read_result(receipt)
                if saved["configuration_fingerprint"] != fingerprint(configuration) or saved["identity"] != identity:
                    raise ValueError("receipt does not belong to this approximation configuration")
                for table in rows:
                    rows[table].extend(saved["result"][table])
                if state["status"] == "pending":
                    state = {**identity, "status": "complete"}
            statuses.append(state)
        # Explicit headers preserve readable pending tables without inventing values.
        for name, data in rows.items():
            fields = [*PATH_KEYS, "bins"]
            if name == "points":
                fields += ["step", "grid", *[f"{kind}_{metric}" for kind in ("exact", "approx") for metric in METRICS]]
            elif name == "paths":
                fields += ["end_step", "baseline_Iw", "fixed_Iw", "adaptive_Iw", "baseline_steps_json", "candidate_steps_json"]
            else:
                fields += ["case_id", "step", "repeat", "exact_total_seconds", "binned_total_seconds"]
            frame = pd.DataFrame(data, columns=list(dict.fromkeys([*fields, *[key for row in data for key in row]])))
            if name == "benchmarks" and len(frame):
                frame = frame.drop_duplicates(["same_input_sha", "repeat", "bins"])
            frame.to_csv(
                output / (name + (".csv" if name == "benchmarks" else ".csv.gz")), index=False,
            )
        complete = sum(row["status"] in ("complete", "cached") for row in statuses)
        failed = sum(row["status"] == "failed" for row in statuses)
        state = (("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed") if failure else
                 ("complete" if complete == len(selected) else ("failed" if failed else "pending")))
        manifest = {"schema": 1, "status": state, "planned_paths": len(selected), "completed_paths": complete,
                    "endpoint_complement": endpoint_complement,
                    "pending_paths": len(selected) - complete - failed, "failed_paths": failed,
                    "selection": selection, "paths": statuses,
                    "configuration_sha256": sha(output / "configuration.json"),
                    "files": {name: sha(output / name) for name in ("points.csv.gz", "paths.csv.gz", "benchmarks.csv")}}
        atomic_json(output / "collection.json", manifest)
        progress(state)
        if failure:
            raise failure
        return manifest

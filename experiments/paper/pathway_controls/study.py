"""Freeze, execute and export raw paired records; scientific metrics stay optional."""

import csv
import json
from pathlib import Path

from common.experiments.plans import freeze_manifest, freeze_plan, read_plan
from common.io.json import atomic_json, fingerprint, read_result, save_result, sha
from ehk.micro.persistent import observe_persistent

from ..recommendation_shared import runtime
from ..study import execution_provenance
from .design import ARMS, arm_metadata, make_plan


def freeze(directory, binary, **options):
    directory = Path(directory).resolve()
    plan = freeze_plan(directory / "plan.json", make_plan(**options))
    freeze_manifest(directory / "runtime" / "binding.json", execution_provenance(binary))
    return plan


def _checkpoint(directory, task, arm, step):
    name = arm_metadata(task, arm)["UniqueName"]
    return directory / "raw" / name / f"checkpoint-{step:09d}.msgpack.lz4"


def _physical(path):
    from smp_bindings.model_state import load_snapshot

    return load_snapshot(str(path))


def _initial_pair(directory, task, arms):
    initial = None
    for arm in arms:
        state = _physical(_checkpoint(directory, task, arm, 0))
        if initial is None:
            initial = state
        elif state != initial:
            raise ValueError(f"process controls have different initial microscopic/RNG states: {task['id']}")


def run_study(directory, binary, workers=4, **options):
    directory, binary = Path(directory).resolve(), Path(binary).resolve()
    if not 1 <= workers <= 6:
        raise ValueError("controls support 1 through 6 concurrent pairs")
    plan = (
        read_plan(directory / "plan.json")
        if (directory / "plan.json").is_file() and not options
        else freeze(directory, binary, **options)
    )
    if plan.get("study") != "paper_pathway_controls":
        raise ValueError("directory contains a different experiment")

    def run_task(task, frozen, binding, previous, progress):
        arms = dict((previous or {}).get("arms", {}))
        if arms.keys() - set(ARMS):
            raise ValueError("record contains an undeclared process control")
        for arm in ARMS:
            if arm not in arms or not runtime.complete_raw(directory, arms[arm]):
                result = observe_persistent(
                    binary,
                    directory / "raw",
                    arm_metadata(task, arm),
                    frozen["protocol"]["snapshot_steps"],
                    fixed_horizon=True,
                )
                raw = Path(result.pop("run_directory"))
                result["raw_record"] = str(raw.relative_to(directory))
                result["binary_sha256"] = binding["binary_sha256"]
                arms[arm] = result
            _initial_pair(directory, task, arms)
            progress({"arms": arms})
        return {"arms": arms}

    return runtime.run_dataset(
        directory,
        binary,
        workers,
        plan_factory=lambda: plan,
        binding_factory=lambda plan, path, binary_sha: execution_provenance(binary),
        task_runner=run_task,
        result_arms=lambda result: result["arms"].values(),
        resume=True,
        cache_valid=lambda result, binding: (
            set(result["arms"]) == set(ARMS)
            and all(arm["binary_sha256"] == binding["binary_sha256"] for arm in result["arms"].values())
        ),
    )


def snapshot_input(path):
    """Export complete directed edges, node opinions and retained post records."""
    state = _physical(path)
    data = state["data"]
    graph = data["Graph"]
    edges = [[int(source), int(target)] for source, targets in graph["adjacency"].items() for target in targets]
    return {
        "schema": 1,
        "step": data["CurStep"] - 1,
        "dynamics": state["dynamics_type"],
        "node_ids": list(range(len(data["Opinions"]))),
        "opinions": data["Opinions"],
        "edges": sorted(edges),
        "posts": [{"sharer": int(sharer), "entries": entries} for sharer, entries in sorted(data["Posts"].items())],
    }


def export_snapshots(directory):
    directory = Path(directory).resolve()
    plan = read_plan(directory / "plan.json")
    rows = []
    for task in plan["tasks"]:
        _initial_pair(directory, task, ARMS)
        for arm in ARMS:
            for step in plan["protocol"]["snapshot_steps"]:
                source = _checkpoint(directory, task, arm, step)
                snapshot = snapshot_input(source)
                if snapshot["step"] != step:
                    raise ValueError(f"checkpoint has the wrong completed step: {source}")
                relative = Path("snapshot_inputs") / task["id"] / arm / f"step-{step:09d}.json.gz"
                save_result(directory / relative, snapshot)
                rows.append(
                    {
                        "pair_id": task["id"],
                        **task["coordinates"],
                        "arm": arm,
                        "step": step,
                        "checkpoint": str(source.relative_to(directory)),
                        "checkpoint_sha256": sha(source),
                        "data_file": str(relative),
                        "state_fingerprint": fingerprint(snapshot),
                    }
                )
    index = {"schema": 1, "plan_fingerprint": plan["plan_fingerprint"], "snapshots": rows}
    atomic_json(directory / "snapshot_inputs" / "index.json", index)
    return index


def analyze(directory):
    """Recompute the paper's Ip/Is/Ih/Iw from actual raw metadata and trajectories."""
    import numpy as np
    from smp_bindings import RawSimulationRecord

    from ehk.micro import observables
    from ehk.micro.observables import get_gradation_index_hp, get_indices

    directory = Path(directory).resolve()
    index = export_snapshots(directory)
    plan = read_plan(directory / "plan.json")
    rows, summaries = [], []
    for task in plan["tasks"]:
        for arm in ARMS:
            requested = arm_metadata(task, arm)
            path = directory / "raw" / requested["UniqueName"] / "metadata.json"
            metadata = json.loads(path.read_text())
            for key in ("RNG", "HKParams", "NodeCount", "RecsysFactoryType", "RecSysParams"):
                if metadata.get(key) != requested[key]:
                    raise ValueError(f"raw control metadata differs in {key}: {path}")
            with RawSimulationRecord(str(directory / "raw"), metadata) as record:
                if record.max_step != plan["protocol"]["horizon"]:
                    raise ValueError(f"control record did not reach its frozen horizon: {path}")
                steps, h, p, s = get_indices(record)
                for step, ih, ip, is_ in zip(steps, h, p, s, strict=True):
                    rows.append(
                        {
                            "pair_id": task["id"],
                            **task["coordinates"],
                            "arm": arm,
                            "step": int(step),
                            "I_p": float(ip),
                            "I_s": float(is_),
                            "I_h": float(ih) if np.isfinite(ih) else None,
                        }
                    )
                summaries.append(
                    {
                        "pair_id": task["id"],
                        **task["coordinates"],
                        "arm": arm,
                        "horizon": record.max_step,
                        "I_w": float(get_gradation_index_hp(p, h)),
                        "sampling": "paper adaptive grid",
                        "endpoint_complement": False,
                    }
                )
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("indices.csv", rows), ("pathway_integrals.csv", summaries)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    atomic_json(
        output / "receipt.json",
        {
            "plan_fingerprint": plan["plan_fingerprint"],
            "snapshots": len(index["snapshots"]),
            "metric_implementation_sha256": sha(observables.__file__),
            "classification": None,
            "metrics": ["I_p", "I_s", "I_h", "I_w"],
        },
    )
    return {"snapshots": len(index["snapshots"]), "index_rows": len(rows), "arms": len(summaries)}


def plot_snapshots(directory, output=None):
    """Preview network inputs with one initial-state layout shared by every arm."""
    import matplotlib.pyplot as plt
    import networkx as nx

    directory = Path(directory).resolve()
    index = export_snapshots(directory)
    output = Path(output).resolve() if output is not None else directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for pair in dict.fromkeys(row["pair_id"] for row in index["snapshots"]):
        rows = [row for row in index["snapshots"] if row["pair_id"] == pair]
        first = read_result(directory / rows[0]["data_file"])
        graph = nx.DiGraph()
        graph.add_nodes_from(first["node_ids"])
        graph.add_edges_from(first["edges"])
        positions = nx.spring_layout(graph.to_undirected(), seed=0)
        steps = sorted({row["step"] for row in rows})
        figure, axes = plt.subplots(len(ARMS), len(steps), squeeze=False, figsize=(3 * len(steps), 8))
        for row in rows:
            snapshot = read_result(directory / row["data_file"])
            graph = nx.DiGraph()
            graph.add_nodes_from(snapshot["node_ids"])
            graph.add_edges_from(snapshot["edges"])
            axis = axes[ARMS.index(row["arm"]), steps.index(row["step"])]
            nx.draw_networkx(
                graph,
                pos=positions,
                ax=axis,
                node_color=snapshot["opinions"],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                node_size=12,
                width=0.2,
                alpha=0.7,
                arrows=False,
                with_labels=False,
            )
            axis.set_title(f"{row['arm']}, step {row['step']}")
            axis.set_axis_off()
        figure.suptitle(pair)
        figure.tight_layout()
        figure.savefig(output / (pair + ".png"), dpi=140)
        plt.close(figure)
    return output

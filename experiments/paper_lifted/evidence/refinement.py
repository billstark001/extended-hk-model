"""Data-dependent follow-up design; no task execution or worker scheduling."""

from pathlib import Path

from common.experiments.plans import freeze_plan, read_plan
from common.io.json import sha
from common.runtime.failures import ProtocolError
from lifted.runtime.records import read
from theory.parameters import Parameters

from .analyze import probability_groups
from .design import Design, chunked


def freeze_followup(root, plan, tasks, *, read_result):
    """Keep saved follow-ups immutable; otherwise freeze selection from committed inputs."""
    root = Path(root)

    path = root / "followup" / "plan.json"
    if path.exists():
        followup = read_plan(path)
        for relative, digest in followup["inputs"].items():
            if sha(root / relative) != digest:
                raise ProtocolError("follow-up selection inputs changed")
    else:
        d = Design("formal")
        for task in list(tasks.values()):
            if task["kind"] not in ("micro_probe", "meso_probe", "micro_tail"):
                continue
            if task["variant"] != "stochastic":
                continue
            result = read_result(task)
            if not result:
                continue
            for row in result["paths"]:
                if row["category"] != "unresolved":
                    continue
                p = Parameters(**task["parameters"])
                d.add(
                    "tails",
                    task["kind"].replace("probe", "extend").replace("tail", "extend"),
                    p,
                    parent=task["id"],
                    initial=task["initial"],
                    mother=task["mother"],
                    at=task["source_step"],
                    start=row["replicate"],
                    paths=1,
                    steps=24000,
                    variant=task["variant"],
                    layer=task["layer"],
                    cohort=task["cohort"],
                )
        for group in probability_groups(root, plan):
            if group["kind"] != "micro_probe" or group["unresolved"] or not 0.7 <= max(group["estimate"]) <= 0.99:
                continue
            task = tasks[group["representative_task"]]
            if len(d.stages.get("windows", [])) < 2 and task["cohort"] == "population":
                parent = read(root, tasks[task["source_task"]])
                center = task["source_step"]
                before = max(s["step"] for s in parent["states"] if s["step"] <= max(0, center - 4))
                points = list(range(max(before, center - 4), center + 5))
                fine = d.add(
                    "windows",
                    "micro_atlas",
                    Parameters(**task["parameters"]),
                    parent=task["source_task"],
                    at=before,
                    steps=points[-1] - before,
                    observe=[t - before for t in points],
                    initial=task["initial"],
                    mother=task["mother"],
                    cohort="mechanism",
                )
                for at in points:
                    chunked(
                        d,
                        "window_probabilities",
                        "micro_probe",
                        Parameters(**task["parameters"]),
                        parent=fine,
                        at=at,
                        initial=task["initial"],
                        mother=task["mother"],
                        paths=128,
                        steps=24000,
                        cohort="mechanism",
                        seed=20261029,
                    )
                    chunked(
                        d,
                        "window_closure",
                        "meso_probe",
                        Parameters(**task["parameters"]),
                        parent=fine,
                        at=at,
                        initial=task["initial"],
                        mother=task["mother"],
                        paths=64,
                        steps=24000,
                        cohort="mechanism",
                        seed=20261029,
                    )
            chunked(
                d,
                "validation",
                "micro_probe",
                Parameters(**task["parameters"]),
                parent=task["source_task"],
                at=task["source_step"],
                initial=task["initial"],
                mother=task["mother"],
                paths=512,
                start=100000,
                steps=24000,
                cohort="mechanism",
                seed=20261028,
            )
            # At most six focused cases, kept separate from the population design.
            if len(d.stages.get("validation", [])) >= 6 * 64:
                break
        if not d.tasks:
            return None
        inputs = {
            str(p.relative_to(root)): sha(p)
            for p in (root / "raw").glob("*.msgpack.lz4")
            if not p.name.startswith("._")
        }
        followup = freeze_plan(
            path, {**d.freeze(), "inputs": inputs, "parent_plan_fingerprint": plan["plan_fingerprint"]}
        )
    return followup

"""Regression checks for reading frozen data and the relocated runtime boundary."""

import ast
import subprocess
import sys
from pathlib import Path

from theory.runtime.kinetic import KineticParameters, KineticResolution, kinetic_request


def test_source_packages_follow_repository_dependency_direction():
    root = Path(__file__).resolve().parents[2]
    for package in ("common", "ehk", "theory"):
        forbidden = {"experiments"} | ({"ehk", "theory"} if package == "common" else set())
        for path in (root / "src" / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                    if isinstance(node, ast.ImportFrom) and not node.level
                    else []
                )
                assert not any(name and name.split(".")[0] in forbidden for name in names), path


def test_imports_need_no_runtime_or_data_and_create_no_outputs(tmp_path):
    output = tmp_path / "absent"
    code = f"""
import os, builtins, importlib, pathlib
os.environ["EHK_PAPER_THEORY_OUTPUT_DIR"] = {str(output)!r}
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.startswith(("smp_bindings", "smp_meso_bindings")):
        raise AssertionError("runtime imported while only loading analysis code")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
for path in pathlib.Path("experiments/paper_theory").rglob("*.py"):
    if path.name != "__main__.py":
        importlib.import_module(".".join(path.with_suffix("").parts))
assert not pathlib.Path({str(output)!r}).exists()
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[2])


def test_rewiring_only_probe_preserves_zero_influence_and_uniform_law():
    p = KineticParameters(influence=0.0, rewiring=0.5, steps=1)
    request = kinetic_request(
        "split",
        p,
        KineticResolution(),
        record_steps=(0, 1),
        snapshot_fields=("rho", "edge"),
        observable_fields=("node_energy", "edge_energy"),
    )
    assert request["dynamics"]["influence"] == 0.0
    assert request["dynamics"]["rewiring_rate"] == 0.5
    assert request["initial"] == {"type": "uniform", "opinion_min": -1.0, "opinion_max": 1.0, "probabilities": []}
    assert request["snapshots"]["record_steps"] == [0, 1]
    assert request["snapshots"]["rho"] and request["snapshots"]["edge"]
    assert request["stopping"]["mode"] == "fixed_steps"

"""Regression checks for reading frozen data and the relocated runtime boundary."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.paper_theory.energy_scan.expanded import config as expanded_config
from experiments.paper_theory.energy_scan.expanded import design as expanded_design
from experiments.paper_theory.energy_scan.expanded import run as expanded_run
from experiments.paper_theory.energy_scan.expanded import storage as expanded_storage
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


def test_analysis_reads_historical_hashes_but_rejects_changed_numerics(tmp_path, monkeypatch):
    monkeypatch.setattr(expanded_config, "INPUT", tmp_path)
    monkeypatch.setattr(expanded_config, "KINETIC", tmp_path / "unavailable-runtime")
    monkeypatch.setattr(expanded_config, "ADAPTER", tmp_path / "unavailable-adapter")
    raw = tmp_path / "raw_shards"
    raw.mkdir()
    for dynamics in expanded_config.DYNAMICS:
        for alpha_value in expanded_config.ALPHAS:
            alpha = float(alpha_value)
            ratios = expanded_design.feasible_ratios(alpha)
            protocol = expanded_design.simulation_protocol(dynamics, alpha, ratios, fingerprint=False)
            protocol.update(solver_binary_sha256="original-solver", python_adapter_sha256="original-adapter")
            np.savez_compressed(
                raw / expanded_storage.shard_path(dynamics, alpha).name,
                protocol=json.dumps(protocol, sort_keys=True),
                epsilon=expanded_config.EPSILONS,
                q_over_alpha=ratios,
            )
    options = dict(workers=1, chunk_size=1, analysis_only=True, rerun=False, max_new_shards=0)
    shards, complete = expanded_run.load_or_run_shards(**options)
    assert complete and len(shards) == 10
    assert json.loads(str(next(iter(shards.values()))["protocol"]))["solver_binary_sha256"] == "original-solver"
    path = next(raw.glob("*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    protocol = json.loads(str(payload["protocol"]))
    protocol["noise_diffusion"] = 0.1
    payload["protocol"] = json.dumps(protocol, sort_keys=True)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="noise_diffusion"):
        expanded_run.load_or_run_shards(**options)


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


def test_source_manifest_has_configurable_data_roots_and_no_cas_dependency(tmp_path):
    from experiments.paper_theory.data import MANIFEST, sources

    manifest = json.loads(MANIFEST.read_text())
    locations = sources(tmp_path / "scan", tmp_path / "clocks")
    assert locations["release_raw"] == tmp_path / "scan/raw_shards"
    assert locations["mean_release_reference"] == tmp_path / "clocks/parameter_clocks/data/parameter_metrics.csv"
    assert manifest["datasets"]["mean_release_reference"]["optional"]
    assert "cas2026" not in locations

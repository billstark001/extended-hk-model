"""The illustrative task has its own identity and uses authentic directed state."""

import json

import pytest

from common.io.json import sha
from common.runtime.failures import ProtocolError
from experiments.paper_theory.evidence.bundle import write_rows
from experiments.paper_theory.evidence.design import design, topics
from experiments.paper_theory.evidence.network_snapshot import default_directory, read_export


def test_snapshot_design_is_independent_and_local_by_default(tmp_path):
    plan = design("network_snapshot")
    assert topics("network_snapshot") == ("network_snapshot",)
    assert len(plan["tasks"]) == 1
    task = plan["tasks"][0]
    assert task["roles"] == ["network_snapshot"]
    assert task["parameters"]["horizon"] == 120
    assert task["request_options"]["snapshot_taus"] == [0, 24, 120]
    assert default_directory(tmp_path) == tmp_path / "outputs/paper_theory/network_snapshot"
    assert task["id"] not in {t["id"] for t in design("main")["tasks"]}
    assert design("network_snapshot", 0.04)["analysis_protocol"]["main_snapshot_tau"] == 0.04
    with pytest.raises(ValueError, match="multiple"):
        design("network_snapshot", 0.03)


def test_network_render_input_is_portable_and_authenticates_both_csvs(tmp_path):
    write_rows(tmp_path / "network_nodes.csv", ["node", "opinion"], [(0, -0.2), (1, 0.4)])
    write_rows(tmp_path / "network_edges.csv.gz", ["source", "target"], [(1, 0)])
    manifest = {"files": {name: sha(tmp_path / name) for name in ("network_nodes.csv", "network_edges.csv.gz")}}
    (tmp_path / "network_manifest.json").write_text(json.dumps(manifest))
    nodes, edges, _ = read_export(tmp_path)
    assert nodes == [(0, -0.2), (1, 0.4)]
    assert edges == [(1, 0)]  # Direction and node identities survive the CSV boundary.
    (tmp_path / "network_nodes.csv").write_text("node,opinion\n0,1\n1,1\n")
    with pytest.raises(ProtocolError, match="changed"):
        read_export(tmp_path)

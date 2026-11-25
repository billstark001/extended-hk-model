"""Recovery checks stay isolated from every production statistics database."""

import signal
from copy import deepcopy

import numpy as np
import pytest

from experiments.storage_benchmark import recovery
from experiments.storage_benchmark.recovery import (
    RecoveryFailure,
    _apply,
    document_hash,
    make_document,
    operations,
    run_recovery,
    run_recovery_round,
    validate_document,
    verify_documents,
)


def _prefix(plan, count):
    result = {}
    for operation in plan[:count]:
        _apply(result, operation)
    return result


def test_hash_covers_scalar_state_dtype_shape_and_array_bytes():
    original = make_document("fixture", 0, 512)
    assert document_hash(deepcopy(original)) == document_hash(original)
    for change in ("scalar", "dtype", "shape", "bytes"):
        altered = deepcopy(original)
        if change == "scalar":
            altered["state_epoch"] = 9
        elif change == "dtype":
            altered["payload"]["x_indices"] = altered["payload"]["x_indices"].astype(np.int64)
        elif change == "shape":
            altered["payload"]["x_indices"] = altered["payload"]["x_indices"].reshape(1, -1)
        else:
            altered["payload"]["x_indices"][0] += 1
        assert document_hash(altered) != document_hash(original)


def test_recovery_accepts_ack_prefix_or_one_whole_unacknowledged_replacement():
    plan = operations(512)
    confirmed = _prefix(plan, 3)
    before = verify_documents(list(confirmed.values()), plan, [0, 1, 2], 3)
    assert not before["unacknowledged_operation_visible"]
    _apply(confirmed, plan[3])
    after = verify_documents(list(confirmed.values()), plan, [0, 1, 2], 3)
    assert after["unacknowledged_operation_visible"]


def test_nested_array_layout_is_covered_by_its_manifest():
    document = make_document("fixture", 1, 512)
    validate_document(document)
    document["payload"]["diagnostics"]["thin_trace"][0] += 1
    with pytest.raises(RecoveryFailure, match="array/metadata"):
        validate_document(document)


def test_partial_metadata_payload_and_partial_status_updates_fail():
    plan = operations(512)
    before = _prefix(plan, 3)
    partial = deepcopy(before)
    partial["mutable"]["payload"] = plan[3]["document"]["payload"]
    with pytest.raises(RecoveryFailure, match="array/metadata"):
        verify_documents(list(partial.values()), plan, [0, 1, 2], 3)
    complete = _prefix(plan, 4)
    partial = deepcopy(complete)
    partial["mutable"]["complete"] = False  # stale and state_epoch were not updated.
    with pytest.raises(RecoveryFailure, match="complete/stale"):
        verify_documents(list(partial.values()), plan, [0, 1, 2, 3], 4)
    _apply(complete, plan[4])
    assert verify_documents(list(complete.values()), plan, [0, 1, 2, 3], 4)["unacknowledged_operation_visible"]


def test_missing_acked_document_and_incomplete_ack_sequence_fail():
    plan = operations(512)
    confirmed = _prefix(plan, 3)
    confirmed.pop("anchor-a")
    with pytest.raises(RecoveryFailure, match="neither the ACKed prefix"):
        verify_documents(list(confirmed.values()), plan, [0, 1, 2], 3)
    with pytest.raises(RecoveryFailure, match="ordered operation prefix"):
        verify_documents([], plan, [0, 2], None)
    with pytest.raises(RecoveryFailure, match="not next"):
        verify_documents([], plan, [0], 3)


def test_unknown_full_document_version_is_not_treated_as_an_unconfirmed_write():
    plan = operations(512)
    confirmed = _prefix(plan, 3)
    confirmed["mutable"] = make_document("mutable", 7, 512)
    validate_document(confirmed["mutable"])
    with pytest.raises(RecoveryFailure, match="neither the ACKed prefix"):
        verify_documents(list(confirmed.values()), plan, [0, 1, 2], 3)


def test_repeated_protocol_requires_at_least_five_rounds(tmp_path):
    with pytest.raises(ValueError, match="at least five"):
        run_recovery("current", tmp_path / "recovery", rounds=4)


def test_recovery_accepts_optimized_backend_and_keeps_six_distinct_rounds(tmp_path, monkeypatch):
    calls = []

    def fault_round(backend, directory, **kwargs):
        calls.append((backend, directory, kwargs))
        return {"passed": True}

    monkeypatch.setattr(recovery, "run_recovery_round", fault_round)
    report = run_recovery("sqlite_optimized", tmp_path / "optimized", payload_bytes=512)
    assert report["all_passed"] and report["planned_rounds"] == 6
    assert [call[0] for call in calls] == ["sqlite_optimized"] * 6
    assert [(kwargs["phase"], kwargs["target"]) for _, _, kwargs in calls] == list(recovery.SCHEDULE)
    assert len({directory for _, directory, _ in calls}) == 6


def test_recovery_cli_accepts_optimized_and_rejects_unknown_or_missing_backend(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        recovery,
        "run_recovery",
        lambda backend, *_args, **kwargs: calls.append((backend, kwargs)) or {"all_passed": True},
    )
    assert recovery.main(["--backend", "sqlite_optimized", "--directory", str(tmp_path)]) == 0
    assert calls == [("sqlite_optimized", {"rounds": 6, "payload_bytes": 262144, "timeout": 30})]
    for arguments in ([], ["--backend"], ["--backend", "missing"]):
        with pytest.raises(SystemExit) as error:
            recovery.main([*arguments, "--directory", str(tmp_path)])
        assert error.value.code == 2


def test_unknown_recovery_backend_fails_before_creating_evidence(tmp_path):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="unknown backend"):
        run_recovery("missing", directory)
    assert not directory.exists()


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX writer-process SIGKILL protocol")
@pytest.mark.parametrize(
    "phase,target", [("after_ack", 2), ("before_commit", 3), ("before_commit", 4), ("before_commit", 5)]
)
def test_subprocess_reopen_preserves_acknowledged_data(tmp_path, phase, target):
    report = run_recovery_round("current", tmp_path / "round", phase=phase, target=target, payload_bytes=512)
    assert report["passed"]
    assert report["integrity_check"] == ["ok"]
    assert report["writer_exitcode"] == -signal.SIGKILL
    assert report["coordinator_pid"] != report["writer_pid"]
    assert report["acknowledged_operations"] == target + (phase == "after_ack")
    assert not report["power_loss_test"]

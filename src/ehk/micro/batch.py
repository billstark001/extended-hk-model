"""Adapter for the SMP JSONL checkpoint and observation protocol."""

import gzip
import json
from pathlib import Path

from common.runtime.process import run_jsonl


def observe_simulation(
    binary: Path,
    metadata: dict,
    steps: list[int],
    *,
    checkpoint: Path | None = None,
    save_checkpoints: dict[int, Path] | None = None,
    policy_change: dict | None = None,
    terminal: bool = True,
    exposure_trace: Path | None = None,
) -> dict:
    from smp_bindings.codec import decode_batch_response, encode_batch_request

    request_id = metadata["UniqueName"]
    experiment = {"mode": "resume" if checkpoint else "fresh", "replicates": 1, "observe_at": steps}
    if checkpoint is not None:
        experiment["checkpoint"] = str(checkpoint)
    if save_checkpoints:
        experiment["checkpoint_at"] = {str(step): str(path) for step, path in save_checkpoints.items()}
    if policy_change is not None:
        experiment["policy_change"] = policy_change
    output: dict = {"pathway": True}
    if exposure_trace is not None:
        output["exposure_trace"] = str(exposure_trace)
    if terminal:
        output["terminal"] = {"major_mass": 0.02}
        output["final_opinions"] = True
    request = {"request_id": request_id, "metadata": metadata, "output": output, "experiment": experiment}
    stdout = run_jsonl(binary, encode_batch_request(request) + "\n")
    reply = decode_batch_response(stdout, expected_request_id=request_id)
    if reply["status"] != "ok":
        raise RuntimeError(f"{request_id}: {reply['error']['message']}")
    result = reply["result"]
    (replicate,) = result["replicates"]
    endpoints = replicate["endpoints"]
    if replicate["stop_reason"] != "horizon" or [point["relative_step"] for point in endpoints] != steps:
        raise ValueError(f"incomplete observation window for {request_id}")
    previous = 0
    for endpoint in endpoints:
        relative = endpoint["relative_step"]
        if endpoint["exposure"]["updates"] != (relative - previous) * metadata["NodeCount"]:
            raise ValueError(f"invalid exposure denominator for {request_id}")
        previous = relative
    if exposure_trace is not None:
        receipt = replicate["exposure_trace"]
        if (
            Path(receipt["path"]) != exposure_trace
            or not exposure_trace.is_file()
            or receipt["updates"] != sum(point["exposure"]["updates"] for point in endpoints)
            or receipt["slots"] != sum(point["exposure"]["recommended"] for point in endpoints)
        ):
            raise ValueError(f"invalid exposure trace receipt for {request_id}")
    if checkpoint is not None:
        replicate["source_sha256"] = result["source_sha256"]
    return replicate


def iter_exposure_trace(dataset: Path, receipt: dict):
    """Stream recorded feed rows using the path relative to the batch dataset."""
    if receipt["version"] != 1:
        raise ValueError("unsupported exposure trace version")
    with gzip.open(dataset / receipt["path"], "rt", encoding="utf8") as source:
        for line in source:
            yield json.loads(line)

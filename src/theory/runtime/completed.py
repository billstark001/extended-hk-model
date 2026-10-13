"""Distinguish completed, parameter-matching records from interrupted writes."""

import json
import math
from collections.abc import Mapping
from typing import Any, Literal, cast
from zipfile import BadZipFile

import numpy as np

from common.io.records import read_record
from common.runtime.failures import ProtocolError
from common.runtime.storage import storage_error
from common.types import PathInput
from theory.types import ConservationDiagnostics, MicroRecord, PairTask, Task, TaskKind

from .records import read_micro


def conservation(diagnostics: ConservationDiagnostics) -> None:
    values = [diagnostics[name] for name in ("max_node_mass_residual", "max_fixed_degree_residual")]
    if not all(np.isfinite(v) and v <= 1e-8 for v in values):
        raise ValueError("invalid conservation diagnostics")


def cached(path: PathInput, task: Task | PairTask, kind: TaskKind | Literal["pair"]) -> bool:
    """Invalid/truncated records are rerun; a different task is never overwritten."""
    try:
        if kind == "measure":
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                # Force ZIP decompression/CRC checks, including final-state fields.
                arrays = {name: archive[name] for name in archive.files}
            diagnostics = json.loads(str(arrays["diagnostics"]))
            conservation(diagnostics)
            time = arrays["time"]
            for name in ("node_energy", "edge_energy", "tau"):
                if arrays[name].shape != time.shape or not np.all(np.isfinite(arrays[name])):
                    raise ValueError(f"invalid {name} series")
            for name in ("final_rho", "final_edge"):
                if not arrays[name].size or not np.all(np.isfinite(arrays[name])):
                    raise ValueError(f"invalid {name}")
            if not len(time) or not np.all(np.isfinite(time)):
                raise ValueError("missing time series")
            if not np.isclose(time[-1], diagnostics["executed_steps"] * task["parameters"]["dt"]):
                raise ValueError("incomplete recorded endpoint")
        else:
            record = read_record(path)
            metadata = record["task"]
            if metadata != task:
                raise ProtocolError(f"cache protocol differs: {path}")
            result: Mapping[str, Any] = record["result"]
            if kind != "pair":
                result = read_micro(path, record=cast(MicroRecord, record))["result"]
            conservation(result["pair"]["diagnostics"])
            p = task["parameters"]
            endpoint = math.ceil(p["horizon"] / (p["alpha"] * p["dt"]) - 1e-12) * p["dt"]
            traces = (result["pair"]["frames"],) if kind == "pair" else (result["frames"], result["pair"]["frames"])
            for frames in traces:
                if not frames or not np.isclose(frames[-1]["time"], endpoint):
                    raise ValueError("incomplete recorded endpoint")
                if any(
                    not np.isfinite(frame.get("exact_" + name, frame.get(name)))
                    for frame in frames
                    for name in ("node_energy", "edge_energy")
                ):
                    raise ValueError("invalid energy series")
    except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile, RuntimeError) as error:
        if storage_error(error):
            raise  # A disconnected volume must not trigger recomputation.
        print(f"incomplete cache; rerun {task['id']}: {error}", flush=True)
        # Even incomplete caches with readable metadata cannot mix protocols.
        if "metadata" in locals() and metadata != task:
            raise ProtocolError(f"cache protocol differs: {path}") from error
        return False
    if metadata != task:
        raise ProtocolError(f"cache protocol differs: {path}")
    return True

"""Resumable case/shard execution and protocol-checked offline scan loading."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def case_from_responses(alpha: float, response: dict[str, Any], probe: dict[str, Any]) -> dict[str, np.ndarray]:
    """Combine a trajectory and its first-step probe into padded-array-ready case data with runtime diagnostics."""
    result = response["result"]
    series = result["series"]
    diagnostics = result["diagnostics"]
    snapshots = result["snapshots"]
    probe_series = probe["result"]["series"]
    return {
        "tau": alpha * np.asarray(series["time"], dtype=float),
        "node_energy": np.asarray(series["node_energy"], dtype=float),
        "edge_energy": np.asarray(series["edge_energy"], dtype=float),
        "executed_steps": np.asarray(int(diagnostics["executed_steps"])),
        "converged": np.asarray(bool(diagnostics["converged"])),
        "final_state_l1_rate": np.asarray(float(diagnostics["final_state_l1_rate"])),
        "final_node_energy_rate": np.asarray(float(diagnostics["final_node_energy_rate"])),
        "final_edge_energy_rate": np.asarray(float(diagnostics["final_edge_energy_rate"])),
        "maximum_node_mass_residual": np.asarray(float(diagnostics["max_node_mass_residual"])),
        "maximum_fixed_degree_residual": np.asarray(float(diagnostics["max_fixed_degree_residual"])),
        "final_node_potential": np.asarray(snapshots["final_node_potential"]),
        "final_edge_potential": np.asarray(snapshots["final_edge_potential"]),
        "first_step_delta": np.asarray(
            (
                float(probe_series["node_energy"][1] - probe_series["node_energy"][0]),
                float(probe_series["edge_energy"][1] - probe_series["edge_energy"][0]),
            )
        ),
    }


def progress_callback(alpha: float, case_numbers: dict[str, int], case_total: int) -> Any:
    """Report public runtime batch progress for the current shard."""

    def report(event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        if kind not in {"request_started", "step_heartbeat", "request_completed"}:
            return
        request_id = str(event.get("request_id", "<unknown>"))
        number = case_numbers.get(request_id, 0)
        process = f"p{event.get('process_index', '?')}/{event.get('process_total', '?')}"
        if kind == "request_started":
            detail = "started"
        elif kind == "request_completed":
            detail = f"completed step={event.get('step', '?')} tau={alpha * float(event.get('step', 0)):g}"
        else:
            step = int(event.get("step", 0))
            total_steps = int(event.get("total_steps", 0))
            detail = f"step={step}/{total_steps} tau={alpha * step:.3g}/{alpha * total_steps:.3g}"
        print(
            f"[case {number}/{case_total} {process}] {request_id}: {detail} ({float(event.get('elapsed_seconds', 0.0)):.1f}s)",
            file=sys.stderr,
            flush=True,
        )

    return report


def assemble_cases(ratios: np.ndarray, cases: dict[tuple[int, int], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pack case trajectories and first-step diagnostics into one alpha/dynamics shard, retaining variable lengths."""
    shape = (config.EPSILONS.size, ratios.size)
    maximum = max((data["tau"].size for data in cases.values()))
    output: dict[str, np.ndarray] = {
        "lengths": np.empty(shape, dtype=np.int64),
        "tau": np.full(shape + (maximum,), np.nan),
        "node_energy": np.full(shape + (maximum,), np.nan),
        "edge_energy": np.full(shape + (maximum,), np.nan),
        "executed_steps": np.empty(shape, dtype=np.int64),
        "converged": np.empty(shape, dtype=bool),
        "final_state_l1_rate": np.empty(shape),
        "final_node_energy_rate": np.empty(shape),
        "final_edge_energy_rate": np.empty(shape),
        "maximum_node_mass_residual": np.empty(shape),
        "maximum_fixed_degree_residual": np.empty(shape),
        "final_node_potential": np.empty(shape + (config.GRID_SIZE,)),
        "final_edge_potential": np.empty(shape + (config.GRID_SIZE,)),
        "first_step_delta": np.empty(shape + (len(config.ENERGY_NAMES),)),
    }
    scalar_names = (
        "executed_steps",
        "converged",
        "final_state_l1_rate",
        "final_node_energy_rate",
        "final_edge_energy_rate",
        "maximum_node_mass_residual",
        "maximum_fixed_degree_residual",
    )
    for index, data in cases.items():
        length = data["tau"].size
        output["lengths"][index] = length
        for name in ("tau", "node_energy", "edge_energy"):
            output[name][index + (slice(0, length),)] = data[name]
        for name in scalar_names:
            output[name][index] = data[name]
        for name in ("final_node_potential", "final_edge_potential", "first_step_delta"):
            output[name][index] = data[name]
    return output


def run_shard(dynamics: str, alpha: float, workers: int, chunk_size: int, *, rerun: bool) -> dict[str, np.ndarray]:
    """Execute unresolved cases for one dynamics/alpha cell group, checkpointing cases before assembling the final shard."""
    from smp_meso_bindings import run_kinetic_batch_parallel

    from experiments.paper_theory.energy_scan.expanded.design import (
        case_request,
        feasible_ratios,
        probe_request,
        simulation_protocol,
        step_controls,
    )
    from experiments.paper_theory.energy_scan.expanded.storage import (
        case_checkpoint_path,
        load_case_checkpoint,
        save_case_checkpoint,
    )

    ratios = feasible_ratios(alpha)
    steps, record_every, stopping = step_controls(alpha)
    protocol = simulation_protocol(dynamics, alpha, ratios)
    specs = [
        (epsilon_index, ratio_index, float(epsilon), float(ratio))
        for epsilon_index, epsilon in enumerate(config.EPSILONS)
        for ratio_index, ratio in enumerate(ratios)
    ]
    cases: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    pending = []
    for case_number, spec in enumerate(specs, 1):
        epsilon_index, ratio_index, epsilon, ratio = spec
        path = case_checkpoint_path(dynamics, alpha, epsilon, ratio)
        if path.is_file() and (not rerun):
            cases[epsilon_index, ratio_index] = load_case_checkpoint(path, protocol, epsilon, ratio)
        else:
            pending.append((case_number, spec))
    print(
        f"[shard {dynamics} alpha={alpha:.8g}] resumed {len(cases)}/{len(specs)} case checkpoints",
        file=sys.stderr,
        flush=True,
    )
    progress_interval = max(1, round(5.0 / alpha))
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        requests = [
            case_request(dynamics, alpha, epsilon, ratio, steps=steps, record_every=record_every, stopping=stopping)
            for _, (_, _, epsilon, ratio) in chunk
        ]
        case_numbers = {
            str(request["request_id"]): case_number for request, (case_number, _) in zip(requests, chunk, strict=True)
        }
        responses = run_kinetic_batch_parallel(
            config.KINETIC,
            requests,
            min(workers, len(requests)),
            progress=progress_callback(alpha, case_numbers, len(specs)),
            progress_step_interval=progress_interval,
        )
        probes = [probe_request(dynamics, alpha, epsilon, ratio) for _, (_, _, epsilon, ratio) in chunk]
        probe_responses = run_kinetic_batch_parallel(config.KINETIC, probes, min(workers, len(probes)))
        for (_, spec), response, probe in zip(chunk, responses, probe_responses, strict=True):
            epsilon_index, ratio_index, epsilon, ratio = spec
            data = case_from_responses(alpha, response, probe)
            save_case_checkpoint(case_checkpoint_path(dynamics, alpha, epsilon, ratio), protocol, epsilon, ratio, data)
            cases[epsilon_index, ratio_index] = data
        print(
            f"[shard {dynamics} alpha={alpha:.8g}] checkpointed {len(cases)}/{len(specs)} cases",
            file=sys.stderr,
            flush=True,
        )
    return assemble_cases(ratios, cases)


def load_or_run_shards(
    *, workers: int, chunk_size: int, analysis_only: bool, rerun: bool, max_new_shards: int
) -> tuple[dict[tuple[str, float], dict[str, np.ndarray]], bool]:
    """Return loaded/generated shards and completeness; analysis_only reads frozen numerical protocols without running a solver."""
    from experiments.paper_theory.energy_scan.expanded.design import feasible_ratios, simulation_protocol
    from experiments.paper_theory.energy_scan.expanded.storage import load_shard, save_shard, shard_path

    shards = {}
    total = len(config.DYNAMICS) * config.ALPHAS.size
    completed = 0
    newly_run = 0
    for dynamics in config.DYNAMICS:
        for alpha_value in config.ALPHAS:
            alpha = float(alpha_value)
            ratios = feasible_ratios(alpha)
            protocol = simulation_protocol(dynamics, alpha, ratios, fingerprint=not analysis_only)
            path = (
                config.INPUT / "raw_shards" / shard_path(dynamics, alpha).name
                if analysis_only
                else shard_path(dynamics, alpha)
            )
            if path.is_file() and (not rerun):
                data = load_shard(path, None if analysis_only else protocol)
                if analysis_only:
                    recorded = json.loads(str(data["protocol"]))
                    for key, value in protocol.items():
                        if key not in ("solver_binary_sha256", "python_adapter_sha256") and recorded.get(key) != value:
                            raise ValueError(f"incompatible numerical protocol ({key}): {path}")
                action = "loaded"
            else:
                if analysis_only:
                    raise FileNotFoundError(f"missing raw shard in analysis-only mode: {path}")
                data = run_shard(dynamics, alpha, workers, chunk_size, rerun=rerun)
                save_shard(path, protocol, ratios, data)
                data = load_shard(path, protocol)
                action = "ran"
                newly_run += 1
            shards[dynamics, alpha] = data
            completed += 1
            print(
                f"[{completed}/{total}] {action} {dynamics} alpha={alpha:.8g} runs={config.EPSILONS.size * ratios.size}",
                flush=True,
            )
            if max_new_shards and newly_run >= max_new_shards:
                print(f"Stopped cleanly after {newly_run} newly completed shard(s).", flush=True)
                return (shards, completed == total)
    return (shards, True)


def print_status(input_dir=None) -> None:
    """Print completed/missing shard status against the declared triangular grid."""
    from experiments.paper_theory.energy_scan.expanded.design import feasible_ratios
    from experiments.paper_theory.energy_scan.expanded.storage import shard_path

    complete = 0
    main_runs = 0
    checkpointed_cases = 0
    total_cases = 0
    for dynamics in config.DYNAMICS:
        for alpha_value in config.ALPHAS:
            alpha = float(alpha_value)
            ratios = feasible_ratios(alpha)
            cases = config.EPSILONS.size * ratios.size
            total_cases += cases
            path = (
                input_dir / "raw_shards" / shard_path(dynamics, alpha).name
                if input_dir
                else shard_path(dynamics, alpha)
            )
            if path.is_file():
                state = "complete"
                complete += 1
                main_runs += cases
                partial = 0
            else:
                directory = (input_dir / "case_checkpoints" if input_dir else config.CHECKPOINTS) / path.stem
                partial = len(tuple(directory.glob("*.npz"))) if directory.is_dir() else 0
                checkpointed_cases += partial
                state = f"partial {partial}/{cases}" if partial else "missing"
            print(f"{dynamics:8s} alpha={alpha:.12g} cases={cases:2d} {state}", flush=True)
    print(
        f"complete shards: {complete}/{len(config.DYNAMICS) * config.ALPHAS.size}; durable main runs: {main_runs + checkpointed_cases}/{total_cases}",
        flush=True,
    )

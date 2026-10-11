"""Atomic case/shard caches, protocol validation and descriptor exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def sha256(path: Path) -> str:
    """Hash file contents in fixed-size chunks for binary/adapter provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_path(dynamics: str, alpha: float) -> Path:
    """Return a shard path keyed by dynamics and alpha; create no directories."""
    return config.RAW / f"{dynamics}_alpha_{alpha:.12g}.npz"


def case_checkpoint_path(dynamics: str, alpha: float, epsilon: float, ratio: float) -> Path:
    """Return a case path keyed by dynamics, alpha, confidence and ratio; create no directories."""
    directory = config.CHECKPOINTS / shard_path(dynamics, alpha).stem
    return directory / f"epsilon_{epsilon:g}_ratio_{ratio:.12g}.npz"


def save_case_checkpoint(
    path: Path, protocol: dict[str, Any], epsilon: float, ratio: float, data: dict[str, np.ndarray]
) -> None:
    """Atomically save one case with its complete protocol, epsilon and rate ratio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".temporary.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(json.dumps(protocol, sort_keys=True)),
        epsilon=np.asarray(epsilon),
        q_over_alpha=np.asarray(ratio),
        **data,
    )
    temporary.replace(path)


def load_case_checkpoint(path: Path, protocol: dict[str, Any], epsilon: float, ratio: float) -> dict[str, np.ndarray]:
    """Return cached case arrays only when protocol and physical keys match the requested case."""
    expected = json.dumps(protocol, sort_keys=True) if protocol is not None else None
    with np.load(path) as archive:
        if str(archive["protocol"].item()) != expected:
            raise ValueError(f"incompatible case checkpoint: {path}")
        if not np.isclose(float(archive["epsilon"]), epsilon):
            raise ValueError(f"epsilon mismatch in case checkpoint: {path}")
        if not np.isclose(float(archive["q_over_alpha"]), ratio):
            raise ValueError(f"ratio mismatch in case checkpoint: {path}")
        ignored = {"protocol", "epsilon", "q_over_alpha"}
        return {name: np.asarray(archive[name]) for name in archive.files if name not in ignored}


def save_shard(path: Path, protocol: dict[str, Any], ratios: np.ndarray, data: dict[str, np.ndarray]) -> None:
    """Atomically save a dynamics/alpha shard with its numerical protocol and coordinate arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".temporary.npz")
    np.savez_compressed(
        temporary,
        protocol=np.asarray(json.dumps(protocol, sort_keys=True)),
        epsilon=config.EPSILONS,
        q_over_alpha=ratios,
        **data,
    )
    temporary.replace(path)


def load_shard(path: Path, protocol: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Return stored shard arrays, optionally rejecting any mismatch with the supplied complete protocol."""
    expected = json.dumps(protocol, sort_keys=True) if protocol is not None else None
    with np.load(path) as archive:
        actual = str(archive["protocol"].item())
        if expected is not None and actual != expected:
            raise ValueError(f"incompatible raw shard: {path}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Export rows to path using their ordered union of fields; the parent directory must exist."""
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_descriptor_grid(rows: list[dict[str, Any]]) -> None:
    """Export curve-descriptor grids for interpolation/analysis using the configured feasible-rate coordinates."""
    names = (
        "initial_energy",
        "plateau",
        "drop_amplitude",
        "first_step_fraction",
        "tau_10",
        "tau_50",
        "tau_90",
        "tau_99",
        "tail_slowdown",
    )
    shape = (
        len(config.DYNAMICS),
        config.EPSILONS.size,
        config.ALPHAS.size,
        config.RATIOS.size,
        len(config.ENERGY_NAMES),
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    dynamics_index = {name: index for index, name in enumerate(config.DYNAMICS)}
    for row in rows:
        index = (
            dynamics_index[row["dynamics"]],
            int(np.flatnonzero(config.EPSILONS == row["epsilon"])[0]),
            int(np.flatnonzero(np.isclose(config.ALPHAS, row["alpha"]))[0]),
            int(np.flatnonzero(np.isclose(config.RATIOS, row["q_over_alpha"]))[0]),
            config.ENERGY_NAMES.index(row["energy"]),
        )
        for name in names:
            arrays[name][index] = row[name]
    np.savez_compressed(
        config.HERE / "descriptor_grid.npz",
        dynamics=np.asarray(config.DYNAMICS),
        epsilon=config.EPSILONS,
        alpha=config.ALPHAS,
        q_over_alpha=config.RATIOS,
        energy=np.asarray(config.ENERGY_NAMES),
        **arrays,
    )

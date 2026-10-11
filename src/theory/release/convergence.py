"""Independent energy plateau confirmation at every required checkpoint."""

from .quantiles import clock_pair


def energy_tail_check(frames, stopping):
    """Reconstruct the original energy criterion at every required checkpoint.

    Convergence requires BOTH energies throughout the final patience interval.
    A flat final difference alone is not a confirmation. Missing checkpoints
    are an error, rather than an implicit pass.
    """
    every = stopping["check_every"]
    by_step = {f["step"]: f for f in frames}
    last = frames[-1]["step"]
    needed = range(every, last + 1, every)
    previous = by_step[0]
    stable = 0
    for step in needed:
        if step not in by_step:
            raise ValueError(f"missing energy checkpoint at {step}")
        current = by_step[step]
        elapsed = step - previous["step"]
        ok = all(
            abs(current[k] - previous[k]) / elapsed
            <= stopping["energy_absolute_tolerance"]
            + stopping["energy_relative_tolerance"] * max(abs(current[k]), abs(previous[k]))
            for k in ("node_energy", "edge_energy")
        )
        stable = stable + elapsed if ok and step >= stopping["minimum_steps"] else 0
        previous = current
    # Fixed prefixes may end between checkpoints; confirmation belongs to the
    # last actual check, not an unobserved interval.
    return {
        "energy_confirmed": stable >= stopping["patience_steps"] and previous["step"] >= stopping["minimum_steps"],
        "energy_stable_steps": stable,
        "last_energy_check_step": previous["step"],
    }


def confirmed_clocks(trace, stopping, zero_tolerance):
    """Return independently energy-confirmed paired clocks, snapping near-zero log ratios using zero_tolerance."""
    check = energy_tail_check(trace["frames"], stopping)
    energy_trace = {**trace, "diagnostics": {"converged": check["energy_confirmed"]}}
    result = {**check, **clock_pair(energy_trace)}
    signs = []
    for p in (10, 50, 90):
        value = result[f"delta{p}"]
        if value is not None:
            if abs(value) <= zero_tolerance:
                result[f"delta{p}"] = 0.0
                signs.append(0)
            else:
                signs.append(1 if value > 0 else -1)
    result["order_consistent"] = len(set(signs)) == 1 if len(signs) == 3 else None
    return result

"""Reference-versus-instrumented public mesoscopic runtime parity checks."""

from __future__ import annotations

import numpy as np

from common.io.codec import decoded


def parity(data):
    """Return reference-versus-hooked runtime output gaps for a parity record, including encoded final states."""
    r, h = data["reference"], data["hooked"]
    last = h["frames"][-1]
    p = decoded(r["snapshots"]["final_rho"])
    e = decoded(r["snapshots"]["final_edge"])
    gap = max(
        np.max(abs(p - np.asarray(last["rho"]))),
        np.max(abs(e.ravel() - np.asarray(last["edge"]))),
        abs(r["summary"]["pathway"] - h["pathway"]),
    )
    if gap > 1e-12:
        raise ValueError(f"production/inspection parity failed: {gap}")
    return float(gap)

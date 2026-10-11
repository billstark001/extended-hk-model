"""Set subprocess resources explicitly; importing this module has no effects."""

import os
from pathlib import Path


def runtime_environment(output=None):
    """Return a copied environment with single-thread batch limits; when output is supplied, create absolute cache/temp directories there."""
    env = {
        **os.environ,
        "GOMAXPROCS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "GOTOOLCHAIN": "local",
        "PYTHONUNBUFFERED": "1",
    }
    if output is not None:
        root = Path(output).expanduser().resolve()
        for key, suffix in {
            "TMPDIR": "tmp",
            "GOTMPDIR": "tmp",
            "GOCACHE": "cache/go-build",
            "MPLCONFIGDIR": "cache/matplotlib",
            "PYTHONPYCACHEPREFIX": "cache/python",
        }.items():
            path = root / suffix
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    return env

"""Read-only Git provenance services."""

import hashlib
import subprocess
from pathlib import Path


def git_fingerprint(root: Path) -> dict[str, str]:
    """Return Git HEAD and tracked working-tree diff hash for the supplied checkout; untracked files are excluded."""
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--no-ext-diff", "HEAD"])
    return {"revision": revision, "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest()}


def revision(path):
    """Return the supplied checkout HEAD."""
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

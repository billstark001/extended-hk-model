"""Output routing only; experiment parameters remain in study design modules."""

from dataclasses import dataclass
from pathlib import Path

from theory.runtime.protocol import FAMILIES

from ..paths import output_root


@dataclass(frozen=True)
class OutputPaths:
    """Route build snapshots, control files and per-family results under one run root.

    An explicit root is resolved absolutely; the default is configured output/validation.
    Path accessors only construct paths and do not create directories.
    """

    root: Path | None = None

    def __post_init__(self):
        value = output_root() / "validation" if self.root is None else self.root
        object.__setattr__(self, "root", Path(value).expanduser().resolve())

    def control(self, label):
        """Return the plan/binding/status directory for a run label."""
        base = self.root
        return base / label

    def family(self, family, label):
        """Return a declared family directory under the label, rejecting unknown families."""
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
        return self.control(label) / family

    @property
    def build(self):
        """Return the default immutable build-snapshot directory under this output root."""
        base = self.root
        return base / "build"

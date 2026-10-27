"""Study-owned engine selection and calibration provenance; builds stay in common."""

from pathlib import Path

from common.experiments.plans import freeze_manifest
from common.io.json import sha
from common.runtime.failures import ProtocolError
from common.runtime.go import build_adapter, engine_manifest, restore_adapter, source_manifest

from .evidence.protocol import MODEL_ROOT, sources


def build(root, *, backend, micro_root, meso_root, calibration=None, binding_sources=None):
    """Reuse theory's initialization instrumentation; build only a thin lifted continuation adapter."""
    selected = {
        # TODO(runtime-initialization): extract the shared fixed-degree/uniform
        # initializer with theory.runtime.population into a model-runtime owner
        # when both studies migrate their schemas. Keep the existing Go source,
        # seed policy and authenticated binaries unchanged in this refactor.
        "initial": MODEL_ROOT / "experiments/paper_theory/runtime/engine",
        "lifted": MODEL_ROOT / "src/lifted/runtime/engine",
    }
    micro, meso = source_manifest(micro_root), source_manifest(meso_root)
    if calibration:
        pilot = Path(calibration["pilot_dir"])
        if any(
            sha(pilot / name) != calibration[key]
            for name, key in (("plan.json", "pilot_plan_sha256"), ("binding.json", "pilot_binding_sha256"))
        ):
            raise ProtocolError("calibrated pilot plan/binding changed")
    binaries = {}
    for name, engine in selected.items():
        if calibration:
            manifest = restore_adapter(
                root / "build" / name,
                Path(calibration["pilot_dir"]) / "build" / name,
                binary_name="paper-" + name + "-engine",
                expected_inputs=dict(backend=backend, micro=micro, meso=meso, engine=engine_manifest(engine)),
                expected_sha256=calibration["runtime_binding"]["binaries"][name],
            )
            binaries[name] = manifest["binary"]
        else:
            manifest = build_adapter(
                root / "build" / name, meso_root, micro_root, backend, engine, binary_name="paper-" + name + "-engine"
            )
            binaries[name] = manifest["binary"]
    binding = {
        "binaries": {k: sha(v) for k, v in binaries.items()},
        "sources": sources() if binding_sources is None else binding_sources,
        "micro": micro,
        "meso": meso,
        "backend": backend,
    }
    freeze_manifest(root / "binding.json", binding)
    return binaries, binding

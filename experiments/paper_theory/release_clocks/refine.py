"""Independent solver check at crossover and delayed-cascade midpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from theory.runtime.kinetic import run_kinetic_batch_parallel

from ..energy_scan.expanded import config as scan_config
from ..energy_scan.expanded import design as scan_design
from ..energy_scan.expanded import run as scan_run
from ..energy_scan.expanded import storage as scan_storage
from .artifact_paths import RAW, ensure_output_dirs


def main():
    """Generate independent runtime refinements around crossing/delayed-cascade rates with immutable cache provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_dirs()
    OUT = RAW
    a = 0.01
    specs = [
        (d, e, r) for d in scan_config.DYNAMICS for e in (0.2, 0.4) for r in (1.2, 1.4, 1.5625, 1.6, 1.8, 2.0, 2.2, 2.4)
    ]
    specs += [("deffuant", 0.2, r) for r in (18.0, 20.0, 22.0, 24.0, 26.0)]
    manifest_path = RAW / "refinement_manifest.json"
    solver_digest = scan_storage.sha256(scan_config.KINETIC)
    generator_digest = scan_storage.sha256(Path(scan_run.__file__))
    adapter_digest = scan_storage.sha256(scan_config.ADAPTER)
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if (
            recorded["solver_sha256"] != solver_digest
            or recorded["source_generator_sha256"] != generator_digest
            or recorded.get("adapter_sha256") != adapter_digest
        ):
            raise RuntimeError(
                "Cached refinements belong to a different solver or generator; preserve the recorded raw data and run new experiments in a separate directory."
            )
    steps, record, stopping = scan_design.step_controls(a)
    for k in range(0, len(specs), 3):
        chunk = [s for s in specs[k : k + 3] if not (OUT / f"{s[0]}_e{s[1]}_r{s[2]}.npz").exists()]
        if not chunk:
            continue
        reqs = [
            scan_design.case_request(d, a, e, r, steps=steps, record_every=record, stopping=stopping)
            for d, e, r in chunk
        ]
        responses = run_kinetic_batch_parallel(scan_config.KINETIC, reqs, len(reqs))
        for (d, e, r), response in zip(chunk, responses):
            # At alpha=.01 every numerical step is recorded, so no separate probe needed.
            data = scan_run.case_from_responses(a, response, response)
            np.savez_compressed(OUT / f"{d}_e{e}_r{r}.npz", dynamics=d, epsilon=e, r=r, alpha=a, **data)
            print(f"{d} e={e} r={r} converged={data['converged']}", flush=True)
    manifest = {
        "solver_sha256": scan_storage.sha256(scan_config.KINETIC),
        "source_generator_sha256": scan_storage.sha256(Path(scan_run.__file__)),
        "adapter_sha256": adapter_digest,
        "specs": specs,
        "alpha": a,
        "purpose": "independent new-rate interpolation validation; four existing-rate anchors check solver compatibility",
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

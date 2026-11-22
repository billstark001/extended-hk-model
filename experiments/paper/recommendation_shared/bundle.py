"""Authenticated portable summary bundles for independent recommendation studies."""

import argparse
import gzip
import json
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

from common.experiments.plans import read_plan
from common.io.json import atomic_json, sha

from .analysis import read_study

SCHEMA = 1


def analyze_study(directory, output, *, make_figures=True):
    """Dispatch by the frozen profile without importing another study's figure code."""
    profile = read_plan(Path(directory) / "plan.json")["profile"]
    packages = {
        "staged": "recommendation",
        "staged_timing": "recommendation",
        "recommender_variants": "recommender_variants",
    }
    try:
        package = packages[profile]
    except KeyError:
        raise ValueError(f"unsupported recommendation profile: {profile}") from None
    return import_module(f"experiments.paper.{package}.plot").analyze(
        directory, output, make_figures=make_figures,
    )


def _compress_csv(path):
    with path.open("rb") as source, path.with_suffix(".csv.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
            shutil.copyfileobj(source, stream)


def _recover_data(output):
    current, backup = output / "data", output / ".previous-data"
    if not backup.exists():
        return
    if current.exists():
        manifest = json.loads((current / "manifest.json").read_text())
        if manifest["schema"] != SCHEMA or any(sha(current / name) != expected for name, expected in manifest["files"].items()):
            raise ValueError("interrupted portable-data replacement requires inspection")
        shutil.rmtree(backup)
    else:
        backup.rename(current)


def export_bundle(dataset, output, *, require_complete=False):
    """Copy authenticated completed summaries, then calculate local figure tables."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("the portable bundle must be separate from the source dataset")
    plan, records = read_study(dataset)
    if not records:
        raise ValueError("no completed recommendation arms")
    output.mkdir(parents=True, exist_ok=True)
    _recover_data(output)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    try:
        study = stage / "study"
        (study / "records").mkdir(parents=True)
        shutil.copy2(dataset / "plan.json", study / "plan.json")
        for identifier in records:
            shutil.copy2(dataset / "records" / f"{identifier}.json.gz", study / "records" / f"{identifier}.json.gz")
        for relative in ("snapshot.json", "runtime/binding.json"):
            source = dataset / relative
            if source.is_file():
                (study / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, study / relative)
        tables = stage / "tables"
        report = analyze_study(study, tables, make_figures=False)
        if report["source_plan"] != plan["plan_fingerprint"]:
            raise ValueError("the source plan changed during export; retry with a stable design")
        if require_complete and not report["complete"]:
            raise ValueError(f"recommendation study is incomplete: {report['completed']}/{report['planned']}")
        for path in tables.glob("*.csv"):
            _compress_csv(path)
            path.unlink()
        manifest = {
            "schema": SCHEMA, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset), "plan_fingerprint": plan["plan_fingerprint"],
            "profile": plan["profile"], "available_records": report["available_records"],
            "completed": report["completed"], "planned": report["planned"],
            "files": {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob("*")) if path.is_file()},
        }
        atomic_json(stage / "manifest.json", manifest)
        backup = output / ".previous-data"
        current = output / "data"
        if current.exists():
            current.rename(backup)
        try:
            stage.rename(current)
        except BaseException:
            if backup.exists():
                backup.rename(current)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def _bundle_manifest(bundle):
    _recover_data(bundle)
    manifest = json.loads((bundle / "data/manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported manuscript data schema")
    for relative, expected in manifest["files"].items():
        path = bundle / "data" / relative
        if not path.resolve().is_relative_to((bundle / "data").resolve()) or sha(path) != expected:
            raise ValueError(f"portable data changed: {relative}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Freeze completed summaries into an authenticated portable bundle")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--require-complete", action="store_true", help="Reject incomplete data and retain any prior export")
    args = parser.parse_args(argv)
    manifest = export_bundle(args.dataset, args.output, require_complete=args.require_complete)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

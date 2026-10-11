"""Fresh or explicitly resumed Cache run, analysis, figures and export."""

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .fresh_reads import FreshReadPolicy, dependency_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--publish-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--meso-root", required=True, type=Path)
    parser.add_argument("--micro-root", required=True, type=Path)
    parser.add_argument("--profile", choices=("paper", "smoke"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=("accelerate", "purego"))
    parser.add_argument("--horizon", type=float)
    parser.add_argument(
        "--resume", action="store_true", help="continue this exact run; do not change scientific settings"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    work, publish = args.work_dir.resolve(), args.publish_dir.resolve()
    if args.work_dir.is_symlink() or args.publish_dir.is_symlink() or work == publish:
        parser.error("independent real work and publication directories required")
    if args.resume and not work.is_dir():
        parser.error("resume work-dir does not exist")
    work.mkdir(parents=True, exist_ok=True)
    # Launch scripts may already have opened stdout.log. No scientific cache or
    # build is allowed to predate this process, even if it matches the same plan.
    if not args.resume and any(p.name not in ("stdout.log", "._stdout.log") for p in work.iterdir()):
        parser.error("work-dir must be fresh (only the current stdout.log may exist)")
    if not args.resume and publish.exists():
        parser.error("publish-dir must be new; existing results are never reused")
    from .resume import session_lock

    with session_lock(work):
        return run(args, work, publish)


def run(args, work, publish):
    from .resume import recover_task_lock, restore_binaries, settings

    # Legacy exports have an ownership marker. Never overwrite another run.
    if args.resume and publish.exists() and any(publish.iterdir()):
        location = publish / "run_location.json"
        manifest = work / "run_manifest.json"
        owned = location.exists() and json.loads(location.read_text())["bulk_root"] == str(work)
        if not location.exists() and manifest.exists():
            owned = json.loads(manifest.read_text())["publish_dir"] == str(publish)
        if not owned:
            # New runs create this marker before export, including partial exports.
            raise ValueError("publication directory is not owned by this run")
    publish.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from common.runtime.environment import runtime_environment

    os.environ.update(runtime_environment(work))
    os.environ["GOMODCACHE"] = str(work / "cache" / "go-modules")
    # Accelerate supplies the native BLAS used by mesoscopic matrix operations.
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    for variable in (
        "EHK_RELEASE_SCAN_INPUT_DIR",
        "EHK_RELEASE_CLOCK_INPUT_DIR",
        "EHK_REVISION_INPUT_DIR",
        "EHK_REVISION_REVIEW_INPUT_DIR",
        "EHK_MANUSCRIPT_INPUT_DIR",
    ):
        os.environ[variable] = str(work / "forbidden_historical_inputs")
    # Link on APFS to avoid zero executable headers from macOS exFAT mmap.
    # Binaries are copied/verified into Cache; scientific records and dependency
    # and build caches remain in the Cache run directory.
    stage = Path(tempfile.mkdtemp(prefix="ehk-main-link-", dir="/private/tmp")).resolve()
    (stage / "tmp").mkdir()
    cache_tmp = os.environ["TMPDIR"]
    cache_go_tmp = os.environ.get("GOTMPDIR", cache_tmp)
    os.environ["GOTMPDIR"] = str(stage / "tmp")
    os.environ["TMPDIR"] = str(stage / "tmp")
    policy = FreshReadPolicy(
        work, publish, (args.model_root, args.meso_root, args.micro_root), dependency_roots(), generated=(stage,)
    )
    policy.install()
    from common.io.json import atomic_json

    from .workflow import perform

    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=work))
    status = {"resume": args.resume, "workers": args.workers, "state": "running", "action": "verify"}
    atomic_json(attempt / "status.json", status)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        config = settings(work, publish, args)
        os.environ["CGO_ENABLED"] = "0" if args.backend == "purego" else "1"
        profile = "main_smoke" if args.profile == "smoke" else "main"
        directory = work / "evidence" / profile
        options = SimpleNamespace(
            action="plan",
            profile=profile,
            horizon=args.horizon,
            backend=args.backend,
            binary=None,
            engine=None,
            force=False,
            limit=None,
            workers=args.workers,
            build_stage=stage,
        )
        # Verify/freeze the whole design before building or reusing any task.
        perform(options, work, work / "build", args.meso_root.resolve(), args.micro_root.resolve())
        atomic_json(
            publish / "run_location.json",
            {
                "bulk_root": str(work),
                "profile": profile,
                "intermediate_inputs": "generated during this run (including explicit resume)",
                "raw_files_copied_to_model": False,
            },
        )
        if args.resume:
            recover_task_lock(directory)
        restored = restore_binaries(work, stage, config) if args.resume else None
        if restored:
            options.engine, options.binary = restored
            print("Restored original verified executables; completed tasks will be skipped", flush=True)
        else:
            options.action = "build"
            status["action"] = "build"
            atomic_json(attempt / "status.json", status)
            perform(options, work, work / "build", args.meso_root.resolve(), args.micro_root.resolve())
        os.environ["TMPDIR"] = cache_tmp
        os.environ["GOTMPDIR"] = cache_go_tmp
        for action in ("run", "analyze", "plot"):
            options.action = action
            status["action"] = action
            atomic_json(attempt / "status.json", status)
            print(f"=== {action}: {profile} ===", flush=True)
            perform(options, work, work / "build", args.meso_root.resolve(), args.micro_root.resolve())
        for name in ("figures", "tables"):
            shutil.copytree(directory / name, publish / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("._*"))
        for name in ("plan.json", "report.json", "report.md", "figure_manifest.json"):
            shutil.copyfile(directory / name, publish / name)
        status["state"] = "complete"
        print(f"Published figures and small tables to {publish}", flush=True)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        report = policy.report()
        atomic_json(attempt / "fresh_read_audit.json", report)
        atomic_json(attempt / "status.json", status)
        atomic_json(work / "fresh_read_audit.json", report)
        atomic_json(publish / "fresh_read_audit.json", report)
        shutil.rmtree(stage)
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())

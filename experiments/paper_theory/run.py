"""Fresh or explicitly resumed Cache run, analysis, figures and export."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .evidence.fresh_reads import FreshReadPolicy, dependency_roots


def launch(args):
    """Launch a parsed main command with local control and exclusive run ownership."""
    # TODO(execution-policy): decouple workers and cumulative budget from legacy
    # manifests after preserving binding-batch/spool limits and FreshReadPolicy.
    # The common ready-stream services can then be adopted without changing topics.
    siblings = Path(__file__).resolve().parents[3]
    args.meso_root = args.meso_root or siblings / "social-media-mesoscopic-models"
    args.micro_root = args.micro_root or siblings / "social-media-models"
    if not 1 <= args.workers <= 4 or args.spool_mib < 64:
        raise ValueError("workers must be 1..4; spool-mib must be >=64")
    work, publish = args.work_dir.resolve(), args.publish_dir.resolve()
    if args.work_dir.is_symlink() or args.publish_dir.is_symlink() or work == publish:
        raise ValueError("independent real work and publication directories required")
    if args.resume and not work.is_dir():
        raise ValueError("resume work-dir does not exist")
    if str(work).startswith("/Volumes/") and not Path(*work.parts[:3]).is_mount():
        raise ValueError("bulk volume is not mounted")
    work.mkdir(parents=True, exist_ok=True)
    # Launch scripts may already have opened stdout.log. No scientific cache or
    # build is allowed to predate this process, even if it matches the same plan.
    if not args.resume:
        profile = args.profile or "main"
        allowed = {
            "stdout.log",
            f"evidence/{profile}/plan.json",
            f"evidence/{profile}/measure/design.json",
            f"evidence/{profile}/micro/design.json",
        }
        if any(
            str(p.relative_to(work)) not in allowed
            for p in work.rglob("*")
            if p.is_file() and not p.name.startswith("._")
        ):
            raise ValueError("work-dir must be fresh or contain only its frozen plan; existing records require resume")
    if not args.resume and publish.exists():
        raise ValueError("publish-dir must be new; existing results are never reused")
    from common.experiments.lifecycle import interruptible, session_lock
    from common.io.json import fingerprint

    owner = work / "storage_owner.json"
    if args.control_dir is None and args.resume and owner.exists():
        args.control_dir = Path(json.loads(owner.read_text())["control"])
    args.control_dir = (
        args.control_dir or args.model_root / "outputs" / "runtime" / fingerprint({"work": str(work)})[:20]
    ).resolve()
    if args.control_dir.is_relative_to(work) or str(args.control_dir).startswith("/Volumes/"):
        raise ValueError("control-dir must be on local APFS, outside the bulk run")
    args.control_dir.mkdir(parents=True, exist_ok=True)
    with session_lock(args.control_dir), interruptible():
        return run(args, work, publish)


def run(args, work, publish):
    from common.experiments.lifecycle import recover_task_lock

    from .evidence.bindings import restore_binaries, settings

    # Exports authenticate their owning run, including partial exports.
    if args.resume and publish.exists() and any(publish.iterdir()):
        location = publish / "run_location.json"
        owned = location.exists() and json.loads(location.read_text())["bulk_root"] == str(work)
        if not owned:
            # New runs create this marker before export, including partial exports.
            raise ValueError("publication directory is not owned by this run")
    publish.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from common.runtime.environment import runtime_environment

    os.environ.update(runtime_environment(args.control_dir))
    os.environ["GOMODCACHE"] = str(args.control_dir / "cache" / "go-modules")
    # Accelerate supplies the native BLAS used by mesoscopic matrix operations.
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
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
        work,
        publish,
        (args.model_root, args.meso_root, args.micro_root),
        dependency_roots(),
        generated=(stage, args.control_dir),
    )
    policy.install()
    from common.io.json import atomic_json

    from .study import TheoryStudy

    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.control_dir))
    status = {"resume": args.resume, "workers": args.workers, "state": "running", "action": "verify"}
    atomic_json(attempt / "status.json", status)
    try:
        config = settings(work, publish, args)
        os.environ["CGO_ENABLED"] = "0" if args.backend == "purego" else "1"
        profile = args.profile
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
            resume=args.resume,
            binding_batch=getattr(args, "binding_batch", False),
            analysis_dir=publish,
            force_analysis=getattr(args, "force", False),
        )
        study = TheoryStudy(options, work, work / "build", args.meso_root.resolve(), args.micro_root.resolve())
        workflow = study.workflow(attempt / "status.json")
        status = workflow.status
        status.update(action="run", operation="plan")
        atomic_json(attempt / "status.json", status)
        study.plan()
        shutil.copyfile(directory / "plan.json", publish / "plan.json")
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
        from common.runtime.storage import RunStorage

        store = RunStorage(work, args.control_dir, max_record_bytes=args.spool_mib * (1 << 20) // (2 * args.workers))
        options.store = store
        store.recover()
        phases = tuple(
            name
            for name in ("run", "analyze", "plot")
            if name in (getattr(args, "phase", None) or ("run", "analyze", "plot"))
        )
        restored = restore_binaries(work, stage, config) if args.resume and "run" in phases else None
        if restored:
            options.engine, options.binary = restored
            print("Restored original verified executables; completed tasks will be skipped", flush=True)
        elif "run" in phases:
            status.update(action="run", operation="build")
            atomic_json(attempt / "status.json", status)
            store.read(study.build)
        os.environ["TMPDIR"] = cache_tmp
        os.environ["GOTMPDIR"] = cache_go_tmp
        workflow.guard = lambda operation: operation()
        status.pop("operation", None)
        workflow.stages["run"] = lambda: store.read(study.run)
        workflow.stages["analyze"] = lambda: store.read(study.analyze)

        def diagnostic(action, error):
            from common.runtime.failures import BatchTaskError

            if action == "run" and isinstance(error, BatchTaskError):
                try:
                    workflow.dispatch("analyze")
                except Exception as diagnostic_error:
                    store.event(None, "partial_analysis_failed", error=str(diagnostic_error))

        workflow.sequence(phases, after_failure=diagnostic)
        status["state"] = "complete"
        print(f"Completed {', '.join(phases)}; local analysis destination: {publish}", flush=True)
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        report = policy.report()
        atomic_json(attempt / "fresh_read_audit.json", report)
        atomic_json(attempt / "status.json", status)
        # The local control journal remains usable while the bulk volume is absent.
        atomic_json(args.control_dir / "fresh_read_audit.json", report)
        try:
            if "store" in locals():
                store.check()
            elif not work.is_dir():
                raise OSError("bulk run unavailable during cleanup")
            atomic_json(work / "fresh_read_audit.json", report)
        except OSError:
            pass
        atomic_json(publish / "fresh_read_audit.json", report)
        shutil.rmtree(stage)
    return 0

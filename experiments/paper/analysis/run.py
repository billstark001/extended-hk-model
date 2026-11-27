import argparse
import json
import os
import sys
from pathlib import Path

from experiments.paper.analysis.tasks import get_mode_names, run_mode

DEFAULT_STAT_CONCURRENCY = 6


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.paper.analysis.run",
        description="Generate scenario statistics and upsert them into stats DB.",
    )
    parser.add_argument("mode", choices=get_mode_names(), help="Statistic mode to run.")
    parser.add_argument("instance_name", nargs="?", help="Simulation workspace instance name defined in sim_ws.json.")
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Worker process count. Resolve order: CLI arg > STAT_THREAD_COUNT env > "
            f"default ({DEFAULT_STAT_CONCURRENCY})."
        ),
    )
    parser.add_argument("--database", type=Path, help="Independent document statistics database.")
    parser.add_argument("--raw-root", type=Path, help="Direct directory containing the raw scenario subdirectories.")
    parser.add_argument(
        "--origin", help="Explicit source dataset label; direct raw-root paths default to their absolute path."
    )
    parser.add_argument("--limit", type=int, help="Analyze only the first N planned tasks, for a bounded smoke run.")
    parser.add_argument(
        "--endpoint-complement", action=argparse.BooleanOptionalAction, default=True,
        help="Include (1-I_p(T))*I_h(T) in I_w (default); --no-endpoint-complement keeps only the observed integral.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute rather than reuse matching completed documents."
    )
    parser.add_argument(
        "--include-pathway-probe",
        action="store_true",
        help="Also collect independent P/E first passage using the native probe.",
    )
    parser.add_argument(
        "--skip-metric-audit", action="store_true", help="Omit the raw reconstruction of historical metric conventions."
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the analysis plan without reading raw data or writing statistics.",
    )
    return parser


def _resolve_concurrency(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(cli_value, 1)

    env_value_raw = os.environ.get("STAT_THREAD_COUNT")
    if env_value_raw is not None and env_value_raw.strip() != "":
        try:
            return max(int(env_value_raw.strip()), 1)
        except ValueError:
            return DEFAULT_STAT_CONCURRENCY

    return DEFAULT_STAT_CONCURRENCY


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.instance_name is None and args.raw_root is None:
        parser.error("provide an instance_name or --raw-root")

    concurrency = _resolve_concurrency(args.concurrency)
    result = run_mode(
        args.mode,
        args.instance_name,
        concurrency,
        database=args.database,
        limit=args.limit,
        force=args.force,
        include_pathway_probe=args.include_pathway_probe,
        include_metric_audit=not args.skip_metric_audit,
        plan_only=args.plan_only,
        raw_root=args.raw_root,
        origin=args.origin,
        endpoint_complement=args.endpoint_complement,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()

"""Prepare exact metric comparisons from portable records, without new simulation."""

import argparse
import json
from pathlib import Path

from .core import collect, compare, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Authenticate local bundles and inventory required raw records")
    preparation.add_argument("--bundle", type=Path, action="append", required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--raw-root", action="append", default=[], metavar="PROFILE=DATASET")
    imported = commands.add_parser("collect", help="Import existing historical metric tables with exact identities")
    imported.add_argument("--preparation", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--points", type=Path)
    imported.add_argument("--paths", type=Path)
    imported.add_argument("--collection", type=Path)
    imported.add_argument("--endpoint-complement", action=argparse.BooleanOptionalAction, default=True)
    raw = commands.add_parser("collect-raw", help="Use existing historical metrics on available persistent records")
    raw.add_argument("--preparation", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    raw.add_argument("--max-paths", type=int)
    raw.add_argument("--workers", type=int, default=1)
    raw.add_argument("--raw-root", action="append", default=[], metavar="PROFILE=DATASET")
    raw.add_argument("--endpoint-complement", action=argparse.BooleanOptionalAction, default=True)
    comparison = commands.add_parser("compare", help="Join both metric sets and report pending observations")
    comparison.add_argument("--preparation", type=Path, required=True)
    comparison.add_argument("--collection", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = {}
    if args.command in ("prepare", "collect-raw"):
        for value in args.raw_root:
            profile, separator, directory = value.partition("=")
            if not separator or not directory or profile in roots:
                parser.error("--raw-root requires distinct PROFILE=DATASET values")
            roots[profile] = directory
    if args.command == "prepare":
        result = prepare(args.bundle, args.output, raw_roots=roots)
        shown = {key: value for key, value in result.items() if key not in ("files", "sources")}
    elif args.command == "collect":
        shown = collect(args.preparation, args.output, points=args.points, paths=args.paths, collection=args.collection,
                        endpoint_complement=args.endpoint_complement)
    elif args.command == "collect-raw":
        from .raw import collect_raw
        result = collect_raw(args.preparation, args.output, max_paths=args.max_paths, raw_roots=roots, workers=args.workers,
                             endpoint_complement=args.endpoint_complement)
        shown = {key: value for key, value in result.items() if key not in ("files", "paths")}
    else:
        shown = compare(args.preparation, args.collection, args.output)
    print(json.dumps(shown, indent=2))


if __name__ == "__main__":
    main()

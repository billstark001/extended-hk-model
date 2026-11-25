"""One benchmark action in a fresh process, with explicit correctness checks."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

from common.io.json import atomic_json

from .backends import _matches, open_backend
from .common import RSSSampler, current_rss_bytes, directory_size, document_digest, latency_summary, peak_rss_bytes
from .corpus import document_at, get_profile, iter_documents


def _catalog(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def _choose_queries(catalog):
    anchor = next((row for row in catalog if row.get("mode") == "gradation"), catalog[len(catalog) // 2])
    scope = {key: anchor[key] for key in ("origin", "mode", "analysis_id")}
    selective = {**scope, **{key: anchor[key] for key in ("recsys_type", "decay", "rewiring", "repost_rate")}}
    if "post_retain_count" in anchor:
        selective["post_retain_count"] = anchor["post_retain_count"]
    nested = {**scope, "metadata.HKParams.Influence": anchor.get("decay")}
    return scope, selective, nested


def run(request):
    if request.get("sources_pack"):
        import hashlib

        import numpy as np

        from . import corpus

        packed = Path(request["sources_pack"]).read_bytes()
        if hashlib.sha256(packed).hexdigest() != request["sources_pack_sha256"]:
            raise ValueError("frozen source pack changed")
        sources = json.loads(packed)
        sources["lengths"] = np.asarray(sources["lengths"], dtype=np.int64)
        sources["length_probabilities"] = np.asarray(sources["length_probabilities"], dtype=float)
        sources["actual_files"] = [Path(path) for path in sources["actual_files"]]
        corpus.load_sources = lambda: sources
    profile, backend_name = request["profile"], request["backend"]
    profile_info = get_profile(profile)
    count = request.get("limit") or profile_info["count"]
    directory = Path(request["directory"])
    catalog = _catalog(request["catalog"])
    catalog = catalog[:count]
    action = request["action"]
    gc.collect()
    baseline_rss = current_rss_bytes()
    sampler = RSSSampler().start()
    started = time.perf_counter()
    backend = open_backend(backend_name, directory, readonly=action not in ("load", "state_update", "replace_full"))
    opened = time.perf_counter()
    result = {
        "backend": backend_name,
        "profile": profile,
        "action": action,
        "count": count,
        "repeat": request.get("repeat", 0),
        "baseline_rss_bytes": baseline_rss,
        "open_seconds": opened - started,
        "cache_regime": "fresh process; OS page cache uncontrolled",
    }
    timings = []
    try:
        if action == "load":
            total_started = time.perf_counter()
            input_digest = __import__("hashlib").sha256()
            for index, document in enumerate(iter_documents(profile, limit=count)):
                expected = catalog[index]
                if document_digest(document) != expected["full_digest"]:
                    raise ValueError("corpus is not deterministic")
                input_digest.update(expected["full_digest"].encode())
                before = time.perf_counter()
                backend.put(document)
                timings.append(time.perf_counter() - before)
                if (index + 1) % 2000 == 0:
                    print(
                        json.dumps({"loaded": index + 1, "backend": backend_name, "profile": profile}),
                        file=sys.stderr,
                        flush=True,
                    )
            result["end_to_end_seconds"] = time.perf_counter() - total_started
            result["corpus_digest"] = input_digest.hexdigest()
            if backend.count() != count:
                raise ValueError("stored document count mismatch")
            for index in sorted(set([0, count // 2, count - 1])):
                actual = backend.get(catalog[index]["_id"])
                if document_digest(actual) != catalog[index]["full_digest"]:
                    raise ValueError("stored payload mismatch")
            checkpoint_started = time.perf_counter()
            result["checkpoint"] = backend.checkpoint()
            result["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
            result["integrity"] = backend.integrity_check()
            if result["integrity"] != ["ok"]:
                raise ValueError("integrity_check did not return ok")
        elif action.startswith("point_"):
            rng = random.Random(1391)
            indices = [rng.randrange(count) for _ in range(min(count, request.get("point_operations", 200)))]
            full = action == "point_full"
            for index in indices:
                expected = catalog[index]
                before = time.perf_counter()
                actual = backend.get(expected["_id"], include_payload=full)
                timings.append(time.perf_counter() - before)
                if document_digest(actual) != expected["full_digest" if full else "header_digest"]:
                    raise ValueError("point-read content mismatch")
                del actual
            result["returned_documents"] = len(indices)
        elif action == "replace_full":
            indices = random.Random(912).sample(range(count), min(count, request.get("replace_operations", 20)))
            for index in indices:
                identifier = catalog[index]["_id"]
                document = backend.get(identifier)
                document["benchmark_replacement_epoch"] = document.get("benchmark_replacement_epoch", 0) + 1
                expected_digest = document_digest(document)
                before = time.perf_counter()
                backend.put(document)
                timings.append(time.perf_counter() - before)
                if document_digest(backend.get(identifier)) != expected_digest:
                    raise ValueError("full replacement content mismatch")
            result["returned_documents"] = len(indices)
        elif action == "state_update":
            indices = random.Random(471).sample(range(count), min(count, request.get("update_operations", 200)))
            initial = backend.get(catalog[indices[0]]["_id"], include_payload=False)
            new_complete = not initial["complete"]
            state = {"complete": new_complete, "stale": not new_complete}
            for index in indices:
                identifier = catalog[index]["_id"]
                before = time.perf_counter()
                backend.update_state(identifier, state)
                timings.append(time.perf_counter() - before)
                expected = document_at(index, profile)
                expected.update(state)
                if document_digest(backend.get(identifier)) != document_digest(expected):
                    raise ValueError("state update changed unrelated payload")
            result["state_written"] = state
        else:
            scope, selective, nested = _choose_queries(catalog)
            filters = (
                {}
                if action.startswith("scan_")
                else scope
                if action.startswith("scope_")
                else nested
                if action.startswith("nested_")
                else selective
            )
            full = action.endswith("_full")
            expected_rows = {row["_id"]: row for row in catalog if _matches(row, filters)}
            if action.startswith("nested_") and not expected_rows:
                result.update(status="not_applicable", reason="corpus has no matching nested HKParams measurement")
                return result
            result["filters"] = filters
            result["returned_documents"] = len(expected_rows)
            result["array_decodes_before"] = dict(backend.counters)
            for _ in range(request.get("query_repetitions", 3)):
                gc.collect()
                before = time.perf_counter()
                actual = backend.query(filters, projection=not full)
                timings.append(time.perf_counter() - before)
                if len(actual) != len(expected_rows) or {row["_id"] for row in actual} != expected_rows.keys():
                    raise ValueError("query returned wrong record IDs")
                for row in actual:
                    if document_digest(row) != expected_rows[row["_id"]]["full_digest" if full else "header_digest"]:
                        raise ValueError("query content mismatch")
                del actual
            result["query_repetitions"] = request.get("query_repetitions", 3)
        result.update(
            status="ok",
            latency=latency_summary(timings),
            operation_seconds=timings,
            counters=dict(backend.counters),
            pragmas=backend.pragmas(),
        )
        result["peak_rss_bytes"] = peak_rss_bytes()
        result["peak_rss_delta_bytes"] = max(0, result["peak_rss_bytes"] - baseline_rss)
    finally:
        backend.close()
        result["peak_rss_action_bytes"] = sampler.finish()
        result["peak_rss_action_delta_bytes"] = max(0, result["peak_rss_action_bytes"] - baseline_rss)
    result["files"] = directory_size(directory)
    result["process_action_seconds"] = time.perf_counter() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.request.read_text()))
    atomic_json(args.output, result)
    print(
        json.dumps(
            {key: result.get(key) for key in ("backend", "profile", "action", "status", "latency", "peak_rss_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

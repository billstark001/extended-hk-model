"""Lossless, frame-streamed MessagePack records and read-only legacy JSON support.

Extension 42 is a flat little-endian float64 vector from Go. Extension 43 stores
an ndarray's dtype, shape and raw bytes. No pickle, float32 conversion or fitted
data are involved. The footer authenticates the entire uncompressed stream.
"""

import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, cast

import lz4.frame
import msgpack
import numpy as np
from numpy.typing import NDArray

from common.types import PathInput, Record, RecordFooter, RecordHeader, TransportFailure, TransportSuccess


def encode_array(value: object) -> msgpack.ExtType:
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise TypeError(f"unsupported record type: {type(value).__name__}")
    value = np.ascontiguousarray(value)
    header = msgpack.packb([value.dtype.str, list(value.shape)], use_bin_type=True)
    return msgpack.ExtType(43, struct.pack("<I", len(header)) + header + value.tobytes())


def decode_array(code: int, data: bytes) -> NDArray[Any]:
    if code == 42:
        if len(data) % 8:
            raise ValueError("truncated float64 vector")
        return np.frombuffer(data, dtype="<f8")
    if code == 43:
        if len(data) < 4:
            raise ValueError("truncated ndarray header")
        size = struct.unpack("<I", data[:4])[0]
        dtype, shape = msgpack.unpackb(data[4 : 4 + size], raw=False)
        dtype = np.dtype(dtype)
        if dtype.hasobject or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("invalid array metadata")
        values = np.frombuffer(data[4 + size :], dtype=dtype)
        if values.size != np.prod(shape, dtype=object):
            raise ValueError("array shape/length mismatch")
        return cast(NDArray[Any], values.reshape(shape))
    raise ValueError(f"unknown record extension: {code}")


def unpack_stream(stream: BinaryIO) -> msgpack.Unpacker:
    return msgpack.Unpacker(
        stream, raw=False, strict_map_key=False, ext_hook=decode_array, read_size=1 << 20, max_buffer_size=1 << 30
    )


def read_transport(path: PathInput, identifier: str) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        objects = unpack_stream(stream)
        try:
            response = cast(TransportSuccess[dict[str, Any]] | TransportFailure, next(objects))
        except StopIteration as error:
            raise ValueError("empty or truncated solver response") from error
        if next(objects, None) is not None:
            raise ValueError("multiple solver responses")
    if response.get("id") != identifier or response.get("status") != "ok":
        raise RuntimeError(f"{identifier}: {response.get('error', 'invalid solver response')}")
    return cast(TransportSuccess[dict[str, Any]], response)["result"]


def write_record(path: PathInput, task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Atomically write bounded frame objects followed by an authenticated footer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    digest = hashlib.sha256()
    packer = msgpack.Packer(use_bin_type=True, default=encode_array)
    try:
        with os.fdopen(fd, "wb") as raw:
            with lz4.frame.open(raw, "wb", compression_level=0, content_checksum=True) as stream:

                def emit(value):
                    block = packer.pack(value)
                    digest.update(block)
                    stream.write(block)

                header: RecordHeader[Mapping[str, Any]] = {"format": "ehk-record", "schema": 2, "task": task}
                emit(header)
                metadata = {k: v for k, v in result.items() if k not in ("frames", "pair")}
                if "frames" in result:
                    metadata["frames"] = [] if result["frames"] is not None else None
                if "pair" in result:
                    metadata["pair"] = {k: v for k, v in result["pair"].items() if k != "frames"}
                    if "frames" in result["pair"]:
                        metadata["pair"]["frames"] = [] if result["pair"]["frames"] is not None else None
                emit({"kind": "metadata", "value": metadata})
                for section, frames in (
                    ("frames", result.get("frames") or []),
                    ("pair", result.get("pair", {}).get("frames") or []),
                ):
                    for frame in frames:
                        emit({"kind": "frame", "section": section, "value": frame})
                footer: RecordFooter = {"kind": "complete", "sha256": digest.hexdigest()}
                stream.write(packer.pack(footer))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_record(path: PathInput) -> Record[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".json":
        return cast(Record[dict[str, Any], dict[str, Any]], json.loads(path.read_text()))
    digest = hashlib.sha256()
    packer = msgpack.Packer(use_bin_type=True, default=encode_array)
    with lz4.frame.open(path, "rb") as stream:
        objects = unpack_stream(stream)
        header = next(objects, None)
        if not header or header.get("format") != "ehk-record" or header.get("schema") != 2:
            raise ValueError("unsupported or incomplete record header")
        digest.update(packer.pack(header))
        result: dict[str, Any] = {}
        complete = False
        for item in objects:
            if item["kind"] == "complete":
                if item["sha256"] != digest.hexdigest() or next(objects, None) is not None:
                    raise ValueError("record content hash mismatch")
                complete = True
                break
            digest.update(packer.pack(item))
            if item["kind"] == "metadata":
                result.update(item["value"])
            elif item["kind"] == "frame":
                destination = result if item["section"] == "frames" else result.setdefault("pair", {})
                destination.setdefault("frames", []).append(item["value"])
            else:
                raise ValueError("unknown record object")
        if not complete:
            raise ValueError("missing record completion footer")
    return {"task": header["task"], "result": result}


def record_path(raw: PathInput, identifier: str) -> Path:
    """Prefer the current encoding; legacy data never need a generating design."""
    raw = Path(raw)
    for suffix in (".msgpack.lz4", ".json"):
        path = raw / (identifier + suffix)
        if path.is_file():
            return path
    return raw / (identifier + ".msgpack.lz4")


def content_fingerprint(value):
    """Bounded hashing for arrays; numeric vectors agree across legacy/native forms."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            digest.update(b"map")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, np.ndarray) or (
            isinstance(item, list) and item and all(isinstance(n, (int, float, np.number)) for n in item)
        ):
            array = np.ascontiguousarray(item, dtype="<f8")
            digest.update(b"array" + str(array.shape).encode())
            digest.update(memoryview(array).cast("B"))
        elif isinstance(item, (list, tuple)):
            digest.update(b"list" + str(len(item)).encode())
            for member in item:
                visit(member)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")

    visit(value)
    return digest.hexdigest()


def promote_arrays(value, fields, dtype="<f8"):
    """Recursively convert declared numeric list fields to typed dense arrays."""
    if isinstance(value, dict):
        return {
            k: np.asarray(v, dtype=dtype)
            if k in fields and isinstance(v, list) and v and all(isinstance(n, (int, float)) for n in v)
            else promote_arrays(v, fields, dtype)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [promote_arrays(v, fields, dtype) for v in value]
    return value


def equivalent(original, converted):
    """Compare nested records exactly, including array dtype and floating bits."""
    if isinstance(converted, np.ndarray) or isinstance(original, np.ndarray):
        dtype = original.dtype if isinstance(original, np.ndarray) else converted.dtype
        a, b = np.asarray(original, dtype=dtype), np.asarray(converted)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(original, dict):
        return (
            isinstance(converted, dict)
            and original.keys() == converted.keys()
            and all(equivalent(original[k], converted[k]) for k in original)
        )
    if isinstance(original, list):
        return (
            isinstance(converted, list)
            and len(original) == len(converted)
            and all(equivalent(a, b) for a, b in zip(original, converted))
        )
    if isinstance(original, float) and isinstance(converted, float):
        return struct.pack("<d", original) == struct.pack("<d", converted)
    return original == converted

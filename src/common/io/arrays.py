"""Legacy bare array codec for stored microscopic records."""

import base64
import zlib

import numpy as np
from numpy.typing import DTypeLike, NDArray


def compress_array_to_b64(arr: NDArray) -> str:
    """Encode zlib-compressed array bytes as base64; dtype and shape must be stored separately."""
    compressed = zlib.compress(arr.tobytes())
    return base64.b64encode(compressed).decode("utf-8")


def decompress_b64_to_array(b64_str: str, dtype: DTypeLike) -> NDArray:
    """Decode a flat read-only array using the caller-supplied dtype; reshape separately."""
    compressed = base64.b64decode(b64_str.encode("utf-8"))
    return np.frombuffer(zlib.decompress(compressed), dtype=dtype)

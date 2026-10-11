"""Shape-aware little-endian array codec for saved runtime requests."""

import base64
import zlib

import numpy as np


def encoded(values):
    """Encode a numeric array as a shape-bearing base64+zlib+f64le runtime JSON value."""
    a = np.asarray(values, dtype="<f8")
    return {
        "encoding": "base64+zlib+f64le",
        "shape": "x".join(map(str, a.shape)),
        "data": base64.b64encode(zlib.compress(a.tobytes())).decode(),
    }


def decoded(array):
    """Decode that runtime format into a shaped read-only array; reject other encoding names."""
    if array["encoding"] != "base64+zlib+f64le":
        raise ValueError("unsupported array encoding")
    shape = tuple(map(int, array["shape"].split("x")))
    return np.frombuffer(zlib.decompress(base64.b64decode(array["data"])), dtype="<f8").reshape(shape)

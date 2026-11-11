"""Content identity for a model's actual inference feed tensors (not task quality)."""

from __future__ import annotations

import hashlib


def fingerprint_feeds(feeds: dict) -> str:
    """Hash named arrays, shapes, dtypes and bytes; independent of NPZ compression.

    The digest distinguishes different input contents even if dimensions match.
    Hashing happens before the timed benchmark region.
    """
    digest = hashlib.sha256()
    for name in sorted(feeds):
        array = feeds[name]
        for value in (name.encode("utf-8"), array.dtype.str.encode("ascii"),
                      repr(tuple(array.shape)).encode("ascii"), array.tobytes(order="C")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()

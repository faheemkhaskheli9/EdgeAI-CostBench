"""Deterministic synthetic ONNX inputs or user-supplied NPZ inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def parse_shapes(raw_shapes: Iterable[str]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for value in raw_shapes:
        name, sep, dims = value.partition("=")
        if not sep or not name or not dims or name in shapes:
            raise ValueError(f"invalid or duplicate --shape {value!r}; expected input=1x3x224x224")
        try:
            parsed = tuple(int(dim) for dim in dims.lower().split("x"))
        except ValueError as exc:
            raise ValueError(f"invalid dimension in --shape {value!r}") from exc
        if not parsed or any(dim <= 0 for dim in parsed):
            raise ValueError(f"--shape dimensions must be positive: {value!r}")
        shapes[name] = parsed
    return shapes


def _tensor_dtypes(np):
    return {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int64)": np.int64,
        "tensor(int32)": np.int32, "tensor(int16)": np.int16,
        "tensor(int8)": np.int8, "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32, "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_,
    }


def build_inputs(model_inputs, shape_overrides: dict[str, tuple[int, ...]],
                 seed: int = 42, npz_path: str | None = None) -> dict:
    """Construct feeds matching ONNX tensor names/types/shapes; no model correctness claim."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy required: pip install '.[cpu]'") from exc
    metadata = {meta.name: meta for meta in model_inputs}
    extras = set(shape_overrides) - set(metadata)
    if extras:
        raise ValueError(f"--shape names not in model: {sorted(extras)}")
    dtypes = _tensor_dtypes(np)
    for meta in model_inputs:
        if meta.type not in dtypes:
            raise ValueError(f"unsupported ONNX input type {meta.type!r} for {meta.name}; use numeric tensor models")
    if npz_path is not None:
        with np.load(Path(npz_path), allow_pickle=False) as archive:
            missing = set(metadata) - set(archive.files)
            extra = set(archive.files) - set(metadata)
            if missing or extra:
                raise ValueError(f"NPZ keys must match model inputs; missing={sorted(missing)}, unexpected={sorted(extra)}")
            feeds = {name: archive[name] for name in metadata}
    else:
        rng = np.random.default_rng(seed)
        feeds = {}
        for name, meta in metadata.items():
            if name in shape_overrides:
                shape = shape_overrides[name]
            else:
                if any(not isinstance(dim, int) or dim < 1 for dim in meta.shape):
                    raise ValueError(f"dynamic shape for {name}: {meta.shape}; pass --shape {name}=...")
                shape = tuple(meta.shape)
            dtype = dtypes[meta.type]
            if np.issubdtype(dtype, np.floating):
                feeds[name] = rng.standard_normal(shape).astype(dtype)
            else:
                feeds[name] = rng.integers(0, 2 if dtype is np.bool_ else 8, size=shape).astype(dtype)
    symbolic_dimensions: dict[str, int] = {}
    for name, meta in metadata.items():
        array = feeds[name]
        expected = np.dtype(dtypes[meta.type])
        if array.dtype != expected:
            raise ValueError(f"{name}: expected dtype {expected}, got {array.dtype}")
        if array.ndim != len(meta.shape):
            raise ValueError(f"{name}: expected rank {len(meta.shape)}, got {array.ndim}")
        if name in shape_overrides and tuple(array.shape) != shape_overrides[name]:
            raise ValueError(f"{name}: NPZ shape {array.shape} differs from --shape {shape_overrides[name]}")
        for expected_dim, actual_dim in zip(meta.shape, array.shape):
            if isinstance(expected_dim, int) and expected_dim > 0 and expected_dim != actual_dim:
                raise ValueError(f"{name}: model expects dimension {expected_dim}, got {actual_dim}")
            if isinstance(expected_dim, str) and expected_dim:
                previous = symbolic_dimensions.setdefault(expected_dim, actual_dim)
                if previous != actual_dim:
                    raise ValueError(f"{name}: symbolic dimension {expected_dim!r} must match {previous}, got {actual_dim}")
        if any(dim <= 0 for dim in array.shape):
            raise ValueError(f"{name}: empty tensor dimensions are not supported")
        if not array.flags.c_contiguous:
            feeds[name] = np.ascontiguousarray(array)
    return feeds

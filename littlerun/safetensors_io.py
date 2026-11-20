"""Read-only safetensors access: header inspection, hashing, and tensor loading.

Original checkpoint files are never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from .errors import CheckpointError

_MAX_HEADER = 100 * 1024 * 1024


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors dtype string, e.g. "BF16", "I32"
    shape: tuple[int, ...]
    file: str
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def read_header(path: str | os.PathLike) -> tuple[dict[str, TensorInfo], dict]:
    """Parse one safetensors header without reading tensor data."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CheckpointError(f"{path.name}: truncated safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > _MAX_HEADER:
            raise CheckpointError(f"{path.name}: implausible header length {n}")
        try:
            header = json.loads(f.read(n))
        except json.JSONDecodeError as e:
            raise CheckpointError(f"{path.name}: header is not valid JSON: {e}") from e
    meta = header.pop("__metadata__", {}) or {}
    out: dict[str, TensorInfo] = {}
    for name, h in header.items():
        out[name] = TensorInfo(name, h["dtype"], tuple(h["shape"]), path.name, tuple(h["data_offsets"]))
    return out, meta


def checkpoint_files(ckpt_dir: str | os.PathLike) -> list[Path]:
    d = Path(ckpt_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [d / f for f in sorted(set(weight_map.values()))]
    single = d / "model.safetensors"
    if single.exists():
        return [single]
    raise CheckpointError(f"no model.safetensors or index in {d}")


def read_inventory(ckpt_dir: str | os.PathLike) -> dict[str, TensorInfo]:
    inv: dict[str, TensorInfo] = {}
    for f in checkpoint_files(ckpt_dir):
        tensors, _ = read_header(f)
        dup = inv.keys() & tensors.keys()
        if dup:
            raise CheckpointError(f"duplicate tensors across shards: {sorted(dup)[:5]}")
        inv.update(tensors)
    return inv


def sha256_file(path: str | os.PathLike, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class TensorReader:
    """Lazy tensor loader over one or more safetensors files (memory-mapped)."""

    def __init__(self, ckpt_dir: str | os.PathLike):
        from safetensors import safe_open

        self._dir = Path(ckpt_dir)
        self._handles = {}
        self.inventory = read_inventory(ckpt_dir)
        for f in checkpoint_files(ckpt_dir):
            self._handles[f.name] = safe_open(str(f), framework="pt", device="cpu")

    def get(self, name: str):
        info = self.inventory.get(name)
        if info is None:
            raise CheckpointError(f"missing tensor: {name}")
        return self._handles[info.file].get_tensor(name)

    def close(self):
        self._handles.clear()

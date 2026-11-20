"""LittleBit / LittleBit-2 checkpoint codec (packing ABI "littlebit-int32-lsb-v1").

Format facts below were verified against SamsungLabs/LittleBit at commit
42d658b0c79f76450b34b6a3547462c7cdc3e1a0 (quantization/utils/binary_packer.py,
quantization/modules/littlebit.py) and against the safetensors header of
LittleBitLLM/littlebit-qwen3-8b-0.55bpw at revision 6b2c47aa97.

Per factorized linear `<prefix>` (out_features = n, in_features = m, rank = s):

  <prefix>.U_packed   int32 [n, ceil(s/32)]   signs of U (n x s)
  <prefix>.U_shape    int64 [2]               == [n, s]
  <prefix>.V_packed   int32 [s, ceil(m/32)]   signs of V (s x m)
  <prefix>.V_shape    int64 [2]               == [s, m]
  <prefix>.u1         float [1, n]            output (row) scale
  <prefix>.u2         float [1, s]            latent scale (U side)
  <prefix>.v1         float [1, s]            latent scale (V side)
  <prefix>.v2         float [1, m]            input (column) scale
  optional: the same set with an "_R" suffix (residual path), <prefix>.bias [n],
  and bookkeeping scalars _eff_bit_actual, _eff_bit_target, _split_dim_final.

Sign convention: bit 0 -> +1, bit 1 -> -1. sign(0) is +1 at quantization time.
Bit order: least-significant bit first within each int32 word, row-major.
Padding: rows are padded to a multiple of 32 with +1 (bit 0).

Forward (reference order, preserved exactly):
  y = ((((x * v2) @ V^T) * (v1 * u2)) @ U^T) * u1   [+ same for _R] [+ bias]
which equals x @ W^T with W = diag(u1) U diag(u2) diag(v1) V diag(v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..errors import CheckpointError
from ..safetensors_io import TensorInfo, TensorReader

FAMILY = "littlebit"
CODEC_VERSION = "1"
PACKING_ABI = "littlebit-int32-lsb-v1"
OPERATION = "littlebit.factorized_linear.v1"

_FLOAT_DTYPES = {"BF16", "F16", "F32"}
_PATH_KEYS = ("U_packed", "U_shape", "V_packed", "V_shape", "u1", "u2", "v1", "v2")
_META_KEYS = ("_eff_bit_actual", "_eff_bit_target", "_split_dim_final")


# ----------------------------------------------------------------------------- packing


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack a 2-D sign matrix into int32 words (LSB first, bit 1 = negative).

    Accepts int8 {-1,+1} or any float/int tensor; values < 0 map to -1, others to +1.
    """
    if signs.dim() != 2:
        raise ValueError("pack_signs expects a 2-D tensor")
    rows, cols = signs.shape
    words = (cols + 31) // 32
    bits = (signs < 0).to(torch.int64)
    pad = words * 32 - cols
    if pad:
        bits = torch.cat([bits, torch.zeros(rows, pad, dtype=torch.int64, device=bits.device)], dim=1)
    weights = torch.ones(32, dtype=torch.int64, device=bits.device) << torch.arange(32, device=bits.device)
    packed = (bits.view(rows, words, 32) * weights).sum(dim=2)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)  # two's complement wrap
    return packed.to(torch.int32).contiguous()


def unpack_signs(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """Inverse of pack_signs. Returns int8 {-1,+1} of shape [rows, cols]."""
    if packed.dtype != torch.int32 or packed.dim() != 2:
        raise CheckpointError(f"packed factor must be 2-D int32, got {packed.dtype} {tuple(packed.shape)}")
    rows, words = packed.shape
    if words != (cols + 31) // 32:
        raise CheckpointError(f"packed width {words} does not match {cols} columns")
    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(rows, words * 32)[:, :cols]
    return (1 - 2 * bits).to(torch.int8)


def padding_is_clean(packed: torch.Tensor, cols: int) -> bool:
    """True when every padding bit in the last word is 0 (i.e. padded with +1)."""
    pad = (-cols) % 32
    if pad == 0:
        return True
    used = 32 - pad
    last = packed[:, -1].to(torch.int64) & 0xFFFFFFFF
    return bool(((last >> used) == 0).all())


# ----------------------------------------------------------------------------- structure


@dataclass
class FactorPath:
    rank: int
    U_packed: str
    V_packed: str
    scales: dict[str, str]  # u1,u2,v1,v2 -> tensor name


@dataclass
class FactorGroup:
    prefix: str
    in_features: int
    out_features: int
    paths: list[FactorPath]  # [primary] or [primary, residual]
    bias: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def residual(self) -> bool:
        return len(self.paths) > 1

    @property
    def ranks(self) -> list[int]:
        return [p.rank for p in self.paths]

    def bits(self) -> int:
        """Stored bits for factors + scales (excludes bias/bookkeeping), 16-bit scales."""
        n, m = self.out_features, self.in_features
        return sum(p.rank * (n + m) + 16 * (n + m + 2 * p.rank) for p in self.paths)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class LittleBitCodec:
    """Validates, inspects and decodes LittleBit factor tensors."""

    family = FAMILY
    version = CODEC_VERSION
    packing_abi = PACKING_ABI

    # --------------------------------------------------------------- inspect / validate
    def inspect(self, inventory: dict[str, TensorInfo]) -> list[str]:
        """Return the module prefixes that look like LittleBit factor groups."""
        return sorted(n[: -len(".U_packed")] for n in inventory if n.endswith(".U_packed"))

    def validate_tensors(
        self,
        inventory: dict[str, TensorInfo],
        reader: TensorReader | None,
        expected: dict[str, tuple[int, int]],
        *,
        residual: bool | None = None,
    ) -> tuple[dict[str, FactorGroup], set[str]]:
        """Validate every expected factorized linear. Returns (groups, claimed tensor names).

        expected: prefix -> (in_features, out_features), declared by the architecture adapter.
        reader: when given, the small *_shape / _split_dim_final tensors and padding bits are checked.
        """
        found = set(self.inspect(inventory))
        missing = sorted(set(expected) - found)
        if missing:
            raise CheckpointError(f"missing factor groups ({len(missing)}): {missing[:5]}")
        unexpected = sorted(found - set(expected))
        if unexpected:
            raise CheckpointError(f"factor groups not declared by the adapter: {unexpected[:5]}")

        groups: dict[str, FactorGroup] = {}
        claimed: set[str] = set()
        for prefix, (m, n) in sorted(expected.items()):
            paths = []
            for suffix in ("", "_R"):
                # U_packed -> U_R_packed; u1 -> u1_R
                names = {
                    k: (f"{prefix}.{k.split('_')[0]}{suffix}_{k.split('_')[1]}" if "_" in k else f"{prefix}.{k}{suffix}")
                    for k in _PATH_KEYS
                }
                present = [nm for nm in names.values() if nm in inventory]
                if suffix == "_R" and not present:
                    break
                absent = [nm for nm in names.values() if nm not in inventory]
                if absent:
                    raise CheckpointError(f"{prefix}: incomplete factor path, missing {absent}")
                paths.append(self._validate_path(prefix + suffix, names, inventory, reader, m, n))
                claimed.update(names.values())
            if residual is not None and (len(paths) > 1) != residual:
                raise CheckpointError(f"{prefix}: residual={len(paths) > 1} disagrees with config residual={residual}")

            bias = f"{prefix}.bias"
            if bias in inventory:
                info = inventory[bias]
                if info.shape != (n,) or info.dtype not in _FLOAT_DTYPES:
                    raise CheckpointError(f"{bias}: expected float [{n}], got {info.dtype} {list(info.shape)}")
                claimed.add(bias)
            else:
                bias = None

            meta = {}
            for k in _META_KEYS:
                nm = f"{prefix}.{k}"
                if nm in inventory:
                    meta[k] = nm
                    claimed.add(nm)
            if reader is not None and "_split_dim_final" in meta:
                s_meta = int(reader.get(meta["_split_dim_final"]).item())
                if s_meta != paths[0].rank:
                    raise CheckpointError(f"{prefix}: _split_dim_final={s_meta} but factor rank={paths[0].rank}")
            groups[prefix] = FactorGroup(prefix, m, n, paths, bias, meta)
        return groups, claimed

    def _validate_path(self, label, names, inv, reader, m, n) -> FactorPath:
        U, V = inv[names["U_packed"]], inv[names["V_packed"]]
        for t in (U, V):
            if t.dtype != "I32" or len(t.shape) != 2:
                raise CheckpointError(f"{t.name}: expected 2-D I32 packed tensor, got {t.dtype} {list(t.shape)}")
        for k in ("U_shape", "V_shape"):
            t = inv[names[k]]
            if t.dtype != "I64" or t.shape != (2,):
                raise CheckpointError(f"{t.name}: expected I64 [2], got {t.dtype} {list(t.shape)}")
        s = V.shape[0]
        if U.shape != (n, _cdiv(s, 32)):
            raise CheckpointError(f"{U.name}: shape {list(U.shape)} != [{n}, {_cdiv(s, 32)}] (out={n}, rank={s})")
        if V.shape != (s, _cdiv(m, 32)):
            raise CheckpointError(f"{V.name}: shape {list(V.shape)} != [{s}, {_cdiv(m, 32)}] (in={m})")
        want = {"u1": n, "u2": s, "v1": s, "v2": m}
        for k, size in want.items():
            t = inv[names[k]]
            if t.dtype not in _FLOAT_DTYPES:
                raise CheckpointError(f"{t.name}: scale dtype {t.dtype} unsupported")
            if t.shape not in ((1, size), (size,)):
                raise CheckpointError(f"{t.name}: shape {list(t.shape)} != [1, {size}]")
        if reader is not None:
            us = tuple(reader.get(names["U_shape"]).tolist())
            vs = tuple(reader.get(names["V_shape"]).tolist())
            if us != (n, s) or vs != (s, m):
                raise CheckpointError(f"{label}: stored shapes U{us} V{vs} disagree with layout U({n},{s}) V({s},{m})")
            if not padding_is_clean(reader.get(names["U_packed"]), s):
                raise CheckpointError(f"{names['U_packed']}: nonzero padding bits")
            if not padding_is_clean(reader.get(names["V_packed"]), m):
                raise CheckpointError(f"{names['V_packed']}: nonzero padding bits")
        return FactorPath(s, names["U_packed"], names["V_packed"], {k: names[k] for k in want})

    # --------------------------------------------------------------- decode
    def load_factors(self, group: FactorGroup, reader: TensorReader) -> dict:
        """Decode one group into sign matrices (int8) and float scales (as stored)."""
        out = {"paths": [], "bias": None}
        for p in group.paths:
            U = unpack_signs(reader.get(p.U_packed), p.rank)
            V = unpack_signs(reader.get(p.V_packed), group.in_features)
            sc = {k: reader.get(v).reshape(-1) for k, v in p.scales.items()}
            out["paths"].append({"U": U, "V": V, **sc})
        if group.bias:
            out["bias"] = reader.get(group.bias)
        return out

    def load_packed(self, group: FactorGroup, reader: TensorReader) -> dict:
        """Return the packed words as stored (already validated) plus float scales."""
        out = {"paths": [], "bias": None}
        for p in group.paths:
            sc = {k: reader.get(v).reshape(-1) for k, v in p.scales.items()}
            out["paths"].append({"U_packed": reader.get(p.U_packed), "V_packed": reader.get(p.V_packed),
                                 "rank": p.rank, **sc})
        if group.bias:
            out["bias"] = reader.get(group.bias)
        return out

    def operation_spec(self) -> str:
        return OPERATION


# ----------------------------------------------------------------------------- bit budget


def samsung_rank_for_bits(in_features: int, out_features: int, eff_bit: float, residual: bool,
                          ratio: float = 1.0, min_rank: int = 8) -> int:
    """Rank chosen by the reference trainer for a bit target (documentation/testing aid)."""
    a, b = in_features, out_features
    base = a + b + 16
    if residual:
        est = (a * b * eff_bit - 32 * (a + b)) / (2 * base)
    else:
        est = (a * b * eff_bit - 16 * (a + b)) / base
    est = int(est * ratio)
    est = (est // 8) * 8
    return max(est if est > 0 else min_rank, min_rank)

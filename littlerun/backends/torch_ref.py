"""PyTorch reference backend for the LittleBit factorized linear operator.

The forward order and per-step dtype match the SamsungLabs reference
(`LittleBitLinear._compute_forward`), so bf16 results are comparable step by step.
Full dense weights are never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..codecs.littlebit import OPERATION

FACTOR_STORAGE = ("int8", "compute", "int8mm", "packed")
PACKED_GEMV_MAX_ROWS = 16  # packed kernel beats unpack+GEMM up to ~16 rows (measured on CPU, 8B factor shapes)
INT8MM_MAX_ROWS = 16  # fused int8 kernel is only faster for decode-sized inputs (measured on CPU)
_K_ALIGN = 32  # aten._weight_int8pack_mm returns wrong results / crashes for K % 32 != 0 (measured, torch 2.15)


def int8mm_available() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


def _pad_cols(t: torch.Tensor, mult: int) -> torch.Tensor:
    pad = (-t.shape[1]) % mult
    return torch.nn.functional.pad(t, (0, pad)) if pad else t


@dataclass(frozen=True)
class OperationSpec:
    op: str
    dtype: torch.dtype
    device: str  # "cpu" | "cuda"
    packing_abi: str


class FactorizedLinear(nn.Module):
    """y = sum_paths ((((x*v2) @ V^T) * (v1*u2)) @ U^T) * u1  (+ bias).

    factor_storage:
      "int8"    keep sign factors as int8 and cast per call (lowest resident memory)
      "compute" pre-cast sign factors to the compute dtype (faster, 2x factor memory in bf16)
      "int8mm"  int8 factors with K zero-padded to 32; inputs with <= INT8MM_MAX_ROWS rows use
                aten._weight_int8pack_mm with the latent/output scales fused (CPU, bf16/fp16 only).
                Larger inputs fall back to the cast path. Not bit-identical to the reference:
                scales are applied before the final bf16 rounding.
      "packed"  keep only the 1-bit packed words (8x less than int8). Inputs with
                <= PACKED_GEMV_MAX_ROWS rows run the compiled packed-sign kernel (fp32 accumulate);
                larger inputs unpack each factor to a dense scratch matrix and use GEMM. CPU only.
    """

    def __init__(self, in_features: int, out_features: int, paths: list[dict], bias: torch.Tensor | None,
                 dtype: torch.dtype, factor_storage: str = "int8"):
        super().__init__()
        if factor_storage not in FACTOR_STORAGE:
            raise ValueError(f"factor_storage must be one of {FACTOR_STORAGE}")
        self.in_features, self.out_features = in_features, out_features
        self.n_paths = len(paths)
        self.factor_storage = factor_storage
        self.compute_dtype = dtype
        if factor_storage == "int8mm":
            if not int8mm_available():
                raise ValueError("factor_storage=int8mm needs torch.ops.aten._weight_int8pack_mm")
            if dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("factor_storage=int8mm supports bf16/fp16 compute only")
        if factor_storage == "packed":
            from ..kernels import packed_kernel, packed_kernel_error

            self._kernel = packed_kernel()
            if self._kernel is None:
                raise ValueError(f"factor_storage=packed needs the compiled kernel: {packed_kernel_error()}")
        fdt = dtype if factor_storage == "compute" else torch.int8
        self._ranks = []
        for i, p in enumerate(paths):
            if factor_storage == "packed":
                self._ranks.append(int(p["rank"]))
                assert p["U_packed"].shape[0] == out_features and p["V_packed"].shape[0] == p["rank"]
                self.register_buffer(f"U{i}", p["U_packed"].contiguous())
                self.register_buffer(f"V{i}", p["V_packed"].contiguous())
                self._register_scales(i, p, dtype)
                continue
            U, V = p["U"], p["V"]
            assert U.shape[0] == out_features and V.shape[1] == in_features and U.shape[1] == V.shape[0]
            self._ranks.append(U.shape[1])
            U, V = U.to(fdt), V.to(fdt)
            if factor_storage == "int8mm":
                U, V = _pad_cols(U, _K_ALIGN), _pad_cols(V, _K_ALIGN)
            self.register_buffer(f"U{i}", U.contiguous())
            self.register_buffer(f"V{i}", V.contiguous())
            self._register_scales(i, p, dtype)
        self.register_buffer("bias", None if bias is None else bias.to(dtype))

    def _register_scales(self, i, p, dtype):
        # Scales: cast to compute dtype, then mid = v1*u2 in compute dtype (reference order).
        v1, u2 = p["v1"].to(dtype), p["u2"].to(dtype)
        self.register_buffer(f"mid{i}", (v1 * u2).contiguous())
        self.register_buffer(f"u1_{i}", p["u1"].to(dtype).contiguous())
        self.register_buffer(f"v2_{i}", p["v2"].to(dtype).contiguous())

    @property
    def rank(self) -> list[int]:
        return list(self._ranks)

    def _path_int8mm(self, x: torch.Tensor, i: int) -> torch.Tensor:
        U, V = getattr(self, f"U{i}"), getattr(self, f"V{i}")
        xs = x * getattr(self, f"v2_{i}")
        if V.shape[1] != self.in_features:
            xs = torch.nn.functional.pad(xs, (0, V.shape[1] - self.in_features))
        t = torch.ops.aten._weight_int8pack_mm(xs.contiguous(), V, getattr(self, f"mid{i}"))
        if U.shape[1] != t.shape[1]:
            t = torch.nn.functional.pad(t, (0, U.shape[1] - t.shape[1]))
        return torch.ops.aten._weight_int8pack_mm(t.contiguous(), U, getattr(self, f"u1_{i}"))

    def _signed(self, packed: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tf = t.float()
        pad = packed.shape[1] * 32 - tf.shape[1]
        if pad:
            tf = torch.nn.functional.pad(tf, (0, pad))
        return self._kernel.signed_matmul(packed, tf.contiguous(), False).to(t.dtype)

    def _unpacked(self, packed: torch.Tensor, cols: int, dtype) -> torch.Tensor:
        kdt = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
        return self._kernel.unpack_signs(packed, kdt)[:, :cols].to(dtype)

    def _path_packed(self, x: torch.Tensor, i: int) -> torch.Tensor:
        Up, Vp = getattr(self, f"U{i}"), getattr(self, f"V{i}")
        xs = x * getattr(self, f"v2_{i}")
        if x.shape[0] <= PACKED_GEMV_MAX_ROWS:
            t = self._signed(Vp, xs) * getattr(self, f"mid{i}")
            return self._signed(Up, t) * getattr(self, f"u1_{i}")
        t = (xs @ self._unpacked(Vp, self.in_features, x.dtype).t()) * getattr(self, f"mid{i}")
        return (t @ self._unpacked(Up, self._ranks[i], x.dtype).t()) * getattr(self, f"u1_{i}")

    def _path(self, x: torch.Tensor, i: int) -> torch.Tensor:
        if self.factor_storage == "packed":
            return self._path_packed(x, i)
        if self.factor_storage == "int8mm" and x.shape[0] <= INT8MM_MAX_ROWS and x.device.type == "cpu":
            return self._path_int8mm(x, i)
        U, V = getattr(self, f"U{i}"), getattr(self, f"V{i}")
        if self.factor_storage == "int8mm":
            U, V = U[:, : self._ranks[i]], V[:, : self.in_features]
        if U.dtype != x.dtype:
            U, V = U.to(x.dtype), V.to(x.dtype)
        t = (x * getattr(self, f"v2_{i}")) @ V.t()
        t = t * getattr(self, f"mid{i}")
        t = t @ U.t()
        return t * getattr(self, f"u1_{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        y = self._path(x2, 0)
        for i in range(1, self.n_paths):
            y = y + self._path(x2, i)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, ranks={self.rank}, "
                f"storage={self.factor_storage}, bias={self.bias is not None}")


class TorchReferenceBackend:
    name = "torch-ref"

    def supports(self, spec: OperationSpec) -> bool:
        return (spec.op == OPERATION and spec.device in ("cpu", "cuda")
                and spec.dtype in (torch.float32, torch.bfloat16, torch.float16))

    def build_linear(self, in_features, out_features, decoded: dict, dtype, factor_storage="int8") -> nn.Module:
        return FactorizedLinear(in_features, out_features, decoded["paths"], decoded["bias"], dtype, factor_storage)

    def diagnostics(self) -> dict:
        d = {"backend": self.name, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            d["device_name"] = torch.cuda.get_device_name(0)
        return d

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

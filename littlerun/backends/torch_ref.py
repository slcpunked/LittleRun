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

FACTOR_STORAGE = ("int8", "compute")


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
        fdt = torch.int8 if factor_storage == "int8" else dtype
        for i, p in enumerate(paths):
            U, V = p["U"], p["V"]
            assert U.shape[0] == out_features and V.shape[1] == in_features and U.shape[1] == V.shape[0]
            self.register_buffer(f"U{i}", U.to(fdt).contiguous())
            self.register_buffer(f"V{i}", V.to(fdt).contiguous())
            # Scales: cast to compute dtype, then mid = v1*u2 in compute dtype (reference order).
            v1, u2 = p["v1"].to(dtype), p["u2"].to(dtype)
            self.register_buffer(f"mid{i}", (v1 * u2).contiguous())
            self.register_buffer(f"u1_{i}", p["u1"].to(dtype).contiguous())
            self.register_buffer(f"v2_{i}", p["v2"].to(dtype).contiguous())
        self.register_buffer("bias", None if bias is None else bias.to(dtype))

    @property
    def rank(self) -> list[int]:
        return [getattr(self, f"U{i}").shape[1] for i in range(self.n_paths)]

    def _path(self, x: torch.Tensor, i: int) -> torch.Tensor:
        U, V = getattr(self, f"U{i}"), getattr(self, f"V{i}")
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

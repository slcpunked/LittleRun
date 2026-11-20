"""Checkpoint loader: manifest -> compatibility resolution -> memory budget -> model.

Compatibility and memory are resolved before any large allocation.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .backends.torch_ref import OperationSpec
from .errors import CheckpointError, CompatibilityError
from .manifest import MANIFEST_NAME, load_manifest, verify_against_checkpoint
from .registry import ARCHITECTURES, BACKENDS, CODECS
from .safetensors_io import TensorReader

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_FLOATS = {"BF16", "F16", "F32"}


@dataclass
class LoadOptions:
    device: str = "auto"
    dtype: str = "bf16"
    factor_storage: str = "auto"  # "auto" | "int8" | "compute" | "int8mm"
    max_context: int = 4096
    verify_hashes: bool = True
    memory_headroom: float = 0.90  # fraction of available memory the estimate may use


@dataclass
class LoadedModel:
    model_id: str
    ckpt_dir: Path
    manifest: dict
    adapter: object
    backend: object
    cfg: object
    model: torch.nn.Module
    tokenizer: object
    generation_config: dict
    device: torch.device
    dtype: torch.dtype
    options: LoadOptions
    memory_estimate: dict = field(default_factory=dict)
    load_seconds: float = 0.0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise CompatibilityError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def available_memory(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise CompatibilityError("cannot determine available host memory")


def estimate_memory(groups, dense_shapes, cfg, adapter, opts: LoadOptions, dtype) -> dict:
    el = torch.empty((), dtype=dtype).element_size()
    if opts.factor_storage == "packed":
        factors = sum(4 * (g.out_features * -(-p.rank // 32) + p.rank * -(-g.in_features // 32))
                      for g in groups.values() for p in g.paths)
    else:
        fel = el if opts.factor_storage == "compute" else 1
        factors = sum(p.rank * (g.in_features + g.out_features) * fel for g in groups.values() for p in g.paths)
    scales = sum((g.in_features + g.out_features + p.rank) * el for g in groups.values() for p in g.paths)
    dense = 0
    for name, shape in dense_shapes.items():
        n = 1
        for s in shape:
            n *= s
        dense += n * el
    kv = adapter.kv_cache_bytes(cfg, opts.max_context, dtype)
    # Workspace: largest transient = one factor cast (int8 mode) + MLP activations for a 512-token prefill chunk.
    biggest = max(p.rank * max(g.in_features, g.out_features) for g in groups.values() for p in g.paths)
    workspace = (biggest * el if opts.factor_storage != "compute" else 0) + 512 * cfg.intermediate_size * el * 4
    workspace += cfg.vocab_size * 4 * 2  # fp32 logits
    total = factors + scales + dense + kv + workspace
    return {"factors": factors, "scales": scales, "dense": dense, "kv_cache": kv, "workspace_est": workspace,
            "total_est": total}


def _resolve_storage(opts, device, dtype, groups, dense_shapes, cfg, adapter, avail) -> LoadOptions:
    """CPU -> packed (if the kernel builds), else int8mm (bf16/fp16). GPU -> compute if it fits, else int8."""
    from dataclasses import replace

    from .backends.torch_ref import int8mm_available
    from .kernels import packed_kernel

    if device.type == "cpu" and packed_kernel() is not None:
        return replace(opts, factor_storage="packed")
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16) and int8mm_available():
        return replace(opts, factor_storage="int8mm")
    compute = replace(opts, factor_storage="compute")
    if estimate_memory(groups, dense_shapes, cfg, adapter, compute, dtype)["total_est"] <= avail * opts.memory_headroom:
        return compute
    return replace(opts, factor_storage="int8")


def load(ckpt_dir: str | Path, *, manifest_path: str | Path | None = None, model_id: str | None = None,
         options: LoadOptions | None = None, progress=None) -> LoadedModel:
    t0 = time.time()
    opts = options or LoadOptions()
    d = Path(ckpt_dir)
    say = progress or (lambda msg: None)

    say("reading manifest")
    m = load_manifest(manifest_path or d / MANIFEST_NAME)
    say("verifying checkpoint against manifest" + (" (sha256)" if opts.verify_hashes else ""))
    verify_against_checkpoint(m, d, verify_hashes=opts.verify_hashes)

    arch = m["architecture"]
    adapter = ARCHITECTURES.get((arch["id"], arch["adapter_revision"]))()
    comp = m["compression"]
    codec = CODECS.get((comp["family"], comp["version"]))()
    if opts.dtype not in _DTYPES:
        raise CompatibilityError(f"dtype must be one of {sorted(_DTYPES)}")
    dtype = _DTYPES[opts.dtype]
    device = resolve_device(opts.device)
    spec = OperationSpec(codec.operation_spec(), dtype, device.type, codec.packing_abi)
    backend = BACKENDS.resolve(spec)

    cfg = adapter.validate_config(m["model_config"])
    if opts.max_context > cfg.max_position_embeddings:
        raise CompatibilityError(f"max_context {opts.max_context} > max_position_embeddings")

    reader = TensorReader(d)
    inv = reader.inventory
    lb_cfg = m.get("littlebit_config") or {}
    groups, claimed = codec.validate_tensors(inv, reader, adapter.linear_shapes(cfg), residual=lb_cfg.get("residual"))
    dense_shapes = adapter.dense_tensors(cfg)
    for name, shape in dense_shapes.items():
        info = inv.get(name)
        if info is None:
            raise CheckpointError(f"missing dense tensor: {name}")
        if info.shape != shape or info.dtype not in _FLOATS:
            raise CheckpointError(f"{name}: expected float {list(shape)}, got {info.dtype} {list(info.shape)}")
    unclaimed = sorted(set(inv) - claimed - set(dense_shapes))
    if unclaimed:
        raise CheckpointError(f"unknown tensors (fail closed): {unclaimed[:5]}")

    avail = available_memory(device)
    if opts.factor_storage == "auto":
        opts = _resolve_storage(opts, device, dtype, groups, dense_shapes, cfg, adapter, avail)
        say(f"factor_storage=auto resolved to {opts.factor_storage}")
    if opts.factor_storage == "packed" and device.type != "cpu":
        raise CompatibilityError("factor_storage=packed is CPU-only in v0.2")
    est = estimate_memory(groups, dense_shapes, cfg, adapter, opts, dtype)
    say(f"memory estimate {est['total_est'] / 2**30:.2f} GiB on {device} (available {avail / 2**30:.2f} GiB)")
    if est["total_est"] > avail * opts.memory_headroom:
        raise CompatibilityError(
            f"estimated {est['total_est'] / 2**30:.2f} GiB exceeds {opts.memory_headroom:.0%} of available "
            f"{avail / 2**30:.2f} GiB on {device}; reduce max_context or use factor_storage=int8")

    say(f"decoding {len(groups)} factorized linears")
    linears = {}
    for i, (prefix, g) in enumerate(sorted(groups.items())):
        decoded = codec.load_packed(g, reader) if opts.factor_storage == "packed" else codec.load_factors(g, reader)
        linears[prefix] = backend.build_linear(g.in_features, g.out_features, decoded, dtype,
                                               opts.factor_storage).to(device)
        del decoded
        if i % 50 == 0:
            gc.collect()
    say("loading dense tensors")
    dense = {name: reader.get(name).to(device=device, dtype=dtype) for name in dense_shapes}
    reader.close()

    model = adapter.build_model(cfg, linears, dense, dtype).to(device).eval()
    del dense, linears
    gc.collect()

    say("loading tokenizer")
    tokenizer = adapter.load_tokenizer(d)
    gen_path = d / "generation_config.json"
    gen_cfg = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    mid = model_id or Path(m["source"]["repository"]).name
    return LoadedModel(mid, d, m, adapter, backend, cfg, model, tokenizer, gen_cfg, device, dtype, opts, est,
                       time.time() - t0)

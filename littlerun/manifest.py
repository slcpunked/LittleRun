"""Versioned checkpoint manifest (schema v1).

A manifest is built once from a checkpoint by explicit command, reviewed, and then
required at load time. Nothing is inferred from file names or bpw labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .codecs.littlebit import (CODEC_VERSION, FAMILY, OPERATION, PACKING_ABI, LittleBitCodec)
from .errors import ManifestError
from .safetensors_io import TensorReader, checkpoint_files, read_inventory, sha256_file

SCHEMA_VERSION = 1
MANIFEST_NAME = "littlerun.manifest.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json")
CONFIG_FILES = ("config.json", "littlebit_config.json", "generation_config.json")

COMPRESSION_CONTRACT = {
    "family": FAMILY,
    "version": CODEC_VERSION,
    "packing_abi": PACKING_ABI,
    "operation": OPERATION,
    "factor_orientation": "U:[out,rank] V:[rank,in]; W = diag(u1) U diag(u2*v1) V diag(v2)",
    "forward_order": "((((x*v2) @ V^T) * (v1*u2)) @ U^T) * u1, residual path added, then bias",
    "sign_convention": "bit0=+1 bit1=-1; sign(0)=+1",
    "bit_order": "lsb_first, int32 words, row-major",
    "padding": "rows padded to 32 columns with +1 (bit 0); padding bits must be 0",
    "endianness": "little (safetensors)",
    "scale_semantics": {"u1": "per-output", "u2": "per-latent (U side)", "v1": "per-latent (V side)",
                        "v2": "per-input"},
}


def inventory_digest(inv) -> str:
    h = hashlib.sha256()
    for name in sorted(inv):
        t = inv[name]
        h.update(f"{name}|{t.dtype}|{','.join(map(str, t.shape))}\n".encode())
    return h.hexdigest()


def build_manifest(ckpt_dir: str | Path, *, source_repository: str, source_revision: str,
                   license_id: str, base_model: str | None = None, base_model_license: str | None = None,
                   arch_id: str = "qwen3", adapter_revision: str = "1", hash_files: bool = True) -> dict:
    from .registry import ARCHITECTURES

    d = Path(ckpt_dir)
    raw_cfg = json.loads((d / "config.json").read_text())
    lb_cfg = json.loads((d / "littlebit_config.json").read_text()) if (d / "littlebit_config.json").exists() else None
    adapter = ARCHITECTURES.get((arch_id, adapter_revision))()
    cfg = adapter.validate_config(raw_cfg)

    inv = read_inventory(d)
    reader = TensorReader(d)
    codec = LittleBitCodec()
    groups, claimed = codec.validate_tensors(inv, reader, adapter.linear_shapes(cfg),
                                             residual=(lb_cfg or raw_cfg).get("residual"))
    reader.close()
    dense = adapter.dense_tensors(cfg)
    unclaimed = sorted(set(inv) - claimed - set(dense))
    if unclaimed:
        raise ManifestError(f"unclaimed tensors: {unclaimed[:5]}")

    # Rank partitions summarized per projection type; must be uniform across layers.
    ranks: dict[str, set] = {}
    for prefix, g in groups.items():
        kind = prefix.split(".", 3)[-1]
        ranks.setdefault(kind, set()).add(tuple(g.ranks))
    rank_summary = {k: [list(r) for r in sorted(v)] for k, v in sorted(ranks.items())}

    factor_bits = sum(g.bits() for g in groups.values())
    linear_params = sum(g.in_features * g.out_features for g in groups.values())

    files = {}
    for f in checkpoint_files(d):
        files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256_file(f) if hash_files else None}
    tok = {n: sha256_file(d / n) for n in TOKENIZER_FILES if (d / n).exists()}
    cfgs = {n: sha256_file(d / n) for n in CONFIG_FILES if (d / n).exists()}

    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": {"id": arch_id, "adapter_revision": adapter_revision},
        "compression": COMPRESSION_CONTRACT,
        "source": {"repository": source_repository, "revision": source_revision, "weight_files": files,
                   "config_files": cfgs},
        "tokenizer": {"files": tok, "chat_template_sha256": adapter.chat_template_sha256(d)},
        "model_config": raw_cfg,
        "littlebit_config": lb_cfg,
        "tensors": {
            "count": len(inv),
            "inventory_sha256": inventory_digest(inv),
            "entries": [[n, inv[n].dtype, list(inv[n].shape)] for n in sorted(inv)],
        },
        "factors": {
            "groups": len(groups),
            "residual": any(g.residual for g in groups.values()),
            "bias": any(g.bias for g in groups.values()),
            "rank_partitions": rank_summary,
            "factorized_linear_params": linear_params,
            "factor_and_scale_bits": factor_bits,
            "measured_bits_per_weight": round(factor_bits / linear_params, 6),
        },
        "attention": {"type": "gqa", "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
                      "head_dim": cfg.head_dim, "qk_norm": "rmsnorm-per-head", "rope_theta": cfg.rope_theta,
                      "rope_scaling": None, "sliding_window": None},
        "context": {"max_position_embeddings": cfg.max_position_embeddings, "validated_context": None},
        "provenance": {"license": license_id, "base_model": base_model, "base_model_license": base_model_license},
        "capabilities": {"completions": True, "chat": True, "streaming": True, "tool_calling": False,
                         "notes": "tool_calling is not validated; requests with tools are rejected"},
    }


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest not found: {p} (build one with `littlerun manifest build`)")
    try:
        m = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON: {e}") from e
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema_version {m.get('schema_version')!r}")
    for key in ("architecture", "compression", "source", "tokenizer", "model_config", "tensors", "provenance"):
        if key not in m:
            raise ManifestError(f"manifest missing section {key!r}")
    comp = m["compression"]
    for k, v in COMPRESSION_CONTRACT.items():
        if comp.get(k) != v:
            raise ManifestError(f"manifest compression.{k} = {comp.get(k)!r} is incompatible with codec ({v!r})")
    return m


def verify_against_checkpoint(m: dict, ckpt_dir: str | Path, *, verify_hashes: bool = True) -> None:
    d = Path(ckpt_dir)
    raw_cfg = json.loads((d / "config.json").read_text())
    if raw_cfg != m["model_config"]:
        raise ManifestError("config.json differs from manifest.model_config")
    inv = read_inventory(d)
    if len(inv) != m["tensors"]["count"] or inventory_digest(inv) != m["tensors"]["inventory_sha256"]:
        raise ManifestError("tensor inventory differs from manifest (names/dtypes/shapes)")
    for name, want in m["source"]["weight_files"].items():
        f = d / name
        if not f.exists():
            raise ManifestError(f"weight file missing: {name}")
        if f.stat().st_size != want["bytes"]:
            raise ManifestError(f"{name}: size {f.stat().st_size} != manifest {want['bytes']}")
        if verify_hashes and want.get("sha256") and sha256_file(f) != want["sha256"]:
            raise ManifestError(f"{name}: sha256 mismatch")
    if verify_hashes:
        for group in ("tokenizer",):
            for name, digest in m[group]["files"].items():
                if sha256_file(d / name) != digest:
                    raise ManifestError(f"{name}: sha256 mismatch")
        for name, digest in m["source"].get("config_files", {}).items():
            if sha256_file(d / name) != digest:
                raise ManifestError(f"{name}: sha256 mismatch")

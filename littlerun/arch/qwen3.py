"""Qwen3 architecture adapter (dense decoder, GQA, per-head q/k RMSNorm, RoPE, SwiGLU).

Matches transformers' Qwen3ForCausalLM numerics for the features declared supported.
Unsupported config features (rope_scaling, sliding window, non-SiLU activations)
fail closed in validate_config().
The adapter knows nothing about packed bit layouts: it receives ready nn.Modules
for every linear projection from the loader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import CompatibilityError, ManifestError

ARCH_ID = "qwen3"
ADAPTER_REVISION = "1"
PROJ_NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dt)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class KVCache:
    """Preallocated single-sequence KV cache: [layers][B=1, kv_heads, max_ctx, head_dim]."""

    def __init__(self, cfg: Qwen3Config, max_ctx: int, dtype, device):
        shape = (1, cfg.num_key_value_heads, max_ctx, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_hidden_layers)]
        self.max_ctx = max_ctx
        self.length = 0

    def reset(self):
        self.length = 0

    @staticmethod
    def bytes_for(cfg: Qwen3Config, max_ctx: int, dtype) -> int:
        el = torch.empty((), dtype=dtype).element_size()
        return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * max_ctx * el


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, q, k, v, o, q_norm, k_norm):
        super().__init__()
        self.cfg = cfg
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        self.q_norm, self.k_norm = q_norm, k_norm

    def forward(self, x, cos, sin, cache_k, cache_v, start: int):
        B, T, _ = x.shape
        D = self.cfg.head_dim
        q = self.q_norm(self.q_proj(x).view(B, T, -1, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, D)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, D).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if cache_k is not None:
            cache_k[:, :, start:start + T] = k
            cache_v[:, :, start:start + T] = v
            k, v = cache_k[:, :, :start + T], cache_v[:, :, :start + T]
        L = k.shape[2]
        if T == 1:
            mask, causal = None, False
        elif L == T:
            mask, causal = None, True
        else:
            mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(diagonal=L - T)
            causal = False
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, attn, mlp, ln1, ln2):
        super().__init__()
        self.self_attn, self.mlp = attn, mlp
        self.input_layernorm, self.post_attention_layernorm = ln1, ln2

    def forward(self, x, cos, sin, ck, cv, start):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, ck, cv, start)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config, embed: torch.Tensor, layers: list[DecoderLayer], norm: RMSNorm,
                 lm_head: torch.Tensor, dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.embed_tokens = nn.Embedding.from_pretrained(embed, freeze=True)
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head_weight = nn.Parameter(lm_head, requires_grad=False)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def rope(self, start: int, T: int, device):
        pos = torch.arange(start, start + T, device=device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq.to(device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, cache: KVCache | None = None, logits: str = "last") -> torch.Tensor:
        """ids: [1, T]. Appends to `cache` when given. Returns float32 logits [T or 1, vocab]."""
        B, T = ids.shape
        start = cache.length if cache is not None else 0
        if cache is not None and start + T > cache.max_ctx:
            raise ValueError(f"context overflow: {start + T} > {cache.max_ctx}")
        cos, sin = self.rope(start, T, ids.device)
        h = self.embed_tokens(ids)
        for i, layer in enumerate(self.layers):
            ck = cache.k[i] if cache is not None else None
            cv = cache.v[i] if cache is not None else None
            h = layer(h, cos, sin, ck, cv, start)
        if cache is not None:
            cache.length = start + T
        h = self.norm(h)
        if logits == "hidden":  # final-normed hidden states [T, hidden]; caller applies lm_head
            return h.reshape(-1, self.cfg.hidden_size)
        if logits == "last":
            h = h[:, -1:]
        return (h @ self.lm_head_weight.t()).float().reshape(-1, self.cfg.vocab_size)


class Qwen3Adapter:
    arch_id = ARCH_ID
    revision = ADAPTER_REVISION

    # ------------------------------------------------------------------ config
    def validate_config(self, raw: dict) -> Qwen3Config:
        if raw.get("model_type") != "qwen3":
            raise CompatibilityError(f"model_type {raw.get('model_type')!r} is not qwen3")
        if raw.get("rope_scaling") not in (None, {}):
            raise CompatibilityError("rope_scaling is not supported by qwen3 adapter r1")
        if raw.get("use_sliding_window"):
            raise CompatibilityError("sliding-window attention is not supported by qwen3 adapter r1")
        if raw.get("hidden_act", "silu") != "silu":
            raise CompatibilityError(f"hidden_act {raw.get('hidden_act')!r} unsupported")
        try:
            h, nh = raw["hidden_size"], raw["num_attention_heads"]
            cfg = Qwen3Config(
                hidden_size=h,
                intermediate_size=raw["intermediate_size"],
                num_hidden_layers=raw["num_hidden_layers"],
                num_attention_heads=nh,
                num_key_value_heads=raw.get("num_key_value_heads", nh),
                head_dim=raw.get("head_dim") or h // nh,
                rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
                rope_theta=float(raw.get("rope_theta", 10000.0)),
                vocab_size=raw["vocab_size"],
                max_position_embeddings=raw.get("max_position_embeddings", 32768),
                tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
                attention_bias=bool(raw.get("attention_bias", False)),
            )
        except KeyError as e:
            raise ManifestError(f"config.json missing required field {e}") from None
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise CompatibilityError("num_attention_heads must be a multiple of num_key_value_heads")
        return cfg

    # ------------------------------------------------------------------ tensor contract
    def linear_shapes(self, cfg: Qwen3Config) -> dict[str, tuple[int, int]]:
        """prefix -> (in_features, out_features) for every projection."""
        h, i, d = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
        q, kv = cfg.num_attention_heads * d, cfg.num_key_value_heads * d
        per = {"self_attn.q_proj": (h, q), "self_attn.k_proj": (h, kv), "self_attn.v_proj": (h, kv),
               "self_attn.o_proj": (q, h), "mlp.gate_proj": (h, i), "mlp.up_proj": (h, i),
               "mlp.down_proj": (i, h)}
        return {f"model.layers.{l}.{p}": s for l in range(cfg.num_hidden_layers) for p, s in per.items()}

    def dense_tensors(self, cfg: Qwen3Config) -> dict[str, tuple[int, ...]]:
        out = {"model.embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size), "model.norm.weight": (cfg.hidden_size,)}
        if not cfg.tie_word_embeddings:
            out["lm_head.weight"] = (cfg.vocab_size, cfg.hidden_size)
        for l in range(cfg.num_hidden_layers):
            p = f"model.layers.{l}"
            out[f"{p}.input_layernorm.weight"] = (cfg.hidden_size,)
            out[f"{p}.post_attention_layernorm.weight"] = (cfg.hidden_size,)
            out[f"{p}.self_attn.q_norm.weight"] = (cfg.head_dim,)
            out[f"{p}.self_attn.k_norm.weight"] = (cfg.head_dim,)
        return out

    # ------------------------------------------------------------------ build
    def build_model(self, cfg: Qwen3Config, linears: dict[str, nn.Module], dense: dict[str, torch.Tensor],
                    dtype) -> Qwen3Model:
        eps = cfg.rms_norm_eps
        layers = []
        for l in range(cfg.num_hidden_layers):
            p = f"model.layers.{l}"
            g = lambda n: linears[f"{p}.{n}"]  # noqa: E731
            attn = Attention(cfg, g("self_attn.q_proj"), g("self_attn.k_proj"), g("self_attn.v_proj"),
                             g("self_attn.o_proj"), RMSNorm(dense[f"{p}.self_attn.q_norm.weight"].to(dtype), eps),
                             RMSNorm(dense[f"{p}.self_attn.k_norm.weight"].to(dtype), eps))
            mlp = MLP(g("mlp.gate_proj"), g("mlp.up_proj"), g("mlp.down_proj"))
            layers.append(DecoderLayer(attn, mlp, RMSNorm(dense[f"{p}.input_layernorm.weight"].to(dtype), eps),
                                       RMSNorm(dense[f"{p}.post_attention_layernorm.weight"].to(dtype), eps)))
        embed = dense["model.embed_tokens.weight"].to(dtype)
        head = embed if cfg.tie_word_embeddings else dense["lm_head.weight"].to(dtype)
        return Qwen3Model(cfg, embed, layers, RMSNorm(dense["model.norm.weight"].to(dtype), eps), head, dtype)

    def create_kv_cache(self, model: Qwen3Model, max_ctx: int, device) -> KVCache:
        return KVCache(model.cfg, max_ctx, model.dtype, device)

    def prefill(self, model: Qwen3Model, ids: list[int], cache: KVCache) -> torch.Tensor:
        device = model.lm_head_weight.device
        return model(torch.tensor([ids], device=device), cache, logits="last")[-1]

    def decode(self, model: Qwen3Model, token: int, cache: KVCache) -> torch.Tensor:
        device = model.lm_head_weight.device
        return model(torch.tensor([[token]], device=device), cache, logits="last")[-1]

    # ------------------------------------------------------------------ tokenizer
    def load_tokenizer(self, ckpt_dir: str | Path):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)

    def chat_template_sha256(self, ckpt_dir: str | Path) -> str | None:
        p = Path(ckpt_dir) / "tokenizer_config.json"
        if not p.exists():
            return None
        tpl = json.loads(p.read_text()).get("chat_template")
        return hashlib.sha256(tpl.encode()).hexdigest() if tpl else None

    def apply_chat_template(self, tokenizer, messages: list[dict], enable_thinking: bool) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=enable_thinking)

    def stop_token_ids(self, tokenizer, generation_config: dict) -> set[int]:
        ids = set()
        eos = generation_config.get("eos_token_id")
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, list):
            ids.update(eos)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            ids.add(tokenizer.eos_token_id)
        return ids

"""Tiny synthetic LittleBit-format Qwen3 checkpoints for tests (packed exactly like the reference)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..codecs.littlebit import pack_signs

CHAT_TEMPLATE = (
    "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


def tiny_config(vocab_size: int, *, layers=2, hidden=64, inter=72, heads=4, kv_heads=2, head_dim=16,
                tie=False, rope_theta=10000.0) -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False,
        "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 1, "head_dim": head_dim,
        "hidden_act": "silu", "hidden_size": hidden, "initializer_range": 0.02, "intermediate_size": inter,
        "max_position_embeddings": 512, "max_window_layers": layers, "num_attention_heads": heads,
        "num_hidden_layers": layers, "num_key_value_heads": kv_heads, "rms_norm_eps": 1e-6,
        "rope_scaling": None, "rope_theta": rope_theta, "sliding_window": None, "tie_word_embeddings": tie,
        "torch_dtype": "float32", "use_cache": True, "use_sliding_window": False, "vocab_size": vocab_size,
        "quant_func": "SmoothSign", "quant_mod": "LittleBitLinear", "residual": True, "eff_bit": 0.5,
        "split_dim": 1024, "kv_factor": 1.0, "min_split_dim": 8, "use_itq": True, "itq_n_iter": 50,
    }


def write_tokenizer(out: Path) -> int:
    """Byte-level BPE tokenizer with Qwen-style specials. Returns vocab size."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["hello world, the quick brown fox jumps over the lazy dog. 0123456789 user assistant system"] * 20
    tok.train_from_iterator(corpus, trainer)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
        "bos_token": None, "unk_token": None, "chat_template": CHAT_TEMPLATE,
        "model_max_length": 512,
    }))
    return tok.get_vocab_size()


def _rand_scale(g, n):
    return (torch.rand(1, n, generator=g) * 0.9 + 0.1)


def write_tiny_checkpoint(out: str | Path, *, seed=0, layers=2, hidden=64, inter=72, heads=4, kv_heads=2,
                          head_dim=16, ranks=(20, 12), residual=True, dtype=torch.bfloat16, tie=False,
                          extra_tensor: str | None = None) -> dict:
    """Write config, tokenizer and a packed model.safetensors. Returns dense reference weights (fp64)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = write_tokenizer(out)
    cfg = tiny_config(vocab, layers=layers, hidden=hidden, inter=inter, heads=heads, kv_heads=kv_heads,
                      head_dim=head_dim, tie=tie)
    cfg["residual"] = residual
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "littlebit_config.json").write_text(json.dumps(
        {k: cfg[k] for k in ("quant_func", "eff_bit", "split_dim", "residual", "kv_factor", "min_split_dim",
                             "quant_mod", "use_itq", "itq_n_iter")}))
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 0]}))

    g = torch.Generator().manual_seed(seed)
    q, kv = heads * head_dim, kv_heads * head_dim
    shapes = {"self_attn.q_proj": (hidden, q), "self_attn.k_proj": (hidden, kv), "self_attn.v_proj": (hidden, kv),
              "self_attn.o_proj": (q, hidden), "mlp.gate_proj": (hidden, inter), "mlp.up_proj": (hidden, inter),
              "mlp.down_proj": (inter, hidden)}
    sd: dict[str, torch.Tensor] = {}
    dense_ref: dict[str, torch.Tensor] = {}

    for l in range(layers):
        for name, (m, n) in shapes.items():
            p = f"model.layers.{l}.{name}"
            W = torch.zeros(n, m, dtype=torch.float64)
            for i, suffix in enumerate(["", "_R"][: 2 if residual else 1]):
                s = ranks[i]
                U = torch.randn(n, s, generator=g)
                V = torch.randn(s, m, generator=g)
                U[0, 0] = 0.0  # exercise sign(0) = +1
                u1, u2, v1, v2 = (_rand_scale(g, k).to(dtype) for k in (n, s, s, m))
                Us = torch.where(U < 0, -1.0, 1.0)
                Vs = torch.where(V < 0, -1.0, 1.0)
                sd[f"{p}.U{suffix}_packed"] = pack_signs(Us)
                sd[f"{p}.U{suffix}_shape"] = torch.tensor([n, s], dtype=torch.long)
                sd[f"{p}.V{suffix}_packed"] = pack_signs(Vs)
                sd[f"{p}.V{suffix}_shape"] = torch.tensor([s, m], dtype=torch.long)
                for k, t in (("u1", u1), ("u2", u2), ("v1", v1), ("v2", v2)):
                    sd[f"{p}.{k}{suffix}"] = t
                d = lambda t: t.double().reshape(-1)  # noqa: E731
                W += (d(u1)[:, None] * Us.double() * d(u2)[None, :]) @ (d(v1)[:, None] * Vs.double() * d(v2)[None, :])
            sd[f"{p}._split_dim_final"] = torch.tensor(ranks[0])
            sd[f"{p}._eff_bit_actual"] = torch.tensor(0.5, dtype=dtype)
            sd[f"{p}._eff_bit_target"] = torch.tensor(0.5, dtype=dtype)
            dense_ref[f"{p}.weight"] = W
        for nm, size in (("input_layernorm", hidden), ("post_attention_layernorm", hidden),
                         ("self_attn.q_norm", head_dim), ("self_attn.k_norm", head_dim)):
            t = (1.0 + 0.1 * torch.randn(size, generator=g)).to(dtype)
            sd[f"model.layers.{l}.{nm}.weight"] = t
            dense_ref[f"model.layers.{l}.{nm}.weight"] = t.double()
    for nm, t in (("model.embed_tokens.weight", torch.randn(vocab, hidden, generator=g) * 0.5),
                  ("model.norm.weight", 1.0 + 0.1 * torch.randn(hidden, generator=g)),
                  ("lm_head.weight", torch.randn(vocab, hidden, generator=g) * 0.1)):
        if nm == "lm_head.weight" and tie:
            continue
        sd[nm] = t.to(dtype)
        dense_ref[nm] = sd[nm].double()
    if extra_tensor:
        sd[extra_tensor] = torch.zeros(1, dtype=dtype)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    return {"config": cfg, "dense": dense_ref}


def build_fixture_with_manifest(out: str | Path, **kw) -> dict:
    from ..manifest import MANIFEST_NAME, build_manifest

    ref = write_tiny_checkpoint(out, **kw)
    m = build_manifest(out, source_repository="local/tiny-littlebit-qwen3", source_revision="fixture",
                       license_id="test-only")
    (Path(out) / MANIFEST_NAME).write_text(json.dumps(m))
    ref["manifest"] = m
    return ref

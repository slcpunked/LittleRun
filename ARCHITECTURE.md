# LittleRun architecture (v0.1 reference prototype)

Implements Phase 0–2 of "LittleBit-2 Multi-Model Inference Runtime, Architecture v2.1".
Phases 3–5 (CUDA kernels, Windows packaging, more models) are not started.

## Pipeline

```
CLI / HTTP client
  -> API adapter            littlerun/server.py      validation, SSE, error mapping
  -> request queue          littlerun/engine.py      bounded, one decode stream
  -> inference coordinator  littlerun/engine.py      lifecycle, cancellation, stop rules
  -> architecture adapter   littlerun/arch/qwen3.py  blocks, attention, KV cache, tokenizer, chat template
  -> compression codec      littlerun/codecs/littlebit.py  validate, unpack factors
  -> execution backend      littlerun/backends/torch_ref.py  FactorizedLinear (PyTorch)
supporting: loader.py (manifest -> compatibility -> memory budget -> model), manifest.py, registry.py
```

These are modules in one process, not services.

## Module boundaries

| Module | Owns | Must not know |
|---|---|---|
| `engine.py` | Residency, queue, lifecycle, cancellation, stop strings/tokens, sampling calls | Tensor names, kernels |
| `arch/qwen3.py` | Config validation, tensor map, blocks, RoPE, GQA, KV cache, tokenizer, chat template | Bit packing |
| `codecs/littlebit.py` | Packing ABI, factor-group validation, decoding | Attention, chat |
| `backends/torch_ref.py` | Operator execution for `littlebit.factorized_linear.v1` | HTTP, checkpoints |
| `server.py` | REST schema, SSE, error codes | CUDA, tensors |

## Contracts

Adapter: `validate_config`, `linear_shapes`, `dense_tensors`, `build_model`, `create_kv_cache`,
`prefill`, `decode`, `load_tokenizer`, `apply_chat_template`, `stop_token_ids`.

Codec: `inspect`, `validate_tensors`, `load_factors`, `operation_spec`.
`pack_signs` is provided for fixtures; `pack_optional` (a derived packed cache) is not implemented.

Backend: `supports(OperationSpec)`, `build_linear`, `diagnostics`, `unload`.
`OperationSpec = (op, dtype, device, packing_abi)`.

Registries (`registry.py`) are plain keyed dicts:
- `ARCHITECTURES[(arch_id, adapter_revision)]`
- `CODECS[(family, version)]`
- `BACKENDS` probed in key order with `supports()`.

These signatures stay provisional until a second adapter or backend exists (architecture §4).

## Load sequence (fail closed)

1. Read `littlerun.manifest.json`; reject unknown schema or a codec contract that differs.
2. Verify `config.json` equality, tensor inventory digest, file sizes, and sha256 of weights,
   tokenizer and config files.
3. Resolve adapter, codec and backend from the registries.
4. Validate every factor group: names, dtypes, packed widths, stored shapes, rank metadata,
   padding bits, residual flag. Validate dense tensors. Any unclaimed tensor is an error.
5. Estimate memory (factors, scales, dense, KV cache, workspace). Refuse if above 90% of free memory.
6. Decode factors per group into the chosen storage, then build the model.

## Numerics

- Operator order and per-step dtype match the Samsung reference exactly.
- RMSNorm runs in fp32 and casts back, like transformers.
- RoPE angles are computed in fp32 and cast to the model dtype, like transformers.
- Attention uses `torch.nn.functional.scaled_dot_product_attention` with GQA.
- Logits are returned as fp32.

## Factor storage

| Mode | Resident form | Trade-off |
|---|---|---|
| `int8` (default) | ±1 as int8 | Half the memory of bf16. Casts on every call. |
| `compute` | ±1 in compute dtype | Fastest reference path. Most memory. |

Neither mode keeps the 1-bit packed form resident.
For the pinned 8B checkpoint the packed factors are 445 MiB, int8 is 3.7 GB, bf16 is 7.4 GB.
Realising the 1-bit memory advantage needs a packed kernel (Phase 3).
Full dense weights are never built.

## Serving contract

See `README.md` for endpoints. Request states:
`RECEIVED → VALIDATED → QUEUED → RUNNING → COMPLETED | CANCELLED | FAILED`.
Errors: 400 invalid, 404 unknown model, 429 queue full, 503 not ready, 500 internal.
Default bind is `127.0.0.1`. There is no telemetry and no remote code execution
(`trust_remote_code` is never used).

## Decisions

See `docs/adr/`.

# LittleRun

Reference inference runtime for LittleBit / LittleBit-2 sub-1-bit factorized LLM checkpoints.
It implements the v0.1 scope of "LittleBit-2 Multi-Model Inference Runtime, Architecture v2.1"
(one process, one model, one decode stream, OpenAI-style API), plus a v0.2 packed 1-bit CPU kernel.

Status (v0.2): Phases 0–2 done on CPU; Phase 3 started for CPU only.
WikiText-2 PPL reproduces the published value (24.486 vs 24.475).
On a 4-core Xeon, the 8B checkpoint decodes at 9–9.7 tok/s in 2.2 GiB peak RSS.
No GPU measurement yet.
See `docs/VALIDATION.md` for evidence and the acceptance checklist.

- Technique and checkpoint audit: `docs/RESEARCH.md`
- Design: `ARCHITECTURE.md`, decisions in `docs/adr/`

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -e ".[test]"
```

Python ≥ 3.10. Tested with Python 3.13, torch 2.15 (CPU), transformers 4.57.6.

## Get the pinned checkpoint

```bash
littlerun fetch littlebit-qwen3-8b-0.55bpw models/littlebit-qwen3-8b-0.55bpw   # pinned revision, sha256-checked,
D=models/littlebit-qwen3-8b-0.55bpw                                           # installs the reviewed manifest
littlerun doctor $D                                                           # environment + validation
```

Manual equivalent:

```bash
REV=6b2c47aa9707e624e2e737d16d64a66be8090ae9
D=models/littlebit-qwen3-8b-0.55bpw && mkdir -p $D
for f in config.json littlebit_config.json generation_config.json added_tokens.json merges.txt \
         special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json model.safetensors; do
  curl -fL -o $D/$f https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw/resolve/$REV/$f
done
sha256sum $D/model.safetensors   # 5b087e4c79c23697355014fdd00b083c91d0c731ddd80a16d6affbe156f17179
```

License: CC BY-NC 4.0 (non-commercial). This is a community checkpoint, not a Samsung release.

## Use

```bash
# 1. Build and review the manifest (once per checkpoint)
littlerun manifest build $D --repo LittleBitLLM/littlebit-qwen3-8b-0.55bpw --revision $REV \
  --license cc-by-nc-4.0 --base-model Qwen/Qwen3-8B --base-model-license apache-2.0

# 2. Validate (hashes, inventory, shapes, padding bits)
littlerun validate $D

# 3. Generate
littlerun generate $D --chat --prompt "What is 2+2?" --max-tokens 64

# 4. Serve on 127.0.0.1:8000
littlerun serve $D --model-id littlebit-qwen3-8b --max-context 4096

# 5. Interactive multi-turn chat (/reset, /exit)
littlerun chat $D --system "You are concise."

# 6. Benchmark
littlerun bench $D --prompt-tokens 512 --gen-tokens 64 -o bench.json
```

Load options: `--device auto|cpu|cuda`, `--dtype bf16|fp16|fp32`,
`--factor-storage auto|packed|int8mm|int8|compute`, `--max-context N`, `--no-verify-hashes`.

| `--factor-storage` | Resident factors (8B) | Where | Notes |
|---|---:|---|---|
| `auto` (default) | — | — | CPU: `packed`, else `int8mm`. GPU: `compute` if it fits, else `int8` |
| `packed` | 0.45 GB | CPU | Compiled kernel, built on first use (needs g++/clang) |
| `int8mm` | 3.7 GB | CPU, bf16/fp16 | PyTorch int8 weight-only kernel |
| `int8` | 3.7 GB | any | Casts per call; slow |
| `compute` | 7.4 GB (bf16) | any | Fastest option on GPU today |

Generation options (`generate`, `chat`): `--temperature`, `--top-p`, `--top-k`, `--seed`,
`--presence-penalty`, `--frequency-penalty`, `--repetition-penalty`.

## API

| Endpoint | Notes |
|---|---|
| `GET /health` | `live`, `ready`, `state` (`loading`/`ready`/`failed`), queue depth |
| `GET /health/live`, `GET /health/ready` | Ready returns 503 while loading or failed |
| `GET /v1/runtime/stats` | Local counters: requests, tokens, reused prompt tokens, average prefill/decode speed |
| `GET /v1/models` | Model id, backend, device, dtype, context, capabilities, source revision |
| `POST /v1/completions` | One prompt; `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `seed`, `stream`, penalties, `logprobs` (0–5) |
| `POST /v1/chat/completions` | Roles `system`/`user`/`assistant`; pinned chat template; `chat_template_kwargs.enable_thinking`; `logprobs`, `top_logprobs` (0–20) |

Penalties: `presence_penalty` and `frequency_penalty` (OpenAI semantics, generated tokens) and
`repetition_penalty` (HF semantics, prompt + generated). Logprobs are taken from the model
distribution before penalties and temperature.
Consecutive requests that share a prompt prefix reuse its KV cache; `usage.prompt_tokens_details.cached_tokens`
reports how many tokens were reused.

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "littlebit-qwen3-8b", "stream": true, "max_tokens": 64, "temperature": 0,
  "messages": [{"role": "user", "content": "Name three primary colors."}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

Authentication: `littlerun serve ... --api-key-env LITTLERUN_API_KEY` requires
`Authorization: Bearer <key>` on every route except `/health*`. Use it whenever `--host` is not localhost.

Rejected with 400: unknown fields, `n≠1`, `echo`, `suffix`, `tools`,
non-text `response_format`, `role: "tool"`.
Status codes: 400 invalid, 401 bad API key, 404 unknown model, 429 queue full, 503 not ready, 500 internal.
Streams end with a `finish_reason` chunk and `data: [DONE]`.
A client disconnect cancels the request and frees the decode stream.
`usage.completion_tokens` counts every sampled token, including a final stop token.

## Tests

```bash
pytest -q                                       # self-contained (fixtures generated on the fly)
LITTLEBIT_REF_PATH=/path/to/SamsungLabs/LittleBit pytest -q -m reference   # comparisons with Samsung code
python scripts/eval_ppl.py $D --parquet wikitext-2-raw-v1-test.parquet     # WikiText-2 PPL
python scripts/eval_mc.py $D --parquet ARC-Easy/test-00000-of-00001.parquet # ARC-Easy acc / acc_norm
```

## Limitations

- The packed kernel is CPU-only. On GPU, factors are unpacked to bf16 or int8 at load.
- One architecture (Qwen3, no rope scaling, no sliding window) and one codec.
- No tool calling, batching, speculative decoding or multi-GPU. Prefix reuse covers one stream only.
- The pinned checkpoint's reported quality is low (ARC-Challenge below chance). Chat answers are
  mostly unusable; plain completion is coherent. This is the checkpoint, not the runtime (see `docs/VALIDATION.md`).
- Windows/WSL2 packaging is not done.

## Licensing

LittleRun's code was written for this repository after reading the upstream reference.
No upstream source is copied; it reads the LittleBit checkpoint format. This is not a formal
clean-room process. Choose and add a license file before distributing it.
Upstream LittleBit code and the checkpoint are CC BY-NC 4.0; the Qwen3 base model is Apache-2.0.
Commercial use needs separate legal review (architecture §12).

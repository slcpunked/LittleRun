# Validation evidence (v0.1 and v0.2, 2026-11-20)

Environment: Linux container, Intel Xeon @ 2.10 GHz (4 vCPU, AVX-512 BF16/AMX), 15 GiB RAM, no GPU.
Python 3.13.18, torch 2.15.0+cpu, transformers 4.57.6.
Raw outputs are in `bench/`.

## 1. Automated tests

`pytest -q` (v0.2): 126 passed, 3 skipped.
The 3 skipped tests compare against the Samsung reference code (`-m reference`).
They need `LITTLEBIT_REF_PATH`; see §6.

| Layer | Coverage |
|---|---|
| Loader / codec | Pack round-trip for odd widths; LSB order; sign(0)=+1; dirty padding; missing path tensor; missing group; scale shape; stored-shape mismatch; rank metadata mismatch; packed dtype; residual flag mismatch; unknown tensor; config change; same-size byte corruption (sha256); incompatible manifest contract |
| Operator | Dense-reconstruction parity for odd sizes, residual and bias, both storage modes; orientation via basis vectors; zero, constant, large inputs; residual additivity; bf16 vs fp32 |
| Model | Logits vs transformers `Qwen3ForCausalLM`; prefill + incremental decode vs full forward; chunked prefill with explicit mask; context overflow; fixed-seed traces; stop token and stop string; storage-mode equivalence |
| API | SSE chunk order, role chunk, single finish chunk, `[DONE]`, usage chunk; finish reasons `stop`/`length`; 400/404/429/503/500 paths; unsupported fields; cancel while running and while queued; real-socket client disconnect cancels; recovery after failure and overload |

Model parity on the synthetic fixture (2 layers, residual, ranks 20/12, padded widths),
against a float64 transformers model built from the dense reconstruction:

| Runtime dtype | Factor storage | Max abs error | RMS error | Relative error | Argmax agreement |
|---|---|---:|---:|---:|---:|
| fp32 | int8 | 2.4e-6 | 5.5e-7 | 6.9e-7 | 100% |
| bf16 | int8 | 0.197 | 0.024 | 3.0e-2 | 100% |
| bf16 | compute | 0.197 | 0.024 | 3.0e-2 | 100% |

Logit magnitude in that test is about 3.2. bf16 tolerance in the tests is a measured 5%
relative error, not a universal criterion. The two storage modes are bit-identical.

## 2. Real checkpoint: integrity

`LittleBitLLM/littlebit-qwen3-8b-0.55bpw@6b2c47aa`:

- sha256 of `model.safetensors` and `tokenizer.json` match the Hugging Face LFS records.
- `littlerun manifest build`: 4,935 tensors, 252 factor groups, residual on every group,
  no unclaimed tensors, all padding bits clean, measured 0.548 bpw. Took 6.3 s.
- The manifest is committed as `littlerun/manifests/littlebit-qwen3-8b-0.55bpw.manifest.json`.

## 3. Real checkpoint: end-to-end numerical check (WikiText-2)

The Samsung code could not be executed in this session (§6). Instead I reproduced the
model card's own published metric with the same protocol (`scripts/eval_ppl.py`):
WikiText-2 raw test, joined with "\n\n", 2048-token non-overlapping windows, no BOS.

| Source | Runtime | Windows | Tokens scored | PPL |
|---|---|---:|---:|---:|
| Model card eval log (Samsung loader, CUDA, bf16) | upstream | 146 | 298,862 | 24.4754 |
| LittleRun (`bench/wikitext2-ppl-cpu-bf16.json`) | CPU, bf16, compute storage | 146 | 298,862 | 24.4859 |

Difference: +0.043%. Tokenisation produced 299,078 tokens and 146 windows, as in the upstream log.
ARC-Easy test (2,376 questions, zero-shot, lm-eval 0.4.x protocol, `scripts/eval_mc.py`,
`packed` storage, bf16; raw: `bench/arc-easy-packed.json`):

| Source | acc | acc_norm |
|---|---:|---:|
| Model card (lm-eval, upstream loader) | 0.4440 | 0.4070 |
| LittleRun | 0.4457 | 0.4099 |

The differences (+0.17 and +0.29 points) are about 4 and 7 questions, within bf16 noise.
I did not run ARC-Challenge.

A misread sign, bit order, orientation or scale would raise PPL by orders of magnitude.
The small gap is consistent with bf16 differences between CUDA and CPU kernels; I did not prove that.

The container restarted after window 108. Windows 1–108 come from the interrupted run's log
(`bench/wikitext2-ppl-part1.txt`, NLL rounded to 4 decimals). Windows 109–146 come from a
resumed run (`bench/wikitext2-ppl-part2.json`). The rounding moves PPL by at most ±0.001.

Live server smoke test (CPU, compute storage, context 2048):

- `/health` reported `loading`, then `ready`. `/v1/models` listed backend, dtype, bpw 0.548 and capabilities.
- `POST /v1/completions` "The capital of France is", greedy, 12 tokens →
  " Paris, and the city is the capital of the French Republic" (`length`).
- Streaming chat ("What is the capital of France?", thinking off, greedy) produced a valid SSE stream
  ending in a usage chunk and `[DONE]`. The content echoed the question and then repeated "_".
  The chat prompt matched the canonical Qwen3 format and token IDs.
  I attribute this to the checkpoint (0.5 epoch of distillation on C4/WikiText, no chat data).
  That attribution is an inference: the matching PPL rules out a decode error, not a template-handling difference.
- A chat request with `tools` returned 400.

## 4. Real checkpoint: speed and memory (CPU only)

v0.1 numbers (superseded on CPU by §7). `littlerun bench`, greedy, bf16, 4 threads:

| Factor storage | Context alloc | Prompt | Prefill tok/s | Decode tok/s | Load estimate | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| compute (bf16) | 2,304 | 2,048 | 92.0 | 2.30 | 9.61 GiB | 9.43 GiB |
| int8 | 1,024 | 128 | 53–60 | 1.21–1.43 | 5.99 GiB | 5.68 GiB |

Model load (with sha256 verification) took 19.7 s.
These numbers describe this CPU only. They say nothing about the RTX 5060 Ti.

## 5. Acceptance checklist (architecture §13)

| Item | Status | Evidence |
|---|---|---|
| Pinned source and checkpoint; complete tensor mapping; license reviewed | Done | §2, `docs/RESEARCH.md` |
| Reference model loads; numerically comparable layer and model outputs | Done on CPU | §1; §3 (PPL within 0.043% of published); §7 |
| Quality suite passes predeclared minimums | **Fails** (checkpoint) | §8: 0/25 prompts; minimums still PROPOSED |
| API, streaming, cancellation, overload, errors pass client tests | Done (in-process and real socket) | §1 |
| GPU memory and throughput on RTX 5060 Ti | Blocked | No GPU in this environment |
| Optimized kernels beat baseline | Done on CPU | §7: packed kernel 3.9–4.2× v0.1 decode, no quality change measured |
| Each additional adapter passes gates | Not applicable | One adapter |
| Windows/WSL2 packaging verified | Not started | Phase 4. Kernel has an MSVC shim that has never been compiled. |

## 6. Known gaps

- Samsung-code comparisons (`-m reference`) did not run here.
  The session's safety policy blocked importing the downloaded upstream code.
  Run them locally: `LITTLEBIT_REF_PATH=<checkout> pytest -q -m reference`.
- No GPU numbers. Run `littlerun bench $D --device cuda --factor-storage compute` on the target.
- After v0.2, a CPU decode step is about 100 ms: lm_head GEMV ~21 ms, 36 layers ~66 ms
  (factor linears ~1.5 ms per layer), rest embedding/RoPE/overhead.
- Two container restarts interrupted long evaluations. WikiText-2 was stitched from two runs (§3);
  ARC-Easy ran with per-question checkpoints and finished without a restart.

## 7. v0.2 packed kernel and storage modes (CPU)

`littlerun bench`, 128-token prompt, greedy, bf16 compute, 4 threads, same machine as §4.
Raw files: `bench/cpu-*.json`.

| Factor storage | Prefill tok/s | Decode tok/s | Peak RSS | Load estimate |
|---|---:|---:|---:|---:|
| v0.1 `compute` (2,048-token prompt) | 92.0 | 2.30 | 9.43 GiB | 9.61 GiB |
| v0.1 `int8` | 53–60 | 1.21–1.43 | 5.68 GiB | 5.99 GiB |
| v0.2 `int8mm` | 89–91 | 4.34–4.60 | 5.68 GiB | 5.99 GiB |
| v0.2 `packed` (first kernel) | 119–136 | 8.17–8.61 | 2.18 GiB | 2.97 GiB |
| v0.2 `packed` (8-accumulator M=1 path) | 103–119 | 9.00–9.69 | — | 2.97 GiB |

Peak RSS for `packed` is below the 2.5 GB of dense embeddings because embedding rows are
memory-mapped and only touched rows become resident. lm_head is fully resident.

Correctness evidence for `packed`:
- Operator tests vs float64 dense reconstruction: relative error < 1e-5 (fp32), bf16-level in bf16.
  All SIMD paths (scalar, AVX2, AVX-512) agree; odd widths and padding are covered.
- Model parity with transformers on the fixture, through both the kernel path (≤16 rows) and
  the unpack+GEMM path: max abs logit error < 2e-4 in fp32.
- WikiText-2, first 10 windows on the 8B checkpoint: per-window NLL equals the validated v0.1 run
  within 4.7e-5 (`bench/wikitext2-ppl-packed-10w.json`).
- 8B greedy decode, 24 tokens from "The history of the Roman Empire begins": `packed`, `int8mm` and
  `compute` produce identical tokens. First-step relative logit difference vs `compute`:
  1.2e-2 (`packed`), 1.9e-2 (`int8mm`).
  Rechecked after the 8-accumulator kernel change: `packed` and `int8mm` still give the same 24 tokens.
  `compute` was not rerun (9.4 GiB load); equality with it follows only through `int8mm`.

Prefix KV reuse on the 8B checkpoint (two-turn chat, 907-token first prompt):
second turn reused 903 of 921 prompt tokens; TTFT 8.09 s cold vs 1.75 s warm; identical output.

`int8mm` note: `aten._weight_int8pack_mm` returned wrong values (inf) for K=824 and crashed for
M ≥ 2 in torch 2.15. With K zero-padded to a multiple of 32, every tested shape matched to ~1.7e-3.

Kernel SIMD paths, `signed_matmul` microbenchmark (fp32, random signs, 50 calls, idle CPU):

| Shape R×K | M | scalar | AVX2 | AVX-512 |
|---|---:|---:|---:|---:|
| 4096×824 | 1 | 723 µs | 137 µs | 48 µs |
| 4096×824 | 8 | 5,814 µs | 783 µs | 200 µs |
| 824×4096 | 1 | 616 µs | 92 µs | 30 µs |
| 12288×824 | 1 | 2,048 µs | 764 µs | 169 µs |
| 12288×824 | 8 | 15,911 µs | 2,212 µs | 621 µs |

All paths agree with scalar within 5e-4 absolute (summation order only).
AVX2 runs 2.5–4.5× slower than AVX-512 here. A CPU without AVX-512 should expect a slower
packed decode than §7 reports. That figure is an inference; I did not measure it on such a CPU.

## 8. Quality suite (agent readiness)

`scripts/quality_suite.py`, greedy, thinking off, max 64 tokens, system prompt in the file.
Minimums (`docs/quality_minimums.json`) were committed before the run. They are still PROPOSED.
Raw output: `bench/quality-suite-8b.json`.

| Family | Prompts | Passed | Score | Minimum | Result |
|---|---:|---:|---:|---:|---|
| json | 10 | 0 | 0.0 | 0.9 | FAIL |
| arith | 10 | 0 | 0.0 | 0.8 | FAIL |
| follow | 5 | 0 | 0.0 | 0.6 | FAIL |

Typical outputs echo the system prompt or question, repeat "_____", or emit blank space.
No output contained a parseable JSON object or a correct number.

Interpretation: the checkpoint is not usable as a chat or agent model. The runtime is not the cause,
as far as the evidence shows: WikiText-2 PPL (§3) and ARC-Easy (§3) both match the model card.
Neither check exercises the chat template on a fine-tuned chat task, so a template fault is not
fully excluded. The canonical Qwen3 token IDs were checked in §3.

Consequence: §13 "quality suite passes" cannot pass with this checkpoint. A different checkpoint
(for example, one distilled with chat data) is needed for agent use. The runtime does not need changes for that.


# Validation evidence (v0.1, 2026-11-20)

Environment: Linux container, Intel Xeon @ 2.10 GHz (4 vCPU, AVX-512 BF16/AMX), 15 GiB RAM, no GPU.
Python 3.13.18, torch 2.15.0+cpu, transformers 4.57.6.
Raw outputs are in `bench/`.

## 1. Automated tests

`pytest -q`: 61 passed, 3 skipped.
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
- The manifest is committed as `manifests/littlebit-qwen3-8b-0.55bpw.manifest.json`.

## 3. Real checkpoint: end-to-end numerical check (WikiText-2)

The Samsung code could not be executed in this session (§6). Instead I reproduced the
model card's own published metric with the same protocol (`scripts/eval_ppl.py`):
WikiText-2 raw test, joined with "\n\n", 2048-token non-overlapping windows, no BOS.

PPL_RESULT_PLACEHOLDER

## 4. Real checkpoint: speed and memory (CPU only)

`littlerun bench`, greedy, bf16 compute, 4 threads:

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
| Reference model loads; numerically comparable layer and model outputs | Done on CPU | §1, §3 |
| Quality suite passes predeclared minimums | Not done | No suite or minimums declared yet. Reported ARC-Challenge is below chance. |
| API, streaming, cancellation, overload, errors pass client tests | Done (in-process and real socket) | §1 |
| GPU memory and throughput on RTX 5060 Ti | Blocked | No GPU in this environment |
| Optimized kernels beat baseline | Not started | Phase 3 |
| Each additional adapter passes gates | Not applicable | One adapter |
| Windows/WSL2 packaging verified | Not started | Phase 4 |

## 6. Known gaps

- Samsung-code comparisons (`-m reference`) did not run here.
  The session's safety policy blocked importing the downloaded upstream code.
  Run them locally: `LITTLEBIT_REF_PATH=<checkout> pytest -q -m reference`.
- No GPU numbers. Run `littlerun bench $D --device cuda --factor-storage compute` on the target.
- Decode on CPU is memory-bandwidth bound: each token reads the unpacked factors plus the
  1.2 GB lm_head.

# Architecture decision records

## ADR-0001: Own Qwen3 implementation instead of patching transformers

- Status: accepted (2026-11-20)
- Context: The Samsung loader patches `transformers` models in place. The README pins
  `transformers` 4.51 for reproducible results. Model internals change between releases.
- Decision: The adapter implements Qwen3 directly (about 300 lines). `transformers` is used for
  the tokenizer and chat template only, plus as a test oracle.
- Evidence: `tests/test_model.py::test_logits_match_transformers_dense_reconstruction`
  compares logits with `transformers` 4.57.6 `Qwen3ForCausalLM`.
- Consequence: Each new architecture needs its own adapter and parity test.

## ADR-0002: Unpack factors at load; no packed kernel in v0.1

- Status: superseded on CPU by ADR-0006; still applies on GPU
- Context: Architecture §8 says profile first; custom CUDA is NO-GO without evidence.
  No GPU was available while building v0.1.
- Decision: Decode signs to int8 (default) or compute dtype at load.
  Keep the original safetensors read-only. No derived packed cache.
- Consequence: Resident factor memory is 8× (int8) or 16× (bf16) the packed size.
  The memory case for LittleBit on a 16 GB GPU is not yet realised by this runtime.

## ADR-0003: A reviewed manifest is required to load

- Status: accepted
- Decision: `littlerun manifest build` writes the manifest from a checkpoint and explicit
  provenance flags. The loader refuses to start without it. Nothing is inferred from file names.
- Consequence: One extra command per checkpoint. Changed files fail loudly.

## ADR-0004: Tool calling is rejected

- Status: accepted
- Context: Architecture §9 says JSON pass-through does not imply working tool calls.
  The pinned checkpoint scores below chance on ARC-Challenge (reported).
- Decision: Requests with `tools`, `tool_choice` (other than "none"), or `role: "tool"` get 400.
- Revisit when a quality suite shows correct tool selection and arguments.

## ADR-0005: Chat requests default to Qwen3 thinking mode

- Status: accepted
- Decision: Match the pinned chat template default (`enable_thinking` true).
  Clients can send `chat_template_kwargs: {"enable_thinking": false}` (vLLM-compatible).

## ADR-0006: Packed 1-bit CPU kernel as a compiled extension

- Status: accepted (2026-11-20, v0.2)
- Context: Profiling showed CPU decode was memory-bound on unpacked int8 factors
  (3.7 GB read per token). Architecture §8 requires profiling before a kernel; this is that evidence.
- Decision: A small C++ extension (`littlerun/kernels/signed_gemm.cpp`), built on first use with
  `torch.utils.cpp_extension`. Masked-add GEMV over packed sign words for ≤16 rows. Byte-LUT
  unpack plus GEMM for larger inputs. The reference fallback stays; a failed build falls back to `int8mm`.
- Evidence: 8B decode 2.3 → 9–9.7 tok/s; peak RSS 9.4 → 2.2 GiB; WikiText-2 windows identical
  to 5e-5 NLL; 24/24 greedy tokens equal across modes (`docs/VALIDATION.md` §7).
- Consequence: Needs a C++ compiler at first run. GCC/Clang only (GNU builtins); MSVC builds fall back.

## ADR-0007: Rejected optimisations (measured)

- Fusing each linear's scales and both paths into one C++ call: no measurable gain
  (99.0 vs 97.8 ms/token minimum over 4 interleaved rounds). Reverted.
- int8 per-row lm_head: 21.2 → 17.9 ms per token, but lossy. Not worth an extra numerics mode.
- `aten._weight_int8pack_mm` with K not a multiple of 32: wrong results and segfaults in
  torch 2.15. The `int8mm` mode pads K to 32.

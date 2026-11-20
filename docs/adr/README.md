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

- Status: accepted
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

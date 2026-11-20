# LittleBit and LittleBit-2: technique and evidence audit

Phase 0 of the architecture plan ("evidence audit"). Every claim below names its source.
"Verified" means I read the code or the bytes myself on 2026-11-20.
"Reported" means a paper or model card says it and I did not reproduce it.

## 1. Sources read

| Source | What I used it for | Pin |
|---|---|---|
| SamsungLabs/LittleBit (GitHub) | Layer math, init, packing ABI, loader | commit `42d658b0c79f76450b34b6a3547462c7cdc3e1a0` (2026-10-09) |
| arXiv 2506.13771, LittleBit (NeurIPS 2025) | Method summary (abstract) | abstract page |
| arXiv 2603.00042, LittleBit-2 (ICML 2026) | Method summary (abstract) | abstract page |
| HF `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` | Checkpoint, model card, eval logs | revision `6b2c47aa9707e624e2e737d16d64a66be8090ae9` |
| HF `Qwen/Qwen3-8B` | Base model license | revision `b968826d9c46dd6066d109eabc6255188de91218` |
| HF `Salesforce/wikitext` | Perplexity data | revision `b08601e04326c79dfdd32d625aee71d232d685c3` |

I read the two arXiv abstracts, not the full papers.
Paper-only details (training budgets, tables) are therefore not restated here.

## 2. The technique

### 2.1 LittleBit (NeurIPS 2025)

LittleBit replaces each dense linear weight `W` (out × in) with binary low-rank factors plus scales.
The deployed weight is:

```
W ≈ diag(u1) · sign(U) · diag(u2 ⊙ v1) · sign(V) · diag(v2)
      out×out    out×r       r×r           r×in     in×in
```

- `sign(U)` and `sign(V)` are ±1 matrices, stored at 1 bit per entry.
- `u1` scales outputs, `v2` scales inputs, `u2 ⊙ v1` scales the latent rank dimension.
  The paper calls this "multi-scale compensation" (abstract).
- Dual-SVID initialisation (verified in `littlebit.py::_decompose_matrix`):
  1. Truncated SVD: `U = U_r·√S`, `V = √S·V_rᵀ`.
  2. Rank-one SVD of `|U|` gives `(u1, u2)`; rank-one SVD of `|V|` gives `(v1, v2)`.
- Residual compensation (verified): a second factor set `(U_R, V_R, …)` approximates
  `W − W_approx`. Both paths run at inference and are summed.
- Quantization-aware training uses `SmoothSign`: forward `sign(x)` with `sign(0)=+1`,
  backward `α·(1 − tanh²(αx))`, `α=100` (verified, `functions/binary.py`).

Inference forward (verified, `_compute_forward`):

```
y = ((((x ⊙ v2) @ sign(V)ᵀ) ⊙ (v1 ⊙ u2)) @ sign(U)ᵀ) ⊙ u1   [+ residual path] [+ bias]
```

This matches the architecture document's schematic `y = h ⊙ [B_U(ℓ ⊙ [B_V(g ⊙ x)])]`
with `h=u1`, `ℓ=v1⊙u2`, `g=v2`.

### 2.2 LittleBit-2 (ICML 2026)

Reported (abstract): low-rank binary approximations can beat tiny-rank float baselines for
heavy-tailed spectra ("Spectral Energy Gain"). Plain SVD factors are "spiky" (high coherence),
which is the worst case for sign quantization ("Latent Geometry Misalignment").

Fix, verified in code (`_compute_itq_rotation`, opt-in `use_itq=True`):

1. Stack `X = [U ; Vᵀ]` ((out+in) × r).
2. Joint Iterative Quantization: start from a random orthogonal `R` (r×r); repeat 50 times:
   `B = sign(X·R)`, then `R` = orthogonal Procrustes solution from `SVD(Bᵀ·X)`.
3. Replace `U ← U·R`, `V ← Rᵀ·V`. The product `U·V` is unchanged because `R·Rᵀ = I`.

Consequence for a runtime (verified): LittleBit-2 changes initialisation only.
The deployed graph, tensor names and packing are identical to LittleBit.
One runtime and one codec serve both. This confirms the assertion in architecture §2.

### 2.3 Bit budget and rank choice

The trainer picks the rank `s` from a bits-per-weight target (verified, `_estimate_split_dim`):

```
residual:     s = ⌊(a·b·bpw − 32(a+b)) / (2(a+b+16)) / 8⌋ · 8
no residual:  s = ⌊(a·b·bpw − 16(a+b)) / (a+b+16) / 8⌋ · 8
```

`tests/test_codec.py::test_rank_formula_matches_real_checkpoint_layout` checks this against
the real checkpoint's ranks (544, 208, 824).

### 2.4 Packing ABI (verified, `binary_packer.py`)

- int32 words, row-major, least-significant bit first.
- Bit 0 = +1, bit 1 = −1.
- Rows are padded to a multiple of 32 with +1, so padding bits are 0.
- Each packed tensor has a companion `*_shape` int64 tensor.
- Scales stay in the training dtype (BF16 in the pinned checkpoint).

## 3. The pinned checkpoint

Verified from the safetensors header and model card.

| Fact | Value |
|---|---|
| Publisher | `LittleBitLLM` (community). The card says: "Independent community release. Not an official Samsung Research model." |
| Base model | Qwen/Qwen3-8B (Apache-2.0 per its card) |
| License | CC BY-NC 4.0 (card), plus the base model's license |
| Method | LittleBit-2 (`use_itq=true`), SmoothSign, residual, `eff_bit=0.55` |
| Training | 0.5 epoch of distillation; the card says the paper uses 5 |
| File | `model.safetensors`, 2,968,804,064 bytes, sha256 `5b087e4c…f17179` (matches HF LFS) |
| Tensors | 4,935: 252 factor groups (36 layers × 7 projections), each with primary and residual paths |
| Ranks (both paths) | q/o 544, k/v 208, gate/up/down 824 |
| Measured bpw (linear layers, factors + scales) | 0.548 |
| Unclaimed tensors | none |

Byte breakdown of the 2.76 GiB file:

| Part | MiB | Share |
|---|---:|---:|
| Embeddings + lm_head (BF16, untied) | 2,374 | 83.9% |
| Packed sign factors | 445 | 15.7% |
| Scales | 11.5 | 0.4% |
| Norms | 0.6 | 0.0% |

The "0.55 bpw" label applies to linear layers only.
The whole file averages about 2.9 bits per original parameter.
Embeddings and the output head dominate the size.

Reported quality (model card, eval log `logs/eval-littlebit-qwen3-8b-0.55bpw-20261013-1212.log`):

| Metric | Qwen3-8B BF16 | This checkpoint |
|---|---:|---:|
| WikiText-2 PPL | 9.73 | 24.48 |
| C4 PPL | 13.30 | 32.10 |
| ARC-Easy | 83.5 | 44.4 |
| ARC-Challenge | 55.5 | 19.5 (below 25% chance) |
| PIQA | 76.5 | 62.6 |

The eval log loaded snapshot `dbedbdc4…`. I checked that its `model.safetensors`,
`config.json` and `tokenizer_config.json` hashes equal those at the pinned revision.

## 4. Assessment against the architecture document

| Document assumption | Finding |
|---|---|
| "Pinned Qwen3-8B experimental LittleBit checkpoint" | Exists and loads. It is a community upload, not a Samsung release. |
| LittleBit-2 keeps the LittleBit inference graph | Verified in code. |
| Licence CC BY-NC 4.0 | Verified for code and checkpoint. Base model is Apache-2.0. Commercial use is not permitted. |
| 8B footprint "4.3 GB" (illustrative) | File is 2.76 GiB. Resident memory depends on factor storage; see `docs/VALIDATION.md`. |
| Quality adequate for coding, reasoning, tool use | Not supported. Reported ARC-Challenge is below chance. This is a NO-GO for agent use under architecture §7. |
| XOR/popcount cannot compute signed weighted sums | Agreed. Activations are real-valued; the factor matmuls need real accumulation. |

## 5. Open questions

- Full-paper numbers for Qwen3 at 5 epochs. Resolve by reading the full LittleBit-2 paper or
  training a 5-epoch checkpoint.
- Whether an official Samsung Qwen3 checkpoint will be released. None was listed under
  `LittleBitLLM` or linked from the Samsung README on 2026-11-20.
- RTX 5060 Ti behaviour. Needs that hardware.

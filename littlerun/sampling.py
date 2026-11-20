"""Token sampling. Greedy when temperature == 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    seed: int | None = None


def make_generator(params: SamplingParams) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if params.seed is not None:
        g.manual_seed(int(params.seed))
    else:
        g.seed()
    return g


def sample(logits: torch.Tensor, params: SamplingParams, gen: torch.Generator) -> int:
    """logits: 1-D float tensor (any device). Sampling runs on CPU for reproducibility."""
    logits = logits.detach().float().cpu()
    if params.temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / params.temperature
    if params.top_k and params.top_k > 0 and params.top_k < logits.numel():
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > params.top_p  # keep the smallest prefix whose mass reaches top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen).item())

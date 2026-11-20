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
    presence_penalty: float = 0.0  # OpenAI semantics, applied to generated tokens
    frequency_penalty: float = 0.0  # OpenAI semantics, applied to generated tokens
    repetition_penalty: float = 1.0  # HF/vLLM semantics, applied to prompt + generated tokens

    @property
    def penalized(self) -> bool:
        return bool(self.presence_penalty or self.frequency_penalty or self.repetition_penalty != 1.0)


def apply_penalties(logits: torch.Tensor, params: SamplingParams, generated: list[int],
                    prompt: list[int]) -> torch.Tensor:
    """Return penalized float32 CPU logits. No-op when no penalty is set."""
    logits = logits.detach().float().cpu()
    if not params.penalized:
        return logits
    logits = logits.clone()
    V = logits.numel()
    if params.repetition_penalty != 1.0:
        seen = torch.tensor(sorted(set(prompt) | set(generated)), dtype=torch.long)
        seen = seen[(seen >= 0) & (seen < V)]
        vals = logits[seen]
        logits[seen] = torch.where(vals > 0, vals / params.repetition_penalty, vals * params.repetition_penalty)
    if generated and (params.presence_penalty or params.frequency_penalty):
        counts = torch.bincount(torch.tensor(generated, dtype=torch.long), minlength=V)[:V].float()
        logits = logits - params.frequency_penalty * counts - params.presence_penalty * (counts > 0).float()
    return logits


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

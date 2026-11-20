import torch

from littlerun.sampling import SamplingParams, apply_penalties, make_generator, sample


def test_greedy_and_seeded_sampling():
    logits = torch.tensor([0.1, 2.0, 1.9, -1.0])
    assert sample(logits, SamplingParams(temperature=0), make_generator(SamplingParams())) == 1
    p = SamplingParams(temperature=1.0, seed=7)
    a = [sample(logits, p, make_generator(p)) for _ in range(5)]
    assert len(set(a)) == 1


def test_top_k_and_top_p_restrict_support():
    logits = torch.tensor([5.0, 4.9, -10.0, -10.0])
    for p in (SamplingParams(1.0, top_k=2, seed=1), SamplingParams(1.0, top_p=0.9, seed=1)):
        g = make_generator(p)
        assert {sample(logits, p, g) for _ in range(50)} <= {0, 1}


def test_presence_and_frequency_penalties():
    logits = torch.zeros(5)
    out = apply_penalties(logits, SamplingParams(presence_penalty=0.5, frequency_penalty=0.25), [1, 1, 3], [0])
    assert torch.allclose(out, torch.tensor([0.0, -1.0, 0.0, -0.75, 0.0]))


def test_repetition_penalty_hf_semantics():
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_penalties(logits, SamplingParams(repetition_penalty=2.0), [1], [0])
    assert torch.allclose(out, torch.tensor([1.0, -4.0, 1.0, 0.5]))


def test_no_penalty_is_identity():
    logits = torch.randn(10)
    assert torch.equal(apply_penalties(logits, SamplingParams(), [1, 2], [3]), logits)


def test_penalties_do_not_mutate_input():
    logits = torch.tensor([2.0, -2.0])
    apply_penalties(logits, SamplingParams(repetition_penalty=2.0, presence_penalty=1.0), [0], [1])
    assert torch.equal(logits, torch.tensor([2.0, -2.0]))

"""Model-layer tests: adapter parity with transformers, KV cache, determinism, stop handling."""

import json

import pytest
import torch

from littlerun.engine import Engine, GenRequest, ModelRunner
from littlerun.loader import LoadOptions, load
from littlerun.sampling import SamplingParams


def _hf_dense_model(d, ref):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = json.loads((d / "config.json").read_text())
    hf = Qwen3ForCausalLM(Qwen3Config(**{k: v for k, v in cfg.items() if k not in ("architectures",)}))
    hf = hf.to(torch.float64).eval()
    sd = {k: v for k, v in ref["dense"].items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
    hf.config._attn_implementation = "sdpa"
    return hf


@pytest.fixture(scope="module")
def loaded_fp32(tiny_ckpt):
    d, _ = tiny_ckpt
    return load(d, options=LoadOptions(device="cpu", dtype="fp32", max_context=128))


def test_logits_match_transformers_dense_reconstruction(tiny_ckpt, loaded_fp32):
    d, ref = tiny_ckpt
    hf = _hf_dense_model(d, ref)
    ids = torch.tensor([[5, 17, 42, 99, 3, 250, 7, 8, 120, 64, 11]])
    with torch.no_grad():
        want = hf(ids.long()).logits[0].float()
    got = loaded_fp32.model(ids, None, logits="all")
    err = (got - want).abs().max().item()
    rms = ((got - want) ** 2).mean().sqrt().item()
    assert err < 2e-4 and rms < 5e-5, (err, rms)
    assert torch.equal(got.argmax(-1), want.argmax(-1))


def test_incremental_decode_matches_full_forward(loaded_fp32):
    lm = loaded_fp32
    ids = [5, 17, 42, 99, 3, 250, 7, 8, 120, 64, 11, 12]
    full = lm.model(torch.tensor([ids]), None, logits="all")
    cache = lm.adapter.create_kv_cache(lm.model, 64, lm.device)
    first = lm.adapter.prefill(lm.model, ids[:5], cache)
    assert torch.allclose(first, full[4], atol=1e-5)
    for i in range(5, len(ids)):
        step = lm.adapter.decode(lm.model, ids[i], cache)
        assert torch.allclose(step, full[i], atol=1e-5), i
    # chunked prefill with existing cache (explicit causal mask path)
    cache.reset()
    lm.model(torch.tensor([ids[:4]]), cache)
    chunk = lm.model(torch.tensor([ids[4:9]]), cache, logits="all")
    assert torch.allclose(chunk, full[4:9], atol=1e-5)


def test_context_overflow_raises(loaded_fp32):
    lm = loaded_fp32
    cache = lm.adapter.create_kv_cache(lm.model, 8, lm.device)
    with pytest.raises(ValueError, match="overflow"):
        lm.adapter.prefill(lm.model, list(range(9)), cache)


def _engine(lm):
    e = Engine(max_queue=2, max_tokens_limit=64)
    e.set_runner(ModelRunner(lm))
    return e


def test_fixed_seed_traces_are_reproducible(loaded_fp32):
    e = _engine(loaded_fp32)
    r = e.runner
    ids = r.encode("hello world")
    outs = [e.submit(GenRequest(ids, 12, SamplingParams(0.0))).result(30) for _ in range(2)]
    assert outs[0] == outs[1]
    s = [e.submit(GenRequest(ids, 12, SamplingParams(0.9, 0.9, 20, seed=123))).result(30) for _ in range(2)]
    assert s[0] == s[1]
    e.shutdown()


def test_bf16_int8_storage_close_to_fp32(tiny_ckpt, loaded_fp32):
    d, _ = tiny_ckpt
    lm16 = load(d, options=LoadOptions(device="cpu", dtype="bf16", factor_storage="int8", max_context=64))
    ids = torch.tensor([[5, 17, 42, 99, 3, 250, 7, 8]])
    a = loaded_fp32.model(ids, None, logits="all")
    b = lm16.model(ids, None, logits="all")
    rel = ((a - b).norm() / a.norm()).item()
    assert rel < 0.05, rel  # measured tolerance for bf16 (see docs/VALIDATION.md)
    lmc = load(d, options=LoadOptions(device="cpu", dtype="bf16", factor_storage="compute", max_context=64))
    assert torch.equal(lmc.model(ids, None, logits="all"), b)  # storage mode must not change numerics


def test_stop_token_and_stop_string(loaded_fp32):
    e = _engine(loaded_fp32)
    r = e.runner
    ids = r.encode("the quick brown")
    text, finish, usage = e.submit(GenRequest(ids, 16, SamplingParams(0.0))).result(30)
    assert finish in ("length", "stop") and usage["prompt_tokens"] == len(ids)
    if len(text) >= 3:
        stop = text[1:3]
        t2, f2, _ = e.submit(GenRequest(ids, 16, SamplingParams(0.0), stop=[stop])).result(30)
        assert f2 == "stop" and t2 == text[: text.find(stop)]
    # force the stop-token path: make every token a stop id
    saved = r.stop_ids
    r.stop_ids = set(range(10_000))
    t3, f3, u3 = e.submit(GenRequest(ids, 16, SamplingParams(0.0))).result(30)
    r.stop_ids = saved
    assert (t3, f3, u3["completion_tokens"]) == ("", "stop", 1)
    e.shutdown()


@pytest.mark.reference
def test_matches_samsung_reference_loader(samsung_ref, tmp_path):
    """End to end: Samsung trainer-side init + packed save -> Samsung loader vs LittleRun loader."""
    import shutil

    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from littlerun.manifest import MANIFEST_NAME, build_manifest
    from littlerun.testing.fixtures import tiny_config, write_tokenizer
    from quantization.utils.quant_util import apply_littlebit_patch, load_quantized_model
    import argparse

    d = tmp_path / "samsung"
    d.mkdir()
    vocab = write_tokenizer(d)
    cfg = tiny_config(vocab, inter=96)
    cfg.update({"eff_bit": 1.0, "use_itq": True, "itq_n_iter": 5, "residual": True})
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(**{k: v for k, v in cfg.items() if k != "architectures"}))
    args = argparse.Namespace(quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=True, split_dim=1024,
                              eff_bit=1.0, min_split_dim=8, use_itq=True, itq_n_iter=5, kv_factor=1.0, model_id="tiny")
    apply_littlebit_patch(hf, args, do_train=True)
    sd = {k: v.contiguous() for k, v in hf.state_dict().items()}
    save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "littlebit_config.json").write_text(json.dumps({k: cfg[k] for k in (
        "quant_func", "eff_bit", "split_dim", "residual", "kv_factor", "min_split_dim", "quant_mod", "use_itq",
        "itq_n_iter")}))
    m = build_manifest(d, source_repository="local/samsung-tiny", source_revision="t", license_id="test")
    (d / MANIFEST_NAME).write_text(json.dumps(m))

    ids = torch.tensor([[5, 17, 42, 99, 3, 250, 7, 8, 120, 64]])
    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        theirs = load_quantized_model(str(d), argparse.Namespace(**vars(args)), dt, device="cpu").eval()
        with torch.no_grad():
            want = theirs(ids).logits[0].float()
        ours = load(d, options=LoadOptions(device="cpu", dtype=dt_name, max_context=64))
        got = ours.model(ids, None, logits="all")
        err = (got - want).abs().max().item()
        tol = 1e-4 if dt is torch.float32 else 5e-2 * want.abs().max().item()
        assert err <= tol, (dt_name, err)
        assert (got.argmax(-1) == want.argmax(-1)).float().mean() >= (1.0 if dt is torch.float32 else 0.9)
    shutil.rmtree(d)

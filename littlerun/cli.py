"""Command line: manifest build, validate, generate, serve, bench."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import sys
import time
from pathlib import Path


def _load_opts(a):
    from .loader import LoadOptions

    return LoadOptions(device=a.device, dtype=a.dtype, factor_storage=a.factor_storage, max_context=a.max_context,
                       verify_hashes=not a.no_verify_hashes)


def _add_load_args(p):
    p.add_argument("checkpoint", help="checkpoint directory (read-only)")
    p.add_argument("--manifest", help="manifest path (default: <checkpoint>/littlerun.manifest.json)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--factor-storage", default="auto", choices=["auto", "int8", "compute", "int8mm", "packed"])
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--no-verify-hashes", action="store_true", help="skip sha256 checks (inventory still checked)")
    p.add_argument("--model-id", help="model id exposed by the API")


def _load(a):
    from .loader import load

    return load(a.checkpoint, manifest_path=a.manifest, model_id=a.model_id, options=_load_opts(a),
                progress=lambda m: print(f"[load] {m}", file=sys.stderr, flush=True))


def cmd_manifest_build(a):
    from .manifest import MANIFEST_NAME, build_manifest

    m = build_manifest(a.checkpoint, source_repository=a.repo, source_revision=a.revision, license_id=a.license,
                       base_model=a.base_model, base_model_license=a.base_model_license, hash_files=not a.no_hash)
    out = Path(a.output) if a.output else Path(a.checkpoint) / MANIFEST_NAME
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(out)  # atomic
    f = m["factors"]
    print(f"wrote {out}: {m['tensors']['count']} tensors, {f['groups']} factor groups, "
          f"residual={f['residual']}, bpw(factors+scales)={f['measured_bits_per_weight']}")
    print("rank partitions:", json.dumps(f["rank_partitions"]))


def cmd_validate(a):
    from .codecs.littlebit import LittleBitCodec
    from .manifest import MANIFEST_NAME, load_manifest, verify_against_checkpoint
    from .registry import ARCHITECTURES
    from .safetensors_io import TensorReader

    m = load_manifest(a.manifest or Path(a.checkpoint) / MANIFEST_NAME)
    t = time.time()
    verify_against_checkpoint(m, a.checkpoint, verify_hashes=not a.no_verify_hashes)
    adapter = ARCHITECTURES.get((m["architecture"]["id"], m["architecture"]["adapter_revision"]))()
    cfg = adapter.validate_config(m["model_config"])
    r = TensorReader(a.checkpoint)
    groups, claimed = LittleBitCodec().validate_tensors(r.inventory, r, adapter.linear_shapes(cfg),
                                                        residual=(m.get("littlebit_config") or {}).get("residual"))
    extra = set(r.inventory) - claimed - set(adapter.dense_tensors(cfg))
    if extra:
        raise SystemExit(f"unknown tensors: {sorted(extra)[:5]}")
    print(f"OK: manifest, inventory, {len(groups)} factor groups, padding and shapes valid "
          f"({time.time() - t:.1f}s, hashes {'skipped' if a.no_verify_hashes else 'verified'})")


def cmd_generate(a):
    from .engine import Engine, GenRequest, ModelRunner
    from .sampling import SamplingParams

    lm = _load(a)
    print(f"[load] done in {lm.load_seconds:.1f}s", file=sys.stderr)
    runner = ModelRunner(lm)
    eng = Engine(max_tokens_limit=max(a.max_tokens, 1))
    eng.set_runner(runner)
    if a.chat:
        prompt = runner.chat_prompt([{"role": "user", "content": a.prompt}], enable_thinking=a.thinking)
    else:
        prompt = a.prompt
    ids = runner.encode(prompt)
    h = eng.submit(GenRequest(ids, a.max_tokens, SamplingParams(a.temperature, a.top_p, a.top_k, a.seed,
                                                                 a.presence_penalty, a.frequency_penalty,
                                                                 a.repetition_penalty)))
    while True:
        ev = h.events.get()
        if ev.kind == "delta":
            print(ev.text, end="", flush=True)
        elif ev.kind == "error":
            raise SystemExit(ev.error)
        else:
            print(f"\n[finish={ev.finish_reason} usage={ev.usage} metrics="
                  f"{ {k: round(v, 3) if isinstance(v, float) else v for k, v in h.metrics.items()} }]",
                  file=sys.stderr)
            break


def run_chat_repl(runner, engine, a, inp=input, out=print):
    """Multi-turn chat. Commands: /reset clears history, /exit quits. Prefix reuse keeps turns cheap."""
    from .engine import GenRequest
    from .sampling import SamplingParams

    sp = SamplingParams(a.temperature, a.top_p, a.top_k, a.seed, a.presence_penalty, a.frequency_penalty,
                        a.repetition_penalty)
    history = [{"role": "system", "content": a.system}] if a.system else []
    while True:
        try:
            line = inp("you> ")
        except EOFError:
            break
        if line.strip() in ("/exit", "/quit"):
            break
        if line.strip() == "/reset":
            history = history[:1] if a.system else []
            out("[history cleared]")
            continue
        if not line.strip():
            continue
        history.append({"role": "user", "content": line})
        ids = runner.encode(runner.chat_prompt(history, enable_thinking=a.thinking))
        if len(ids) + a.max_tokens > runner.max_context:
            history.pop()
            out(f"[context full: {len(ids)} prompt tokens; use /reset]")
            continue
        h = engine.submit(GenRequest(ids, a.max_tokens, sp))
        parts = []
        while True:
            ev = h.events.get()
            if ev.kind == "delta":
                parts.append(ev.text)
                out(ev.text, end="", flush=True)
            elif ev.kind == "error":
                out(f"\n[error: {ev.error}]")
                history.pop()
                break
            else:
                cached = (ev.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                out(f"\n[{ev.finish_reason}; {ev.usage['completion_tokens']} tokens; "
                    f"{cached}/{ev.usage['prompt_tokens']} prompt tokens reused]")
                history.append({"role": "assistant", "content": "".join(parts)})
                break
    return history


def cmd_chat(a):
    from .engine import Engine, ModelRunner

    lm = _load(a)
    runner = ModelRunner(lm)
    eng = Engine(max_tokens_limit=max(a.max_tokens, 1))
    eng.set_runner(runner)
    print("[chat ready: /reset clears history, /exit quits]", file=sys.stderr)
    run_chat_repl(runner, eng, a)
    eng.shutdown()


def cmd_serve(a):
    import uvicorn

    from .engine import Engine, ModelRunner
    from .manifest import MANIFEST_NAME, load_manifest
    from .server import create_app

    m = load_manifest(a.manifest or Path(a.checkpoint) / MANIFEST_NAME)
    model_id = a.model_id or Path(m["source"]["repository"]).name
    a.model_id = model_id
    eng = Engine(max_queue=a.max_queue, max_tokens_limit=a.max_tokens_limit)
    eng.load_async(lambda: ModelRunner(_load(a)))
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {a.host} exposes the API beyond localhost", file=sys.stderr)
    uvicorn.run(create_app(eng, model_id), host=a.host, port=a.port, log_level="info")


def _peak_host_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def cmd_bench(a):
    import torch

    from .sampling import SamplingParams, make_generator, sample

    t0 = time.time()
    lm = _load(a)
    load_s = time.time() - t0
    dev = lm.device
    cache = lm.adapter.create_kv_cache(lm.model, lm.options.max_context, dev)
    gen = make_generator(SamplingParams(temperature=0))
    base = lm.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 400, add_special_tokens=False)
    prompt = base[: a.prompt_tokens]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    runs = []
    for r in range(a.warmup + a.repeats):
        cache.reset()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        sync()
        t = time.perf_counter()
        logits = lm.adapter.prefill(lm.model, prompt, cache)
        sync()
        t_prefill = time.perf_counter() - t
        tok = sample(logits, SamplingParams(temperature=0), gen)
        t = time.perf_counter()
        for _ in range(a.gen_tokens - 1):
            logits = lm.adapter.decode(lm.model, tok, cache)
            tok = sample(logits, SamplingParams(temperature=0), gen)
        sync()
        t_decode = time.perf_counter() - t
        if r >= a.warmup:
            row = {"prefill_tokens": len(prompt), "prefill_s": t_prefill, "prefill_tok_s": len(prompt) / t_prefill,
                   "ttft_s": t_prefill, "decode_tokens": a.gen_tokens - 1,
                   "decode_tok_s": (a.gen_tokens - 1) / t_decode if a.gen_tokens > 1 else None}
            if dev.type == "cuda":
                row["cuda_peak_allocated"] = torch.cuda.max_memory_allocated(dev)
                row["cuda_peak_reserved"] = torch.cuda.max_memory_reserved(dev)
            runs.append(row)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    result = {
        "checkpoint": str(lm.ckpt_dir), "source": lm.manifest["source"]["repository"],
        "revision": lm.manifest["source"]["revision"],
        "options": vars(lm.options), "device": str(dev), "load_s": load_s,
        "memory_estimate": lm.memory_estimate, "peak_host_rss_bytes": _peak_host_bytes(),
        "host": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                 "cpu_threads": torch.get_num_threads(),
                 "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None},
        "runs": runs,
    }
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1, default=str))
        print(f"wrote {a.output}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="littlerun", description="LittleBit-2 reference inference runtime")
    sub = p.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("manifest", help="manifest operations")
    msub = mp.add_subparsers(dest="mcmd", required=True)
    b = msub.add_parser("build", help="inspect a checkpoint and write a versioned manifest")
    b.add_argument("checkpoint")
    b.add_argument("--repo", required=True, help="source repository id, e.g. org/name")
    b.add_argument("--revision", required=True, help="immutable source revision (commit sha)")
    b.add_argument("--license", required=True)
    b.add_argument("--base-model")
    b.add_argument("--base-model-license")
    b.add_argument("--no-hash", action="store_true")
    b.add_argument("-o", "--output")
    b.set_defaults(fn=cmd_manifest_build)

    v = sub.add_parser("validate", help="verify checkpoint against manifest and codec contract")
    v.add_argument("checkpoint")
    v.add_argument("--manifest")
    v.add_argument("--no-verify-hashes", action="store_true")
    v.set_defaults(fn=cmd_validate)

    g = sub.add_parser("generate", help="one-shot generation")
    _add_load_args(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--chat", action="store_true", help="wrap prompt with the pinned chat template")
    g.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode in chat template")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--seed", type=int)
    g.add_argument("--presence-penalty", type=float, default=0.0)
    g.add_argument("--frequency-penalty", type=float, default=0.0)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("chat", help="interactive multi-turn chat in the terminal")
    _add_load_args(c)
    c.add_argument("--system", default="")
    c.add_argument("--thinking", action="store_true")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-p", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=20)
    c.add_argument("--seed", type=int)
    c.add_argument("--presence-penalty", type=float, default=0.0)
    c.add_argument("--frequency-penalty", type=float, default=0.0)
    c.add_argument("--repetition-penalty", type=float, default=1.0)
    c.set_defaults(fn=cmd_chat)

    s = sub.add_parser("serve", help="run the OpenAI-style HTTP API")
    _add_load_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-queue", type=int, default=4)
    s.add_argument("--max-tokens-limit", type=int, default=2048)
    s.set_defaults(fn=cmd_serve)

    bn = sub.add_parser("bench", help="measure load, prefill and decode speed")
    _add_load_args(bn)
    bn.add_argument("--prompt-tokens", type=int, default=128)
    bn.add_argument("--gen-tokens", type=int, default=32)
    bn.add_argument("--warmup", type=int, default=1)
    bn.add_argument("--repeats", type=int, default=3)
    bn.add_argument("-o", "--output")
    bn.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

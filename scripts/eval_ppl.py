"""WikiText-2 perplexity with the same protocol as SamsungLabs/LittleBit eval.py.

Protocol (eval.py + utils/datautils.py@42d658b): join test split with "\\n\\n", tokenize once,
non-overlapping windows of `seqlen`, no BOS, mean token NLL over all windows' (seqlen-1) targets.

Usage:
  python scripts/eval_ppl.py CKPT --parquet wikitext-2-raw-v1-test.parquet [--windows N] [-o out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

from littlerun.loader import LoadOptions, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--parquet", required=True, help="wikitext-2-raw-v1 test split parquet file")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--windows", type=int, default=0, help="0 = all windows")
    p.add_argument("--start", type=int, default=0, help="first window index (to resume an interrupted run)")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--factor-storage", default="compute")
    p.add_argument("-o", "--output")
    a = p.parse_args()

    import pyarrow.parquet as pq

    texts = pq.read_table(a.parquet).column("text").to_pylist()
    lm = load(a.checkpoint, options=LoadOptions(device=a.device, dtype=a.dtype, factor_storage=a.factor_storage,
                                                max_context=a.seqlen, verify_hashes=False))
    ids = lm.tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    total = ids.numel() // a.seqlen
    n = total if a.windows <= 0 else min(a.windows, total)
    print(f"tokens={ids.numel()} windows={total} evaluating={n}", file=sys.stderr, flush=True)

    nll_sum, count, per_window = 0.0, 0, []
    head = lm.model.lm_head_weight
    t0 = time.time()
    for i in range(a.start, n):
        x = ids[i * a.seqlen:(i + 1) * a.seqlen].to(lm.device)
        hidden = lm.model(x[None], None, logits="hidden")[:-1]
        targets = x[1:]
        w_nll = 0.0
        for s in range(0, hidden.shape[0], 256):  # chunk lm_head to bound memory
            logits = (hidden[s:s + 256] @ head.t()).float()
            w_nll += F.cross_entropy(logits, targets[s:s + 256], reduction="sum").item()
        nll_sum += w_nll
        count += targets.numel()
        per_window.append(w_nll / targets.numel())
        print(f"window {i + 1}/{n} nll={per_window[-1]:.4f} running_ppl={math.exp(nll_sum / count):.4f} "
              f"elapsed={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    ppl = math.exp(nll_sum / count)
    res = {"dataset": "wikitext-2-raw-v1/test", "seqlen": a.seqlen, "start": a.start, "windows": n, "total_windows": total,
           "tokens_scored": count, "ppl": ppl, "dtype": a.dtype, "device": str(lm.device),
           "factor_storage": a.factor_storage, "per_window_nll": per_window, "seconds": time.time() - t0}
    print(json.dumps({k: v for k, v in res.items() if k != "per_window_nll"}))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

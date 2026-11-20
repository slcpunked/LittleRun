"""Zero-shot multiple-choice log-likelihood eval (ARC-Easy / ARC-Challenge), lm-eval style.

Mirrors lm-evaluation-harness 0.4.x `arc_easy`/`arc_challenge`:
  context      = "Question: {question}\\nAnswer:"
  continuation = " {choice text}"
  acc      = argmax of summed continuation log-likelihood
  acc_norm = argmax of log-likelihood / len(continuation in UTF-8 bytes)
Tokenisation follows lm-eval: encode(context + continuation) and take the tokens after encode(context).
The context K/V is computed once per question and reused for every choice.

Usage:
  python scripts/eval_mc.py CKPT --parquet ARC-Easy-test.parquet [--limit N] [-o out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from littlerun.loader import LoadOptions, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--parquet", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--factor-storage", default="auto")
    p.add_argument("-o", "--output")
    a = p.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(a.parquet).to_pylist()
    if a.limit:
        rows = rows[: a.limit]
    lm = load(a.checkpoint, options=LoadOptions(device=a.device, factor_storage=a.factor_storage, max_context=1024,
                                                verify_hashes=False))
    tok, model = lm.tokenizer, lm.model
    cache = lm.adapter.create_kv_cache(model, 1024, lm.device)
    correct = correct_norm = 0
    details = []
    t0 = time.time()
    for qi, row in enumerate(rows):
        ctx = f"Question: {row['question']}\nAnswer:"
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        cache.reset()
        if len(ctx_ids) > 1:
            model(torch.tensor([ctx_ids[:-1]], device=lm.device), cache, logits="last")
        base = cache.length
        scores, norms = [], []
        for text in row["choices"]["text"]:
            cont = " " + text
            whole = tok.encode(ctx + cont, add_special_tokens=False)
            cont_ids = whole[len(ctx_ids):]
            if whole[: len(ctx_ids)] != ctx_ids or not cont_ids:
                raise SystemExit(f"tokenisation boundary mismatch at question {qi}")
            cache.length = base
            inp = torch.tensor([[ctx_ids[-1]] + cont_ids[:-1]], device=lm.device)
            logits = model(inp, cache, logits="all")
            lp = torch.log_softmax(logits.float(), dim=-1)
            ll = lp[torch.arange(len(cont_ids)), torch.tensor(cont_ids)].sum().item()
            scores.append(ll)
            norms.append(ll / len(cont.encode("utf-8")))
        gold = row["choices"]["label"].index(row["answerKey"])
        pred, pred_n = max(range(len(scores)), key=scores.__getitem__), max(range(len(norms)), key=norms.__getitem__)
        correct += pred == gold
        correct_norm += pred_n == gold
        details.append({"id": row["id"], "gold": gold, "pred": pred, "pred_norm": pred_n, "ll": scores})
        if (qi + 1) % 100 == 0 or qi + 1 == len(rows):
            print(f"{qi + 1}/{len(rows)} acc={correct / (qi + 1):.4f} acc_norm={correct_norm / (qi + 1):.4f} "
                  f"elapsed={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    n = len(rows)
    res = {"parquet": a.parquet, "n": n, "acc": correct / n, "acc_norm": correct_norm / n,
           "factor_storage": lm.options.factor_storage, "dtype": str(lm.dtype), "seconds": time.time() - t0,
           "details": details}
    print(json.dumps({k: v for k, v in res.items() if k != "details"}))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()

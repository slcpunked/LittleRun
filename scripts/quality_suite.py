"""Agent-readiness quality suite with predeclared minimums (architecture §7).

Runs fixed prompts through the real engine (chat template, greedy, thinking off) and grades
them with deterministic checks. Minimums live in docs/quality_minimums.json and must be set
before results are seen. The suite prints PASS/FAIL per family and exits 1 on any FAIL.

Families (all prompts are in this file so they are archived with the results):
  json      - produce a JSON object with given keys and types; graded by json.loads + schema
  arith     - single-step arithmetic with an exact integer answer
  follow    - simple format instructions (uppercase, word count, list length)

Usage:
  python scripts/quality_suite.py CKPT [--minimums docs/quality_minimums.json] [-o results.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

JSON_TASKS = [
    ("Return a JSON object with keys \"name\" (string) and \"age\" (integer) for a person named Ana aged 31.",
     {"name": str, "age": int}),
    ("Return a JSON object with key \"colors\" whose value is a list of three color names.", {"colors": list}),
    ("Return a JSON object with keys \"city\" (string) and \"population\" (integer) for a city of your choice.",
     {"city": str, "population": int}),
    ("Return a JSON object with key \"ok\" set to the boolean true.", {"ok": bool}),
    ("Return a JSON object with keys \"x\" and \"y\", both integers, for the point (3, 4).", {"x": int, "y": int}),
    ("Return a JSON object with key \"title\" (string) for a book about the sea.", {"title": str}),
    ("Return a JSON object with keys \"a\" (integer 1) and \"b\" (integer 2).", {"a": int, "b": int}),
    ("Return a JSON object with key \"items\" containing a list of two fruit names.", {"items": list}),
    ("Return a JSON object with key \"temperature\" (number) equal to 21.5.", {"temperature": (int, float)}),
    ("Return a JSON object with keys \"first\" and \"last\" (strings) for the name Grace Hopper.",
     {"first": str, "last": str}),
]

ARITH_TASKS = [("What is 7 + 5?", 12), ("What is 9 * 6?", 54), ("What is 100 - 37?", 63), ("What is 48 / 6?", 8),
               ("What is 15 + 27?", 42), ("What is 12 * 12?", 144), ("What is 81 / 9?", 9), ("What is 250 - 125?", 125),
               ("What is 13 * 3?", 39), ("What is 64 + 36?", 100)]

FOLLOW_TASKS = [
    ("Write the word hello in uppercase letters only.", lambda t: "HELLO" in t and "hello" not in t),
    ("List exactly three animals, one per line, with no other text.",
     lambda t: len([ln for ln in t.strip().splitlines() if ln.strip()]) == 3),
    ("Reply with exactly the single word: yes", lambda t: t.strip().strip(".").lower() == "yes"),
    ("Write a sentence that ends with the word 'blue'.", lambda t: t.strip().rstrip(".!").lower().endswith("blue")),
    ("Count from 1 to 5 separated by commas.", lambda t: re.sub(r"\s", "", t).startswith("1,2,3,4,5")),
]

SYSTEM = "You are a precise assistant. Follow instructions exactly and answer briefly."


def extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def grade_json(text, schema) -> bool:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return False
    for k, t in schema.items():
        if k not in obj:
            return False
        if t is int and isinstance(obj[k], bool):
            return False
        if not isinstance(obj[k], t):
            return False
    return True


def grade_arith(text, answer) -> bool:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return bool(nums) and float(nums[0]) == float(answer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--minimums", default=str(Path(__file__).resolve().parent.parent / "docs" / "quality_minimums.json"))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("-o", "--output")
    a = p.parse_args()

    from littlerun.engine import Engine, GenRequest, ModelRunner
    from littlerun.loader import LoadOptions, load
    from littlerun.sampling import SamplingParams

    minimums = json.loads(Path(a.minimums).read_text())
    lm = load(a.checkpoint, options=LoadOptions(max_context=1024, verify_hashes=False))
    runner = ModelRunner(lm)
    eng = Engine(max_tokens_limit=a.max_tokens)
    eng.set_runner(runner)

    def ask(prompt: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        ids = runner.encode(runner.chat_prompt(msgs, enable_thinking=False))
        text, _, _ = eng.submit(GenRequest(ids, a.max_tokens, SamplingParams(temperature=0.0))).result(600)
        return text

    t0 = time.time()
    results = {}
    for fam, tasks, grader in (("json", JSON_TASKS, grade_json), ("arith", ARITH_TASKS, grade_arith),
                               ("follow", FOLLOW_TASKS, lambda t, f: f(t))):
        rows = []
        for prompt, spec in tasks:
            out = ask(prompt)
            rows.append({"prompt": prompt, "output": out, "pass": bool(grader(out, spec))})
            print(f"[{fam}] {'PASS' if rows[-1]['pass'] else 'fail'}: {out[:70]!r}", file=sys.stderr, flush=True)
        score = sum(r["pass"] for r in rows) / len(rows)
        results[fam] = {"score": score, "minimum": minimums[fam], "pass": score >= minimums[fam], "rows": rows}
    eng.shutdown()
    summary = {k: {"score": v["score"], "minimum": v["minimum"], "pass": v["pass"]} for k, v in results.items()}
    overall = all(v["pass"] for v in results.values())
    print(json.dumps({"overall_pass": overall, "families": summary, "seconds": round(time.time() - t0, 1)}, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps({"checkpoint": a.checkpoint, "minimums_file": a.minimums,
                                              "decoding": "greedy, thinking off, max_tokens=%d" % a.max_tokens,
                                              "overall_pass": overall, "results": results}, indent=1))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

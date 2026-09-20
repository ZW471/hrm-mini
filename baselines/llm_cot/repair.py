"""Repair a wrong teacher trace: cut it at its first committed wrong placement, let the teacher
continue from the verified prefix, keep the result only if the final grid is exactly right.

A wrong Sudoku trace is usually right for a long time and then commits one bad placement that
cascades. With the gold solution in hand the bad step can be found *exactly*, not judged by
another model: every `rXcY=D` the trace asserts is checked against gold. Two guards keep valid
reasoning from being cut:

* hypothetical placements ("suppose r5c1=7 ... contradiction") are skipped when the assertion
  sits in an if/suppose/try/assume context, or is retracted shortly after;
* when the trace did produce a final grid, only cells that are wrong *in that grid* count, and the
  cut is at the first assertion of such a cell with its (wrong) final value -- the root of the
  cascade.

The prefix is then continued with DeepSeek's chat-prefix completion (the teacher keeps writing
in the same voice), and the spliced trace `[verified prefix | continuation]` is kept only if its
final grid matches gold. Records go to their own JSONL (`--out`, marked `repaired: true`) so this
can run alongside `generate.py`; `train.py` takes several trace files and keeps one correct record
per puzzle.

    python -m baselines.llm_cot.repair --source cot/deepseek_deepseek-v4-pro/train.jsonl \
        --out cot/deepseek_deepseek-v4-pro/repair.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time

import requests

from baselines.llm_cot.generate import api_key
from baselines.llm_rlvr.common import grade, make_prompt

URL = "https://api.deepseek.com/beta/chat/completions"
PLACEMENT = re.compile(r"\b[rR](\d)[cC](\d)\s*(?:=|is|must be)\s*(\d)\b")
HYPOTHETICAL = re.compile(r"\b(if|suppose|assume|assuming|try|trying|case|option|hypothes\w*|would|let'?s say|say)\b[^.\n]{0,40}$",
                          re.IGNORECASE)


def first_wrong_placement(reasoning: str, gold: str, pred: str | None) -> int | None:
    """Character offset of the first committed wrong placement, or None if no such step is found."""
    wrong_final = None
    if pred:
        wrong_final = {(i // 9 + 1, i % 9 + 1): int(pred[i]) for i in range(81) if pred[i] != gold[i]}
    for m in PLACEMENT.finditer(reasoning):
        r, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d == 0 or d == int(gold[(r - 1) * 9 + (c - 1)]):
            continue
        if wrong_final is not None and wrong_final.get((r, c)) != d:
            continue  # a wrong value the trace did not end up committing to
        if HYPOTHETICAL.search(reasoning[max(0, m.start() - 60):m.start()]):
            continue
        # retracted soon after? (same cell asserted with another digit within the next 3000 chars)
        window = reasoning[m.end():m.end() + 3000]
        retracted = any(int(n.group(1)) == r and int(n.group(2)) == c and int(n.group(3)) != d
                        for n in PLACEMENT.finditer(window))
        if retracted and wrong_final is None:
            continue
        return m.start()
    return None


def cut_point(reasoning: str, offset: int) -> int:
    """Start of the line containing `offset`, so the prefix ends on a clean line."""
    return reasoning.rfind("\n", 0, offset) + 1


def continue_from(key: str, model: str, prompt: str, prefix: str, max_tokens: int, temperature: float,
                  timeout: float) -> dict:
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": prefix, "prefix": True}]}
    delay, attempt = 5.0, 0
    while True:
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
            j = r.json() if r.status_code == 200 else {}
            if "choices" in j:
                return j
            err = str(j.get("error") or r.text[:300])
            if r.status_code == 429 or "rate_limit" in err or "Too many requests" in err:
                time.sleep(random.uniform(20, 60))
                continue
        except requests.RequestException as e:  # noqa: PERF203
            err = repr(e)
        attempt += 1
        if attempt >= 4:
            return {"error": err}
        time.sleep(delay)
        delay *= 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="the generate.py JSONL with the wrong traces")
    p.add_argument("--out", required=True, help="JSONL for repaired records (appended; indices done are skipped)")
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--max-total-tokens", type=int, default=131072, help="prefix + continuation")
    p.add_argument("--max-prefix-tokens", type=int, default=90000, help="skip traces cut later than this")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--workers", type=int, default=200)
    p.add_argument("--limit", type=int, default=None, help="repair at most this many (smoke tests)")
    p.add_argument("--include-unparsed", action="store_true",
                   help="also repair traces that never produced a grid (budget exhausted)")
    args = p.parse_args()
    key = api_key("DEEPSEEK_API_KEY")

    by_index: dict[int, dict] = {}
    solved, repaired = set(), set()
    for line in open(args.source):
        r = json.loads(line)
        if r.get("exact"):
            solved.add(r["index"])
        if "error" not in r and r.get("reasoning") and (r["index"] not in by_index or r.get("parsed")):
            by_index[r["index"]] = r  # prefer an attempt that produced a grid: its cascade can be anchored
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            repaired.add(r["index"])
            if r.get("exact"):
                solved.add(r["index"])
    todo = []
    for i, r in by_index.items():
        if i in solved or i in repaired:
            continue
        if not r.get("parsed") and not args.include_unparsed:
            continue
        off = first_wrong_placement(r["reasoning"], r["answer"], r.get("pred"))
        if off is None:
            continue
        cut = cut_point(r["reasoning"], off)
        if cut // 3 > args.max_prefix_tokens:  # ~3 chars/token for this kind of text
            continue
        todo.append((r, cut))
    random.Random(1).shuffle(todo)
    todo = todo[: args.limit] if args.limit else todo
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[repair] {len(todo)} traces to repair ({len(solved)} puzzles already solved, "
          f"{len(repaired)} already repaired)", flush=True)
    lock = threading.Lock()
    stats = {"n": 0, "exact": 0, "errors": 0}

    def work(item):
        r, cut = item
        prefix = r["reasoning"][:cut]
        prompt = make_prompt(r["question"], r.get("layout", "rows"))[0]["content"]
        budget = max(4096, args.max_total_tokens - len(prefix) // 3)
        t0 = time.time()
        j = continue_from(key, args.model, prompt, prefix, budget, args.temperature, args.timeout)
        rec = {"index": r["index"], "question": r["question"], "answer": r["answer"], "augmented": False,
               "seed": r.get("seed"), "backend": "deepseek", "model": args.model, "layout": r.get("layout", "rows"),
               "repaired": True, "attempt": r.get("attempt", 1), "prefix_chars": cut,
               "source_wrong_cells": None if not r.get("pred") else sum(a != b for a, b in zip(r["pred"], r["answer"])),
               "secs": round(time.time() - t0, 1)}
        if "error" in j:
            rec.update(error=j["error"], exact=False)
        else:
            cont = j["choices"][0]["message"].get("content") or ""
            full = prefix + cont
            g = grade(f"<think>\n{full}\n</think>\n\n{cont[-2000:]}", r["question"], r["answer"])
            # the answer the student should emit after </think>: the final grid, as nine rows
            content = "\n".join(r["answer"][k:k + 9] for k in range(0, 81, 9)) if g["exact"] else cont[-2000:]
            rec.update(reasoning=full, content=content, continuation_chars=len(cont),
                       finish=j["choices"][0].get("finish_reason"), usage=j.get("usage", {}), **g)
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            stats["n"] += 1
            stats["exact"] += int(rec.get("exact", False))
            stats["errors"] += int("error" in rec)
            if stats["n"] % 10 == 0 or stats["n"] == len(todo):
                print(f"[repair] {stats['n']}/{len(todo)} done, {stats['exact']} repaired correctly "
                      f"({stats['exact'] / stats['n']:.0%}), {stats['errors']} errors", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    main()

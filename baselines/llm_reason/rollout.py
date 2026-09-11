"""Let the LLM reason about a Sudoku, then read off its answer. Rollouts for training and eval.

The reasoning baseline in this package trains on the model's *own* reasoning: there is no CoT data
for Sudoku-Extreme, so the trace comes from the current policy and only the final answer carries a
loss (`train.py`). This script produces those traces with vLLM, and doubles as the evaluator because
the protocol is the same both times:

    1. think: sample from the chat-templated prompt (thinking on) until `</think>` or the token
       budget runs out;
    2. answer: close the think block -- forcibly, if the model did not -- and greedily decode
       the reply, from which the 81-digit grid is parsed.

Step 2 is s1-style budget forcing. Without it a trace that overruns the budget is simply a
non-answer, and on Sudoku-Extreme most traces overrun any budget one can afford; forcing an answer
turns "ran out of tokens" into a wrong-or-right grid and makes train and eval sequences the same
shape. `closed_rate` records how often the model finished thinking by itself.

Each puzzle is written to JSONL with the think-trace token ids, which `train.py` splices in front
of the gold answer verbatim, so the training sequence is exactly the one the model produced.

    # training rollouts: 2048 augmented views of the 1000 training puzzles, sharded 2 ways
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-vllm/bin/python -m baselines.llm_reason.rollout \
        --model Qwen/Qwen3.8-27B --split train --offset 0 --count 2048 --dp-rank 0 --dp-size 2 \
        --out rollouts/round1/train
    # evaluation: the first 512 test_hard puzzles, un-augmented
    ... --split test_hard --count 512 --no-augment --out rollouts/round1/eval
    # combine the shards and print / log the metrics
    ... --merge rollouts/round1/eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

# The repo imports below touch CUDA before vLLM forks its workers; spawned workers do not care.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# FlashInfer's sampling kernels are built for a newer CUDA than driver 535 runs ("device kernel
# image is invalid"); vLLM's own top-k/top-p sampler is fine.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from baselines.llm_sft.data import GRID, read_split
from baselines.llm_sft.eval import is_legal_solution
from baselines.llm_sft.prompt_eval import INSTRUCTION, parse_grid
from dataset.sudoku import shuffle_sudoku

THINK_END = "</think>"


def augmented_rows(rows: list[tuple[str, str]], indices: range, seed: int, augment: bool = True):
    """The same (seed, index)-addressed augmentation as `llm_sft.data.SudokuSFTDataset`."""
    out = []
    for idx in indices:
        question, answer = rows[idx % len(rows)]
        if augment:
            np.random.seed((seed * 1_000_003 + idx) % (2**31 - 1))
            board = np.frombuffer(question.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            solution = np.frombuffer(answer.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            board, solution = shuffle_sudoku(board, solution)
            question = "".join(map(str, board.flatten().tolist()))
            answer = "".join(map(str, solution.flatten().tolist()))
        out.append((idx, question, answer))
    return out


def prompt_ids(tok, question: str) -> list[int]:
    """Chat-templated prompt with thinking on; ends in `<think>\\n`, where the trace starts."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION.format(puzzle=question)}],
        tokenize=False, add_generation_prompt=True,
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def close_ids(tok, think: list[int]) -> list[int]:
    """Tokens that end the think block: `</think>\\n\\n`, on its own line."""
    newline = tok.encode("\n", add_special_tokens=False)
    tail = "" if think and think[-len(newline):] == newline else "\n"
    return tok.encode(f"{tail}{THINK_END}\n\n", add_special_tokens=False)


def generate(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    think_end = tok.convert_tokens_to_ids(THINK_END)
    specials = set(tok.all_special_ids) | {think_end}

    rows = read_split(args.data_dir, args.split)
    indices = range(args.offset, args.offset + args.count)
    samples = augmented_rows(rows, indices, args.seed, augment=not args.no_augment)
    samples = samples[args.dp_rank :: args.dp_size]
    prompts = [prompt_ids(tok, q) for _, q, _ in samples]

    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
        max_model_len=max(map(len, prompts)) + args.max_think_tokens + args.max_answer_tokens + 8,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed + args.dp_rank,
        enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
    )

    # 1. think. The recommended thinking-mode sampling for this model family; greedy decoding of
    #    long reasoning degenerates into loops, so it is not an option here.
    think_params = SamplingParams(
        max_tokens=args.max_think_tokens, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, stop_token_ids=[think_end], seed=args.seed + args.dp_rank,
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], think_params, use_tqdm=True)
    thinks, closed = [], []
    for o in outs:
        ids = list(o.outputs[0].token_ids)
        closed.append(o.outputs[0].finish_reason == "stop" and o.outputs[0].stop_reason == think_end)
        while ids and ids[-1] in specials:  # the stop token, or a stray EOS
            ids.pop()
        thinks.append(ids)

    # 2. answer. Closed for the model if it did not close by itself; greedy, so the readout of
    #    a given trace is deterministic.
    answer_params = SamplingParams(max_tokens=args.max_answer_tokens, temperature=0.0)
    closes = [close_ids(tok, t) for t in thinks]
    outs = llm.generate(
        [{"prompt_token_ids": p + t + c} for p, t, c in zip(prompts, thinks, closes)],
        answer_params, use_tqdm=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.rank{args.dp_rank}.jsonl", "w") as f:
        for (idx, q, a), t, c, was_closed, o in zip(samples, thinks, closes, closed, outs):
            reply = o.outputs[0].text
            grid = parse_grid(reply, reasoning=False)
            rec = {
                "index": idx, "question": q, "answer": a,
                "think_ids": t, "close_ids": c, "n_think": len(t), "closed": was_closed,
                "reply": reply,
                "pred": None if grid is None else "".join(map(str, grid.tolist())),
            }
            f.write(json.dumps(rec) + "\n")


def merge(args) -> dict:
    """Score every shard of `--merge` together; print, and log to W&B unless told not to."""
    paths = sorted(glob.glob(f"{args.merge}.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no shards match {args.merge}.rank*.jsonl")
    recs = [json.loads(line) for p in paths for line in open(p)]
    n = len(recs)
    gold = np.array([[int(c) for c in r["answer"]] for r in recs])
    given = np.array([[int(c) for c in r["question"]] for r in recs])
    parsed = np.array([r["pred"] is not None for r in recs])
    pred = np.array([[int(c) for c in (r["pred"] or "0" * GRID)] for r in recs])
    n_think = np.array([r["n_think"] for r in recs])
    correct = (pred == gold).all(axis=1) & parsed
    clue = given > 0
    metrics = {
        "n": n,
        "exact_match": float(correct.mean()),
        "cell_accuracy": float(((pred == gold) & parsed[:, None]).sum() / (n * GRID)),
        "legal_grid": float((is_legal_solution(pred) & parsed).mean()),
        "clues_kept": float(((pred == given) & clue & parsed[:, None]).sum() / clue.sum()),
        "parse_rate": float(parsed.mean()),
        "closed_rate": float(np.mean([r["closed"] for r in recs])),
        "think_tokens_mean": float(n_think.mean()),
        "think_tokens_p50": float(np.median(n_think)),
        "think_tokens_p90": float(np.percentile(n_think, 90)),
        "exact_match_when_closed": float(correct[[r["closed"] for r in recs]].mean())
        if any(r["closed"] for r in recs) else 0.0,
    }
    print(f"{args.merge}  n={n}")
    for k, v in metrics.items():
        if k != "n":
            print(f"  {k:24s} = {v:.4f}")
    with open(f"{args.merge}.metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.wandb_id:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow",
                         settings=wandb.Settings(x_disable_stats=True))
        prefix = args.wandb_prefix
        payload = {f"{prefix}/{k}": v for k, v in metrics.items() if k != "n"}
        if prefix == "eval":
            # train.py's checkpoint-selection metric, under the name the other baselines use
            payload["eval/test_hard_exact_match"] = metrics["exact_match"]
        run.log(payload, step=args.wandb_step)
        run.finish()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", default="train")
    p.add_argument("--offset", type=int, default=0, help="first augmentation index")
    p.add_argument("--count", type=int, default=2048, help="puzzles to roll out")
    p.add_argument("--no-augment", action="store_true", help="the raw rows (for evaluation)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dp-rank", type=int, default=0)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--tp", type=int, default=4, help="tensor parallel size of this replica")
    # The torch sampler's top-k/top-p pass over [max_num_seqs, 248k] fp32 logits is not in
    # vLLM's memory profile, so leave headroom and cap the concurrency it sizes that buffer by.
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-think-tokens", type=int, default=32768)
    p.add_argument("--max-answer-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", default=None, help="JSONL prefix; a shard writes <out>.rank<i>.jsonl")
    p.add_argument("--merge", default=None, help="score the shards under this prefix instead")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--wandb-id", default=None, help="log merged metrics into this W&B run")
    p.add_argument("--wandb-prefix", default="eval", choices=["eval", "rollout"])
    p.add_argument("--wandb-step", type=int, default=0)
    args = p.parse_args()

    if args.merge:
        merge(args)
    else:
        if not args.out:
            p.error("--out is required to generate")
        generate(args)


if __name__ == "__main__":
    main()

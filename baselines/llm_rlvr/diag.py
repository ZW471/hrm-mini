"""Go/no-go diagnostic for RLVR: pass@k, closed-rate and the hint-difficulty curve, with vLLM.

RL with a verifiable reward sharpens what the policy can already sample; it does not conjure
solutions the base model never produces. So before spending GPU-days on GRPO, measure how much
room there is: pass@k against pass@1 on `test_hard`, and how the success rate rises as training
puzzles are made easier with hints (`common.hinted_puzzle`) -- the curve the curriculum will
walk back down.

Each config is `<split>:<empties>:<budget>[:<count>]` (`empties` = blanks left after hints;
ignored for `test_hard`, which is always the raw puzzle; `count` overrides `--count`); one vLLM
engine sized to the largest budget serves all of them. Sharded over GPUs by `--dp-rank/--dp-size` (one process per GPU, TP=1 for a small
model), then `--merge` scores the shards.

    for i in 0..7: CUDA_VISIBLE_DEVICES=$i python -m baselines.llm_rlvr.diag --model Qwen/Qwen3.5-4B \
        --config test_hard:0:8192 --config train:8:8192 --n 8 --count 128 --dp-rank $i --dp-size 8 --out diag/base
    python -m baselines.llm_rlvr.diag --merge diag/base
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from baselines.llm_rlvr.common import eval_rows, grade, summarize, train_rows


def parse_config(s: str, default_count: int) -> tuple[str, int, int, int]:
    split, empties, budget, *rest = s.split(":")
    return split, int(empties), int(budget), int(rest[0]) if rest else default_count


def generate(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    configs = [parse_config(c, args.count) for c in args.config]
    max_budget = max(b for _, _, b, _ in configs)

    jobs = []  # (config, row, prompt_ids)
    for split, empties, budget, count in configs:
        if split == "test_hard":
            rows = eval_rows(args.data_dir, count, args.layout)
        else:
            rows = train_rows(args.data_dir, args.seed, range(args.offset, args.offset + count), empties, args.layout)
        rows = rows[args.dp_rank :: args.dp_size]
        for r in rows:
            text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
            jobs.append(((split, empties, budget), r, tok(text, add_special_tokens=False)["input_ids"]))

    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
        max_model_len=max(len(p) for _, _, p in jobs) + max_budget + 8,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed + args.dp_rank,
        enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.rank{args.dp_rank}.jsonl", "w") as f:
        for cfg in {c for c, _, _ in jobs}:
            mine = [(r, p) for c, r, p in jobs if c == cfg]
            if not mine:
                continue
            params = SamplingParams(n=args.n, max_tokens=cfg[2], temperature=args.temperature,
                                    top_p=args.top_p, top_k=args.top_k, seed=args.seed + args.dp_rank)
            outs = llm.generate([{"prompt_token_ids": p} for _, p in mine], params, use_tqdm=True)
            for (r, _), o in zip(mine, outs):
                samples = []
                for s in o.outputs:
                    g = grade(s.text, r["question"], r["answer"])
                    g["n_tokens"] = len(s.token_ids)
                    g["truncated"] = s.finish_reason == "length"
                    samples.append(g)
                f.write(json.dumps({"config": list(cfg), "index": r["index"], "empties": r["empties"],
                                    "samples": samples}) + "\n")


def merge(args) -> dict:
    paths = sorted(glob.glob(f"{args.merge}.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no shards match {args.merge}.rank*.jsonl")
    recs = [json.loads(line) for p in paths for line in open(p)]
    by_cfg: dict[tuple, list[dict]] = {}
    for r in recs:
        by_cfg.setdefault(tuple(r["config"]), []).append(r)

    report = {}
    for cfg, rows in sorted(by_cfg.items()):
        flat = [s for r in rows for s in r["samples"]]
        m = summarize(flat, [s["n_tokens"] for s in flat])
        k = len(rows[0]["samples"])
        m["pass@1"] = m.pop("exact_match")
        m[f"pass@{k}"] = float(np.mean([any(s["exact"] for s in r["samples"]) for r in rows]))
        m["n_puzzles"] = len(rows)
        m["truncated_rate"] = float(np.mean([s["truncated"] for s in flat]))
        key = f"{cfg[0]}/empties{cfg[1]}/budget{cfg[2]}"
        report[key] = m
        print(f"{key}  puzzles={len(rows)} samples/puzzle={k}")
        for name, v in m.items():
            if name not in ("n", "n_puzzles"):
                print(f"  {name:24s} = {v:.4f}")
    with open(f"{args.merge}.metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.wandb_id:
        import wandb

        run = wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow",
                         settings=wandb.Settings(x_disable_stats=True))
        # Own x-axis, so evals can be logged into a run the trainer also writes to, in any order.
        run.define_metric(f"{args.wandb_prefix}/step")
        run.define_metric(f"{args.wandb_prefix}/*", step_metric=f"{args.wandb_prefix}/step")
        payload = {f"{args.wandb_prefix}/{key}/{name}": v for key, m in report.items()
                   for name, v in m.items() if name not in ("n", "n_puzzles")}
        payload[f"{args.wandb_prefix}/step"] = args.wandb_step
        run.log(payload)
        run.finish()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--config", action="append", default=None, help="<split>:<empties>:<budget>[:<count>], repeatable")
    p.add_argument("--layout", default="rows", choices=["rows", "line"], help="how the grid is written in the prompt")
    p.add_argument("--n", type=int, default=8, help="samples per puzzle")
    p.add_argument("--count", type=int, default=128, help="puzzles per config, unless the config says")
    p.add_argument("--offset", type=int, default=0, help="first augmentation index for train configs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dp-rank", type=int, default=0)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", default=None, help="JSONL prefix; a shard writes <out>.rank<i>.jsonl")
    p.add_argument("--merge", default=None, help="score the shards under this prefix instead")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--wandb-prefix", default="diag")
    p.add_argument("--wandb-step", type=int, default=0)
    args = p.parse_args()

    if args.merge:
        merge(args)
    else:
        if not args.out or not args.config:
            p.error("--out and at least one --config are required to generate")
        generate(args)


if __name__ == "__main__":
    main()

"""Can the model already solve Sudoku by prompting, with no fine-tuning?

Separate from `eval.py` on purpose. That script measures a fine-tuned model that emits 81 digits
directly, with decoding constrained to the digit alphabet. This one measures an instruct/reasoning
model the way such a model is meant to be used: its own chat template, unconstrained decoding, and
as much room to think as it wants -- then the grid is parsed out of whatever it wrote.

Constraining a reasoning model to 81 immediate digits would measure the opposite of what we want.

    torchrun --nproc-per-node 8 -m baselines.llm_sft.prompt_eval \
        --model Qwen/Qwen3.8-27B --limit 200 --max-new-tokens 8192
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.llm_sft.data import GRID, read_split, targets
from baselines.llm_sft.eval import is_legal_solution, log_to_wandb

INSTRUCTION = (
    "Solve this Sudoku puzzle.\n\n"
    "The puzzle is given as 81 digits in row-major order, where 0 marks an empty cell:\n"
    "{puzzle}\n\n"
    "Work it out, then give your final answer as exactly 81 digits on a single line, "
    "with no spaces or separators."
)


def parse_grid(text: str, reasoning: bool = False) -> np.ndarray | None:
    """Pull an 81-cell grid out of free-form model output, or return None.

    `reasoning=True` means the chat template opened a `<think>` block, so a completion without a
    closing `</think>` is a model that ran out of budget mid-thought and never gave an answer.
    Scraping digits out of unfinished reasoning would score it as a confident wrong answer, which
    is worse than useless -- it looks like a real 0% instead of a truncated run.
    """
    if reasoning and "</think>" not in text:
        return None
    tail = text.rsplit("</think>", 1)[-1]  # ignore the reasoning block if there is one

    # 81 consecutive digits, tolerating whitespace the model may have inserted
    runs = re.findall(r"\d{81}", re.sub(r"\s+", "", tail))
    if runs:
        return np.array([int(c) for c in runs[-1]], dtype=np.int64)

    # or nine lines of nine digits, which is how models usually like to print a grid
    rows = [re.sub(r"\D", "", line) for line in tail.splitlines()]
    rows = [r for r in rows if len(r) == 9]
    if len(rows) >= 9:
        return np.array([int(c) for r in rows[-9:] for c in r], dtype=np.int64)

    return None


def load(model_id: str, device):
    """Qwen3.8 is a conditional-generation (multimodal) checkpoint; try the text-LM class first."""
    last = None
    for loader in (AutoModelForCausalLM, None):
        try:
            if loader is None:
                from transformers import AutoModelForImageTextToText as loader  # type: ignore
            return loader.from_pretrained(model_id, dtype=torch.bfloat16, device_map=None).to(device).eval()
        except Exception as e:  # noqa: BLE001 - we genuinely want to try the next class
            last = e
    raise RuntimeError(f"could not load {model_id}: {last}")


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", default="test_hard")
    p.add_argument("--limit", type=int, default=200, help="reasoning generation is slow; keep this small")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32768,
                   help="reasoning budget; Sudoku-Extreme needs a lot of it")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--save", default=None, help="write per-puzzle outputs to this JSONL file")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    rank, world_size = 0, 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # prompts are equal length, but be explicit for generation
    model = load(args.model, device)

    rows = read_split(args.data_dir, args.split)[: args.limit]
    shard = rows[rank::world_size]
    gold = targets(shard)
    given = np.array([[int(c) for c in q] for q, _ in shard], dtype=np.int64)

    solved = cells = legal = kept = kept_total = parsed = truncated = 0
    records = []

    for start in range(0, len(shard), args.batch_size):
        batch = shard[start : start + args.batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION.format(puzzle=q)}],
                tokenize=False, add_generation_prompt=True,
            )
            for q, _ in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature or None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        generated = out[:, enc["input_ids"].shape[1]:]
        completions = tok.batch_decode(generated, skip_special_tokens=True)
        # A completion that used the whole budget without closing </think> never reached an answer.
        hit_cap = [int((row != (tok.pad_token_id or tok.eos_token_id)).sum()) >= args.max_new_tokens
                   for row in generated]

        for j, (text, (q, a)) in enumerate(zip(completions, batch)):
            reasoning = "<think>" in texts[j] or "</think>" in text or hit_cap[j]
            grid = parse_grid(text, reasoning=reasoning)
            if grid is None and hit_cap[j]:
                truncated += 1
            truth = gold[start + j]
            if grid is None:
                # unparseable counts as wrong, never as a crash
                records.append({"question": q, "answer": a, "parsed": False, "output": text})
                continue
            parsed += 1
            cells += int((grid == truth).sum())
            solved += int(bool((grid == truth).all()))
            legal += int(bool(is_legal_solution(grid[None])[0]))
            clue = given[start + j] > 0
            kept += int((grid[clue] == given[start + j][clue]).sum())
            kept_total += int(clue.sum())
            records.append({"question": q, "answer": a, "parsed": True,
                            "predicted": "".join(map(str, grid.tolist())), "output": text})

    t = torch.tensor([solved, cells, legal, kept, kept_total, parsed, truncated, len(shard)],
                     dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(t)
    solved, cells, legal, kept, kept_total, parsed, truncated, total = t.tolist()

    if args.save:
        with open(f"{args.save}.rank{rank}", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    if rank == 0:
        print(
            f"{args.model}  split={args.split}  n={int(total)}  max_new_tokens={args.max_new_tokens}\n"
            f"  parse_rate    = {parsed / total:.4f}  (reached a readable final answer)\n"
            f"  truncated     = {truncated / total:.4f}  (ran out of reasoning budget)\n"
            f"  exact_match   = {solved / total:.4f}\n"
            f"  cell_accuracy = {cells / (total * GRID):.4f}\n"
            f"  legal_grid    = {legal / total:.4f}\n"
            f"  clues_kept    = {kept / max(kept_total, 1):.4f}"
        )
        if not args.no_wandb:
            log_to_wandb(
                args.wandb_project, f"prompted_{args.model.split('/')[-1]}",
                {"exact_match": solved / total,
                 "cell_accuracy": cells / (total * GRID),
                 "legal_grid": legal / total,
                 "clues_kept": kept / max(kept_total, 1),
                 "parse_rate": parsed / total,
                 "truncated": truncated / total,
                 "solved_given_finished": solved / parsed if parsed else 0.0},
                {"model": args.model, "split": args.split, "n": int(total),
                 "max_new_tokens": args.max_new_tokens, "fine_tuned": False},
            )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

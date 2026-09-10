"""Greedy-decode a (fine-tuned) LLM on Sudoku and score it exactly like `train.py` does.

`exact_match` here is the same quantity as `eval/test_hard_exact_match` in the from-scratch runs:
the fraction of puzzles whose all 81 cells are right. Decoding is greedy and, by default,
constrained to the digits 1-9, so a model never loses points for emitting unparseable text --
this baseline is meant to measure Sudoku ability, not format compliance.

    torchrun --nproc-per-node 8 -m baselines.llm_sft.eval \
        --model "checkpoints/llm_sft_qwen3_1.7b exotic-bat/seed_1" --split test_hard

The system prompt has to match the one the checkpoint was trained with, so `train.py` writes it
to `sudoku_codec.json` in the checkpoint directory and this script reads it back.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from baselines.llm_sft.data import (
    DEFAULT_SYSTEM_PROMPT,
    GRID,
    Codec,
    build_codec,
    read_split,
    targets,
)

CODEC_META = "sudoku_codec.json"


def is_legal_solution(grids: np.ndarray) -> np.ndarray:
    """`[n]` bool: does each `[n, 81]` grid satisfy the row, column and box constraints?"""
    g = grids.reshape(-1, 9, 9)
    boxes = g.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    want = np.arange(1, 10)
    return (
        (np.sort(g, axis=2) == want).all(axis=(1, 2))
        & (np.sort(g, axis=1).transpose(0, 2, 1) == want).all(axis=(1, 2))
        & (np.sort(boxes, axis=2) == want).all(axis=(1, 2))
    )


class AllowOnly(LogitsProcessor):
    """Restrict sampling to a fixed set of token ids (the nine answer digits)."""

    def __init__(self, allowed_ids: list[int], vocab_size: int, device) -> None:
        self.block = torch.ones(vocab_size, dtype=torch.bool, device=device)
        self.block[torch.tensor(allowed_ids, device=device)] = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores.masked_fill(self.block[: scores.shape[-1]], torch.finfo(scores.dtype).min)


@torch.inference_mode()
def evaluate(
    model,
    codec: Codec,
    rows: list[tuple[str, str]],
    batch_size: int,
    device,
    rank: int = 0,
    world_size: int = 1,
    constrain_digits: bool = True,
    demos: Sequence[tuple[str, str]] = (),
    verbose: bool = False,
) -> dict[str, float]:
    """Greedy-decode `rows` (sharded across ranks) and return exact-match / per-cell accuracy.

    `demos` are in-context demonstrations for the no-fine-tuning control; empty for a fine-tuned
    model, which was trained on the zero-shot prompt.
    """
    was_training = model.training
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = True

    processors = None
    if constrain_digits:
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
        processors = LogitsProcessorList([AllowOnly(codec.answer_ids, vocab_size, device)])

    # Every rank must run the same number of forward passes or the collectives below (and FSDP's
    # internal all-gathers) deadlock, so give every rank an equal share.
    if world_size > 1:
        rows = rows[: (len(rows) // world_size) * world_size]
    shard = rows[rank::world_size]
    gold = targets(shard)
    given = np.array([[int(c) for c in q] for q, _ in shard], dtype=np.int64)
    correct_cells = solved = legal = clues_kept = clues_total = 0

    steps = range(0, len(shard), batch_size)
    if verbose and rank == 0:
        import tqdm

        steps = tqdm.tqdm(steps, desc="eval")

    for start in steps:
        batch = shard[start : start + batch_size]
        # Every prompt has the same length, so no padding and no left-padding subtleties.
        prompts = torch.tensor(
            [codec.prompt(question, demos) for question, _ in batch], dtype=torch.long, device=device
        )
        out = model.generate(
            input_ids=prompts,
            attention_mask=torch.ones_like(prompts),
            max_new_tokens=GRID,
            min_new_tokens=GRID,
            do_sample=False,
            num_beams=1,
            logits_processor=processors,
            pad_token_id=codec.eos_id,
        )
        preds = codec.cells(out[:, prompts.shape[1] :].cpu().numpy())
        truth = gold[start : start + len(batch)]
        correct_cells += int((preds == truth).sum())
        solved += int((preds == truth).all(axis=-1).sum())
        # Constrained decoding guarantees a well-formed grid, not a legal one. Tracking the two
        # separately says whether a wrong answer is a near-miss or a constraint violation.
        legal += int(is_legal_solution(preds).sum())
        clue = given[start : start + len(batch)] > 0
        clues_kept += int((preds[clue] == given[start : start + len(batch)][clue]).sum())
        clues_total += int(clue.sum())

    model.config.use_cache = use_cache
    if was_training:
        model.train()

    tallies = torch.tensor(
        [solved, correct_cells, legal, clues_kept, clues_total, len(shard)],
        dtype=torch.float64, device=device,
    )
    if world_size > 1:
        dist.all_reduce(tallies)
    solved, correct_cells, legal, clues_kept, clues_total, total = tallies.tolist()
    return {
        "exact_match": solved / total,
        "cell_accuracy": correct_cells / (total * GRID),
        "legal_grid": legal / total,
        "clues_kept": clues_kept / max(clues_total, 1),
        "n": int(total),
    }


def log_to_wandb(project: str, name: str, metrics: dict, config: dict) -> None:
    """Publish a one-shot evaluation as a W&B run so it sits alongside the training runs.

    `eval/test_hard_exact_match` goes to history so it charts against the training curves; the
    diagnostics go to the summary, where they show as table columns without adding chart series.
    """
    import coolname
    import wandb

    run = wandb.init(project=project, name=f"{name} {coolname.generate_slug(2)}",
                     group=name, config=config,
                     settings=wandb.Settings(x_disable_stats=True))
    split = config.get("split", "test_hard")
    run.log({f"eval/{split}_exact_match": metrics["exact_match"]}, step=0)
    for key, value in metrics.items():
        if key != "exact_match":
            run.summary[f"eval/{split}_{key}"] = value
    run.finish()


def load_causal_lm(model_id: str, dtype):
    """Load a text LM, tolerating checkpoints published as multimodal conditional-generation.

    Qwen3.8-27B ships as `Qwen3_5ForConditionalGeneration`, which `AutoModelForCausalLM` may
    refuse; fall through to the image-text-to-text auto class, which wraps the same text stack.
    """
    import inspect

    from transformers import AutoModelForCausalLM

    candidates = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText

        candidates.append(AutoModelForImageTextToText)
    except ImportError:
        pass

    errors = []
    for cls in candidates:
        try:
            return cls.from_pretrained(model_id, dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - try the next auto class
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"could not load {model_id}\n" + "\n".join(errors))


def supports_logits_to_keep(model) -> bool:
    """Not every wrapper plumbs `logits_to_keep` through to the LM head."""
    import inspect

    return "logits_to_keep" in inspect.signature(type(model).forward).parameters


def load_model(path: str, base: str | None, device):
    """Load a full fine-tune, a PEFT adapter directory, or an untouched pretrained model."""
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    return model.to(device).eval(), AutoTokenizer.from_pretrained(base or path)


def resolve_system_prompt(model_path: str, override: str | None) -> str:
    """A checkpoint must be evaluated with the prompt it was trained on."""
    if override is not None:
        return override
    meta = os.path.join(model_path, CODEC_META)
    if os.path.isdir(model_path) and os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)["system_prompt"]
    return DEFAULT_SYSTEM_PROMPT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint dir, adapter dir, or HF model id")
    p.add_argument("--base", default=None, help="base model id, for tokenizer lookup with adapters")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", default="test_hard")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N puzzles (0 = all)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fewshot", type=int, default=0, help="in-context demos, for the no-fine-tuning control")
    p.add_argument("--system-prompt", default=None,
                   help="override; defaults to the prompt recorded in the checkpoint")
    p.add_argument("--free-decode", action="store_true", help="do not constrain decoding to digits 1-9")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    rank, world_size = 0, 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    model, tokenizer = load_model(args.model, args.base, device)
    codec = build_codec(tokenizer, resolve_system_prompt(args.model, args.system_prompt))

    rows = read_split(args.data_dir, args.split)
    if args.limit:
        rows = rows[: args.limit]
    demos = read_split(args.data_dir, "train")[: args.fewshot] if args.fewshot else ()

    metrics = evaluate(
        model, codec, rows, args.batch_size, device,
        rank=rank, world_size=world_size,
        constrain_digits=not args.free_decode, demos=demos, verbose=True,
    )

    if rank == 0:
        print(
            f"{args.model}  split={args.split}  n={metrics['n']}  fewshot={args.fewshot}\n"
            f"  exact_match   = {metrics['exact_match']:.4f}\n"
            f"  cell_accuracy = {metrics['cell_accuracy']:.4f}\n"
            f"  legal_grid    = {metrics['legal_grid']:.4f}  (row/col/box constraints satisfied)\n"
            f"  clues_kept    = {metrics['clues_kept']:.4f}  (given cells reproduced)"
        )
        if not args.no_wandb:
            # checkpoints live at <group>/seed_N, so the basename alone would just say "seed_1"
            parts = args.model.rstrip("/").split(os.sep)
            tag = parts[-2] if len(parts) > 1 and re.fullmatch(r"seed_\d+", parts[-1]) else parts[-1]
            log_to_wandb(args.wandb_project, f"eval_{tag}",
                         metrics, {"model": args.model, "split": args.split,
                                   "fewshot": args.fewshot, "constrained": not args.free_decode})
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

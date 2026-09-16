"""RLVR (GRPO with a verifiable reward) on Sudoku, for a small thinking LLM.

The companion to `baselines/llm_reason`, whose answer-only SFT on the model's own traces collapsed
into "skip thinking, emit digits" (round 1: think tokens 32k -> 81, exact match -> 0). Here the
only signal is a verified reward on the final grid, so the shortcut earns nothing and the trace is
shaped by outcome rather than imitated. The completion is the model's whole turn -- thinking,
`</think>`, answer -- sampled at T=1 with no budget forcing: a trace that overruns
`--max-completion-length` is a zero-reward, loss-masked sample, so "finish in budget" is part of
what is learned.

Data: the same 1000 augmented training puzzles as every other arm (`common.py`), made easier
by a hint curriculum -- cells revealed from the solution until only `n_empty` blanks are left.
A *frontier* level is walked up towards the full puzzle as the success rate at the frontier
allows, and every batch mixes frontier puzzles with a replay of easier levels (`HintCurriculum`).
Without hints a small model never solves an Extreme puzzle, every GRPO group is all-zero and
there is no gradient; without replay, a frontier that is too hard starves the batch of signal and
the model forgets what it learned (v1 of this script did exactly that at ~43 blanks).

The reward is exact match only. v1 also paid 0.05 for a well-formed answer, and once the exact
reward was zero across a group, per-group std-normalisation turned that 0.05 into a full-strength
push towards "stop thinking, emit any grid" -- the collapse was complete within ten steps.

Full fine-tune, FSDP2, vLLM colocated on the same GPUs (sleep mode during the optimizer step).
Evaluation is external (`run.sh` alternates training chunks with `diag.py` on `test_hard`), so
this script only trains for `--max-steps` and resumes from the newest checkpoint if there is one.

    torchrun --nproc-per-node 8 -m baselines.llm_rlvr.train --model Qwen/Qwen3.5-4B \
        --output-dir checkpoints/llm_rlvr_qwen3.5_4b --max-steps 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback


def _shim_trl_for_vllm_029() -> None:
    """TRL 1.12 pins vllm<=0.27 and imports a server-mode NCCL helper that 0.29 renamed. Only the
    colocate path is used here, so give the import something to find rather than downgrade vLLM."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine

    if not hasattr(nccl_engine, "NCCLTrainerSendWeightsArgs"):
        nccl_engine.NCCLTrainerSendWeightsArgs = object


_shim_trl_for_vllm_029()
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from baselines.llm_reason.rollout import augmented_rows
from baselines.llm_rlvr.common import load_puzzles, make_prompt, puzzle_with_empties, reward_format
from baselines.llm_rlvr.common import reward_exact as _reward_exact
from baselines.llm_sft.eval import load_causal_lm


def reward_exact(completions, question, answer, empties, full, frontier, log_metric=None, **kwargs) -> list[float]:
    """`common.reward_exact`, plus the curriculum diagnostics TRL averages into the step's log.

    Every key is logged on every call: TRL gathers them across ranks in sorted-key order, so a
    key that one rank skips would mis-attribute the others."""
    r = _reward_exact(completions, question, answer)
    if log_metric is not None:
        sel = lambda flags: [x for x, f in zip(r, flags) if f]  # noqa: E731
        log_metric("curriculum/empties", float(np.mean(empties)))
        log_metric("curriculum/full_puzzle_frac", float(np.mean(full)))
        log_metric("curriculum/frontier_frac", float(np.mean(frontier)))
        log_metric("curriculum/frontier_exact", float(np.mean(sel(frontier))) if any(frontier) else 0.0)
        log_metric("curriculum/replay_exact", float(np.mean(sel([not f for f in frontier]))) if not all(frontier) else 0.0)
        log_metric("curriculum/full_exact", float(np.mean(sel(full))) if any(full) else 0.0)
    return r


class HintCurriculum:
    """Mutable difficulty -- blanks left in the puzzle -- shared by the dataset transform and the
    callback that moves it.

    A prompt is a *frontier* puzzle with `e` to `e + spread` blanks, or (with probability
    `replay`) a replay of an easier level drawn uniformly from `[floor, e)`. The frontier moves up
    when the success rate *on frontier prompts* clears `up`, and back down after two consecutive
    windows below `down`. Every rank sees the same gathered rates in `on_log`, so the level stays
    in lockstep without any communication of its own; `e >= 81` means the untouched puzzle.
    """

    FULL = 81  # more blanks than any puzzle has: no hints

    def __init__(self, start: int, spread: int, replay: float, floor: int, every: int, up: float, down: float,
                 step_up: int, step_down: int):
        self.e = start
        self.spread, self.replay, self.floor = spread, replay, floor
        self.every, self.up, self.down = every, up, down
        self.step_up, self.step_down = step_up, step_down
        self.history: list[float] = []  # frontier success per step
        self.low_windows = 0

    def draw(self, seed: int, index: int) -> tuple[int, bool]:
        """(blanks, is_frontier) for one prompt; a function of (seed, index, level) only."""
        rng = np.random.default_rng([seed, index, self.e, 7919])
        if self.e > self.floor and rng.random() < self.replay:
            return int(rng.integers(self.floor, self.e)), False
        return self.e + int(rng.integers(0, self.spread + 1)), True

    def update(self, frontier_exact: float) -> None:
        self.history.append(frontier_exact)
        if len(self.history) % self.every:
            return
        recent = float(np.mean(self.history[-self.every:]))
        if recent > self.up:
            self.e = min(self.FULL, self.e + self.step_up)
            self.low_windows = 0
        elif recent < self.down:
            self.low_windows += 1
            if self.low_windows >= 2:
                self.e = max(self.floor, self.e - self.step_down)
                self.low_windows = 0
        else:
            self.low_windows = 0

    def state(self) -> dict:
        return {"e": self.e, "history": self.history, "low_windows": self.low_windows}

    def load(self, d: dict) -> None:
        self.e, self.history, self.low_windows = d["e"], d["history"], d.get("low_windows", 0)


class CurriculumCallback(TrainerCallback):
    def __init__(self, curriculum: HintCurriculum):
        self.cur = curriculum

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "curriculum/frontier_exact" in logs:  # a training log, not an eval one
            self.cur.update(logs["curriculum/frontier_exact"])

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "curriculum.json")
            with open(path, "w") as f:
                json.dump(self.cur.state(), f)


def build_train_dataset(rows, seed: int, size: int, curriculum: HintCurriculum, layout: str) -> Dataset:
    ds = Dataset.from_dict({"index": list(range(size))})

    def transform(batch):
        out = {"prompt": [], "question": [], "answer": [], "empties": [], "full": [], "frontier": [], "index": []}
        for i in batch["index"]:
            _, q, a = augmented_rows(rows, range(i, i + 1), seed)[0]
            n_empty, is_frontier = curriculum.draw(seed, i)
            hq = puzzle_with_empties(q, a, n_empty, seed, i)
            out["prompt"].append(make_prompt(hq, layout))
            out["question"].append(hq)
            out["answer"].append(a)
            out["empties"].append(hq.count("0"))
            out["full"].append(hq == q)
            out["frontier"].append(is_frontier)
            out["index"].append(i)
        return out

    ds.set_transform(transform)
    return ds


def latest_checkpoint(output_dir: str) -> str | None:
    cks = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    cks = [c for c in cks if re.fullmatch(r"checkpoint-\d+", os.path.basename(c))
           and os.path.exists(os.path.join(c, "trainer_state.json"))]
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[1])) if cks else None


def patch_vllm_param_names(trainer: GRPOTrainer, tied_embeddings: bool) -> None:
    """vLLM loads Qwen3.5 as the multimodal `Qwen3_5ForConditionalGeneration`, whose weight
    mapper expects the checkpoint's `model.language_model.*` names; the text-only HF model we
    train has `model.*`. Rename on the way into vLLM, otherwise the sync silently misses.

    TRL pushes one tensor per `load_weights` call, and vLLM refuses a tied `lm_head.weight` that
    arrives without its `embed_tokens` in the same call -- so with tied embeddings the head is
    not pushed at all; it follows the embedding, which is."""
    gen = getattr(trainer, "vllm_generation", None)
    if gen is None:
        return
    original = gen._iter_named_params

    def renamed():
        for name, param in original():
            if tied_embeddings and name.endswith("lm_head.weight"):
                continue
            if name.startswith("model.") and not name.startswith("model.language_model."):
                name = "model.language_model." + name[len("model."):]
            yield name, param

    gen._iter_named_params = renamed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=32)
    p.add_argument("--per-device-batch", type=int, default=2, help="completions per micro-batch")
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL to the reference; 0 = no ref model")
    p.add_argument("--epsilon-high", type=float, default=0.28, help="DAPO clip-higher")
    p.add_argument("--loss-type", default="dapo")
    p.add_argument("--format-weight", type=float, default=0.0,
                   help="shaping reward for a well-formed answer; 0 (default) = exact match only, see module doc")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--no-sleep", action="store_true", help="disable vLLM sleep mode")
    p.add_argument("--layout", default="rows", choices=["rows", "line"], help="how the grid is written in the prompt")
    # curriculum: blanks left in a training puzzle
    p.add_argument("--empties-start", type=int, default=8, help="initial frontier")
    p.add_argument("--empties-spread", type=int, default=4, help="frontier prompts span [e, e + spread]")
    p.add_argument("--replay", type=float, default=0.4, help="fraction of prompts drawn from easier levels")
    p.add_argument("--empties-floor", type=int, default=6, help="easiest replay level")
    p.add_argument("--curriculum-every", type=int, default=4, help="steps between level updates")
    p.add_argument("--curriculum-up", type=float, default=0.3, help="frontier success above which the frontier moves up")
    p.add_argument("--curriculum-down", type=float, default=0.03, help="... below which (twice in a row) it moves down")
    p.add_argument("--empties-step-up", type=int, default=3)
    p.add_argument("--empties-step-down", type=int, default=2)
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--wandb-group", default=None)
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % (world * args.per_device_batch):
        raise ValueError("prompts_per_step * num_generations must be divisible by world * per_device_batch")
    grad_accum = completions_per_step // (world * args.per_device_batch)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_causal_lm(args.model, torch.float32)  # fp32 master weights; bf16 compute under FSDP
    declared = getattr(model, "_no_split_modules", None) or []
    present = {type(m).__name__ for m in model.modules()}
    layers = [n for n in declared if n in present]

    curriculum = HintCurriculum(args.empties_start, args.empties_spread, args.replay, args.empties_floor,
                                args.curriculum_every, args.curriculum_up, args.curriculum_down,
                                args.empties_step_up, args.empties_step_down)
    resume = latest_checkpoint(args.output_dir)
    if resume and os.path.exists(os.path.join(resume, "curriculum.json")):
        curriculum.load(json.load(open(os.path.join(resume, "curriculum.json"))))

    rows = load_puzzles(args.data_dir, "train")
    train_ds = build_train_dataset(rows, args.seed, size=1_000_000, curriculum=curriculum, layout=args.layout)

    run_name = args.run_name or os.path.basename(args.output_dir.rstrip("/"))
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_group:
        os.environ.setdefault("WANDB_RUN_GROUP", args.wandb_group)

    config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        chat_template_kwargs={"enable_thinking": True},
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        beta=args.beta,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        mask_truncated_completions=True,
        reward_weights=[1.0, args.format_weight] if args.format_weight else None,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=not args.no_sleep,
        vllm_max_model_length=args.max_completion_length + 512,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "version": 2,
            "transformer_layer_cls_to_wrap": layers,
            "state_dict_type": "FULL_STATE_DICT",
            "reshard_after_forward": True,
        },
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_exact, reward_format] if args.format_weight else [reward_exact],
        args=config,
        train_dataset=train_ds,
        callbacks=[CurriculumCallback(curriculum)],
    )
    text_config = getattr(model.config, "text_config", model.config)
    patch_vllm_param_names(trainer, bool(getattr(text_config, "tie_word_embeddings", False)))
    if trainer.accelerator.is_main_process:
        print(f"[rlvr] world={world} grad_accum={grad_accum} completions/step={completions_per_step} "
              f"fsdp over {layers} empties={curriculum.e} resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    # The last checkpoint is the deliverable (`--save-steps` should divide `--max-steps`); a separate
    # end-of-run save is not attempted because process teardown with colocated vLLM is not reliable.
    if trainer.accelerator.is_main_process:
        print(f"[rlvr] done at step {trainer.state.global_step}, empties={curriculum.e}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

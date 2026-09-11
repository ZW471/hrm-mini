"""Answer-only SFT on the model's own reasoning traces.

Each example is `[chat prompt | <think> trace </think> | 81 gold digits | <|im_end|>]`, where the
trace is what the model itself produced in `rollout.py`. The loss covers the 81 digits and the
end-of-turn token only: there is no reference reasoning for Sudoku-Extreme, so the trace is
context, not target. What the model learns is p(correct grid | prompt, its own thoughts); the
thoughts move only indirectly, through the shared weights. Repeating rollout -> train keeps the
traces on-policy (`run.sh`).

Sequences are up to ~33k tokens and vary in length, so the per-device batch is one sequence and
the global batch comes from gradient accumulation; with the answer always at the end,
`logits_to_keep=82` slices the supervised positions without a per-sample index.

    torchrun --nproc-per-node 8 -m baselines.llm_reason.train \
        --model Qwen/Qwen3.8-27B --rollouts rollouts/round1/train --output-dir checkpoints/x/round1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.loss.loss_utils import fixed_cross_entropy

from baselines.llm_reason.rollout import prompt_ids
from baselines.llm_sft.data import GRID
from baselines.llm_sft.eval import load_causal_lm

N_TARGET = GRID + 1  # 81 digits + <|im_end|>


def load_examples(tok, prefix: str, max_len: int) -> tuple[list[dict], dict]:
    paths = sorted(glob.glob(f"{prefix}.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rollouts match {prefix}.rank*.jsonl")
    examples, dropped, lengths = [], 0, []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            digits = [tok.convert_tokens_to_ids(c) for c in r["answer"]]
            ids = prompt_ids(tok, r["question"]) + r["think_ids"] + r["close_ids"] + digits + [tok.eos_token_id]
            if len(ids) > max_len:
                dropped += 1
                continue
            lengths.append(len(ids))
            examples.append({"input_ids": ids})
    stats = {"n": len(examples), "dropped": dropped,
             "len_mean": sum(lengths) / max(len(lengths), 1), "len_max": max(lengths, default=0)}
    return examples, stats


def collate(features: list[dict]) -> dict[str, torch.Tensor]:
    assert len(features) == 1, "one variable-length sequence per device; use --grad-accum"
    ids = torch.tensor([features[0]["input_ids"]], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[:, -N_TARGET:] = ids[:, -N_TARGET:]  # so Trainer counts 82 loss tokens per sequence
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


class AnswerOnlyTrainer(Trainer):
    """Loss on the last 82 tokens only; W&B metrics under `train.py`'s names, offset by round."""

    def __init__(self, *args, run=None, step_offset: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run, self.step_offset = run, step_offset
        self._reset()

    def _reset(self) -> None:
        self._cells = self._exact = self._seen = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids = inputs["input_ids"]
        outputs = model(input_ids=ids, attention_mask=inputs["attention_mask"],
                        logits_to_keep=N_TARGET + 1, use_cache=False)
        # logits[:, i] predicts token i + 1: hidden states [-83, -1) predict the last 82 tokens.
        logits = outputs.logits[:, :-1]
        labels = ids[:, -N_TARGET:]
        loss = fixed_cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), num_items_in_batch
        )
        with torch.no_grad():
            correct = logits[:, :GRID].argmax(-1) == labels[:, :GRID]
            self._cells += correct.float().mean().item()
            self._exact += correct.all().float().item()
            self._seen += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time)
        if self.run is None:
            return
        payload: dict[str, float] = {}
        if "loss" in logs:
            payload["train/loss"] = logs["loss"]
            if self._seen:
                payload["train/per_position_accuracy"] = self._cells / self._seen
                payload["train/exact_match"] = self._exact / self._seen
            self._reset()
        if "learning_rate" in logs:
            payload["train/lr"] = logs["learning_rate"]
        if payload:
            self.run.log(payload, step=self.step_offset + self.state.global_step)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint to continue from")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    p.add_argument("--rollouts", required=True, help="JSONL prefix written by rollout.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-len", type=int, default=34_000, help="drop longer sequences")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2, help="global batch = 8 GPUs x this")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--optim", default="adamw_torch_8bit")
    p.add_argument("--liger", action="store_true")
    p.add_argument("--no-fsdp", action="store_true")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--no-save", action="store_true", help="smoke tests: skip the 150GB checkpoint")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--wandb-id", default=None, help="resume this W&B run (shared across rounds)")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--step-offset", type=int, default=0, help="optimizer steps done in earlier rounds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size = 0, 1
    if "LOCAL_RANK" in os.environ:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    examples, stats = load_examples(tok, args.rollouts, args.max_len)
    random.Random(args.seed).shuffle(examples)

    # fp32 master weights with bf16 autocast, as in llm_sft: at lr 1e-5 the updates are below
    # bf16's resolution of the weights.
    model = load_causal_lm(args.model, torch.float32)
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("visual", "vision_tower", "vision_model")):
            param.requires_grad_(False)

    global_batch = world_size * args.grad_accum
    steps = len(examples) // global_batch
    run = None
    if rank == 0:
        print(f"[{args.output_dir}] {args.model}\n"
              f"  rollouts : {stats['n']} sequences from {args.rollouts} ({stats['dropped']} dropped "
              f"> {args.max_len}), mean len {stats['len_mean']:.0f}, max {stats['len_max']}\n"
              f"  steps    : {steps} at global batch {global_batch}, lr {args.lr}", flush=True)
        if args.wandb_id:
            import wandb

            run = wandb.init(project=args.wandb_project, id=args.wandb_id, resume="allow",
                             group=args.wandb_group, settings=wandb.Settings(x_disable_stats=True))
            run.log({f"rollout/{k}": v for k, v in stats.items()}, step=args.step_offset)

    fsdp, fsdp_config, accelerator_config = "", None, None
    if not args.no_fsdp:
        # See llm_sft/train.py for why: sync every micro-batch keeps gradients sharded, and only
        # the layer classes actually present can be wrap targets.
        accelerator_config = {"gradient_accumulation_kwargs": {"sync_each_batch": True}}
        present = {type(m).__name__ for m in model.modules()}
        layers = [n for n in (getattr(model, "_no_split_modules", None) or []) if n in present]
        fsdp = "full_shard auto_wrap"
        fsdp_config = {"transformer_layer_cls_to_wrap": layers, "state_dict_type": "FULL_STATE_DICT",
                       "use_orig_params": True, "limit_all_gathers": True}

    trainer = AnswerOnlyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            seed=args.seed,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=args.warmup_steps,
            weight_decay=0.0,
            adam_beta2=args.beta2,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=args.logging_steps,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            ddp_find_unused_parameters=False,
            fsdp=fsdp,
            fsdp_config=fsdp_config,
            accelerator_config=accelerator_config,
            optim=args.optim,
            use_liger_kernel=args.liger,
        ),
        train_dataset=examples,
        data_collator=collate,
        processing_class=tok,
        run=run,
        step_offset=args.step_offset,
    )
    trainer.train()
    if args.no_save:
        return
    trainer.save_model(args.output_dir)
    if rank == 0:
        with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
            json.dump({"steps": steps, "step_offset": args.step_offset, "rollouts": args.rollouts,
                       "from": args.model, "stats": stats, "args": vars(args)}, f, indent=2)
        print(f"saved to {args.output_dir} after {steps} steps", flush=True)
    if run is not None:
        run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

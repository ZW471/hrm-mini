"""Fine-tune a pretrained LLM on Sudoku-Extreme-1k, as a baseline for the from-scratch models.

Same 1000 training puzzles and the same on-the-fly band/stack/digit augmentation the HRM /
recurrent-transformer / AR-transformer runs get (`dataset.sudoku.shuffle_sudoku`), the same
81-cell exact-match metric, and -- deliberately -- the same W&B project, run/group naming and
metric keys as `train.py`, so these runs overlay the existing baselines chart for chart.

The training budget is set in *augmented samples seen* (`--num-samples`), which is the unit
`train.py` effectively varies through `data.repeat x epochs`.

Full fine-tune on 8 GPUs:

    torchrun --nproc-per-node 8 -m baselines.llm_sft.train \
        --model Qwen/Qwen3-1.7B-Base --run-name llm_sft_qwen3_1.7b

LoRA on a bigger model:

    torchrun --nproc-per-node 8 -m baselines.llm_sft.train \
        --model Qwen/Qwen3-8B-Base --lora-r 32 --batch-size 8 --grad-accum 8 \
        --run-name llm_lora_qwen3_8b

`train.py` runs every seed in one process and so shares one W&B group across them. Here each
seed is its own invocation, so pass `--group` to put several seeds in one group:

    for s in 1 2 3; do torchrun ... --seed $s --group "llm_sft_qwen3_1.7b brave-otter"; done
"""

from __future__ import annotations

import argparse
import json
import os

# The 152k-vocab logits tensor is the peak allocation here and it churns; expandable segments
# keep it from fragmenting the caching allocator. Must be set before the CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import coolname
import torch
import torch.distributed as dist
import wandb
from transformers.loss.loss_utils import fixed_cross_entropy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from baselines.llm_sft.data import (
    DEFAULT_SYSTEM_PROMPT,
    GRID,
    SudokuSFTDataset,
    build_codec,
    collate,
    read_split,
)
from baselines.llm_sft.eval import CODEC_META, evaluate, load_causal_lm, supports_logits_to_keep


class SudokuTrainer(Trainer):
    """`Trainer` that reports `train.py`'s metric names, so W&B charts line up across baselines.

    The logged set is exactly `train.py`'s -- `train/loss`, `train/lr`,
    `train/per_position_accuracy`, `train/exact_match`, `eval/test_hard_exact_match` -- and
    nothing else, so the charts overlay without stray series. The first two training metrics are
    teacher-forced off the training batch, reproduced here from the same logits. Note that teacher-forced exact match badly overstates
    an autoregressive model -- the honest number is the greedy-decoded `eval/test_hard_exact_match`
    logged by `GenerationEval`, exactly as for `ar_param_matched`.
    """

    def __init__(self, *args, codec, run=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.codec = codec
        self.run = run
        self._keep: torch.Tensor | None = None
        self._slice_logits = supports_logits_to_keep(self.model)
        self._reset_batch_metrics()

    def _reset_batch_metrics(self) -> None:
        self._cells = 0.0
        self._exact = 0.0
        self._seen = 0.0

    def _keep_index(self, device) -> torch.Tensor:
        """Positions whose predictions carry loss: the 81 solution cells plus the EOS.

        logits[:, i] predicts token i + 1, so the cells at [p, p + 81) and the EOS at p + 81 are
        predicted from hidden states [p - 1, p + 81). The other ~60% of positions are prompt and
        contribute nothing, so running `lm_head` (a 152k-way projection) on them is pure waste.
        """
        if self._keep is None or self._keep.device != device:
            p = self.codec.prompt_len
            self._keep = torch.arange(p - 1, p + GRID, device=device)
        return self._keep

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        p = self.codec.prompt_len
        # `logits_to_keep` takes an index tensor and slices the hidden states before `lm_head`,
        # so logits are only ever materialised for the 82 supervised positions.
        if self._slice_logits:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                logits_to_keep=self._keep_index(labels.device),
                use_cache=False,
            )
            logits = outputs.logits  # [batch, 82, vocab]
        else:
            # Wrapper does not plumb `logits_to_keep`; pay for the full logits and slice after.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"], use_cache=False)
            logits = outputs.logits[:, p - 1 : p + GRID]
        # Mirrors transformers' ForCausalLMLoss, including its num_items_in_batch handling, so
        # gradients match what the stock path produces.
        loss = fixed_cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels[:, p:].reshape(-1),
            num_items_in_batch,
        )

        with torch.no_grad():
            preds = logits[:, :GRID].argmax(dim=-1)
            correct = preds == labels[:, p : p + GRID]
            self._cells += correct.float().sum().item() / GRID
            self._exact += correct.all(dim=-1).float().sum().item()
            self._seen += correct.shape[0]
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time)  # console only; W&B is driven from here instead
        if self.run is None:
            return

        payload: dict[str, float] = {}
        if "loss" in logs:
            payload["train/loss"] = logs["loss"]
            if self._seen:
                payload["train/per_position_accuracy"] = self._cells / self._seen
                payload["train/exact_match"] = self._exact / self._seen
            self._reset_batch_metrics()
        if "learning_rate" in logs:
            payload["train/lr"] = logs["learning_rate"]
        payload.update({k: v for k, v in logs.items() if k.startswith("eval/")})

        if payload:
            self.run.log(payload, step=self.state.global_step)


class GenerationEval(TrainerCallback):
    """Periodically greedy-decode a held-out subsample and log `eval/test_hard_exact_match`.

    This is the metric `train.py` selects checkpoints on, and the only one comparable across the
    encoder-only and autoregressive baselines.
    """

    def __init__(self, codec, rows, batch_size: int, every: int, rank: int, world_size: int,
                 best_dir: str | None = None) -> None:
        self.codec, self.rows = codec, rows
        self.batch_size, self.every = batch_size, every
        self.rank, self.world_size = rank, world_size
        self.best_dir = best_dir
        self.best = -1.0
        self.trainer: Trainer | None = None

    def _run(self, model) -> None:
        metrics = evaluate(
            model, self.codec, self.rows, self.batch_size, model.device,
            rank=self.rank, world_size=self.world_size,
        )
        if self.trainer is None:
            return
        self.trainer.log({"eval/test_hard_exact_match": metrics["exact_match"]})
        # train.py keeps a `best.pt` selected on this metric; without it a run that overfits
        # leaves only the degraded final weights behind. All ranks must call save_model so
        # FSDP can gather a full state dict.
        if self.best_dir is not None and metrics["exact_match"] > self.best:
            self.best = metrics["exact_match"]
            self.trainer.save_model(self.best_dir)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every > 0 and state.global_step % self.every == 0:
            self._run(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run(model)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data-dir", default="downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--run-name", default=None)
    p.add_argument("--group", default=None,
                   help="W&B group / checkpoint dir; defaults to '<run-name> <random-slug>'")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--num-samples", type=int, default=2_000_000,
                   help="augmented puzzles seen over the whole run (the training budget)")
    # 32 x 164 positions x 152k vocab is already ~2 GiB of logits per copy, which is what caps
    # the micro-batch on an 80 GiB card; accumulation buys back the global batch size.
    p.add_argument("--batch-size", type=int, default=32, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=2)

    # tuned_hrm / ar_param_matched: lr 1e-4, 2000 warmup steps, then held constant
    # (their `lr_min_ratio: 1.0` means no decay).
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="constant_with_warmup",
                   choices=["constant_with_warmup", "cosine", "linear"])
    p.add_argument("--warmup-steps", type=int, default=2000)
    # NOT matched to tuned_hrm's `weight_decay: 1.0`. That is a sane regulariser for a model
    # trained from scratch, but decoupled decay at lr 1e-4 shrinks every weight by a factor
    # (1 - 1e-4)^83333 ~ 2e-4 over the run, which would erase the pretrained initialisation
    # this baseline exists to test.
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.95)

    p.add_argument("--lora-r", type=int, default=0, help="0 = full fine-tune")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                   help="weight dtype; default fp32 for full fine-tunes, bf16 for LoRA")
    p.add_argument("--no-augment", action="store_true", help="ablate the augmentation")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                   help="instruction prefixed to every puzzle ('' for none)")

    p.add_argument("--eval-every", type=int, default=500, help="steps between generation evals (0 = off)")
    p.add_argument("--eval-limit", type=int, default=1024, help="test_hard puzzles per in-training eval")
    p.add_argument("--eval-batch-size", type=int, default=128)

    p.add_argument("--train-vision", action="store_true",
                   help="also train the vision tower of a multimodal checkpoint (pointless here)")
    p.add_argument("--optim", default="adamw_torch",
                   help="e.g. adamw_8bit -- 8-bit Adam frees ~20GB/GPU at 27B, buying micro-batch")
    p.add_argument("--liger", action="store_true",
                   help="Liger fused kernels; its fused linear+CE avoids materialising the "
                        "248k-vocab logits, which is the largest activation here")
    p.add_argument("--fsdp", action="store_true",
                   help="shard params/grads/optimizer across GPUs (needed above ~8B)")
    p.add_argument("--no-save-best", action="store_true",
                   help="do not keep the best-by-eval checkpoint (train.py keeps one)")
    p.add_argument("--wandb-project", default="sudoku")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"llm_sft_{args.model.split('/')[-1]}"
    # Mirrors train.py: one name shared by every seed, with a slug to keep reruns distinct.
    group_name = args.group or os.environ.get(
        "MLP_TASK_NAME", f"{run_name} {coolname.generate_slug(2)}"
    )

    rank, world_size = 0, 1
    if "LOCAL_RANK" in os.environ:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codec = build_codec(tokenizer, args.system_prompt)  # asserts one token per cell

    train_rows = read_split(args.data_dir, "train")
    eval_rows = read_split(args.data_dir, "test_hard")[: args.eval_limit] if args.eval_every else []
    train_dataset = SudokuSFTDataset(
        train_rows, codec, args.num_samples, augment=not args.no_augment, seed=args.seed
    )

    # Full fine-tuning keeps fp32 master weights (Trainer autocasts the forward pass to bf16):
    # at lr 1e-5 the updates are close to bf16's relative precision, so pure-bf16 weights lose
    # them. LoRA freezes the base model, so bf16 weights there cost nothing.
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if args.lora_r else torch.float32)
    model = load_causal_lm(args.model, dtype)
    model.config.use_cache = not args.gradient_checkpointing

    # Multimodal checkpoints (Qwen3.8-27B) carry a vision tower that never sees a gradient here,
    # because this task is text-only and we never pass pixel_values. Left trainable it confuses
    # FSDP/DDP's unused-parameter handling and carries dead optimizer state, so freeze it.
    frozen = 0
    if not args.train_vision:
        for name, param in model.named_parameters():
            if any(tag in name for tag in ("visual", "vision_tower", "vision_model")):
                param.requires_grad_(False)
                frozen += param.numel()

    if args.lora_r:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))

    global_batch = args.batch_size * world_size * args.grad_accum
    steps = args.num_samples // global_batch
    # train.py names the checkpoint dir after the group too, e.g. "ar_param_matched noisy-kingfisher"
    output_dir = os.path.join("checkpoints", group_name, f"seed_{args.seed}")

    run = None
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[{group_name}] {args.model}\n"
            f"  frozen (vision)  : {frozen / 1e6:.1f}M\n"
            f"  trainable params : {trainable / 1e6:.1f}M ({str(dtype).replace('torch.', '')})"
            f"{' LoRA r=%d' % args.lora_r if args.lora_r else ' full fine-tune'}\n"
            f"  seq len          : {codec.total_len} tokens ({codec.prompt_len} prompt / 82 supervised,"
            f" instruction {len(codec.prefix_ids)})\n"
            f"  budget           : {args.num_samples:,} augmented samples"
            f" = {args.num_samples * codec.total_len / 1e9:.2f}B tokens\n"
            f"  optimizer steps  : {steps:,} at global batch {global_batch}\n"
            f"  output           : {output_dir}",
            flush=True,
        )
        if not args.no_wandb:
            run = wandb.init(
                project=args.wandb_project,
                name=group_name,
                group=group_name,
                config=vars(args) | {
                    "seed": args.seed,
                    "world_size": world_size,
                    "global_batch_size": global_batch,
                    "optimizer_steps": steps,
                    "trainable_params": trainable,
                    "seq_len": codec.total_len,
                },
                settings=wandb.Settings(x_disable_stats=True),
            )
            # Bare log_code() walks the venvs too (~19k files); keep it to project sources.
            run.log_code(
                ".",
                include_fn=lambda path: path.endswith((".py", ".yaml"))
                and not any(part in path for part in ("/.venv", "/wandb/", "/checkpoints/", "/outputs/")),
            )

    callbacks = []
    if args.eval_every:
        callbacks.append(GenerationEval(
            codec, eval_rows, args.eval_batch_size, args.eval_every, rank, world_size,
            best_dir=None if args.no_save_best else os.path.join(output_dir, "best"),
        ))

    fsdp, fsdp_config, accelerator_config = "", None, None
    if args.fsdp:
        # Under FSDP, accelerate runs accumulation micro-steps inside no_sync(), which makes FSDP
        # skip the reduce-scatter and hold FULL unsharded gradients -- 108GB of fp32 at 27B, an
        # instant OOM. Syncing each micro-batch keeps gradients sharded at 1/world_size.
        accelerator_config = {"gradient_accumulation_kwargs": {"sync_each_batch": True}}
        # `_no_split_modules` is declared for the full checkpoint, but AutoModelForCausalLM may
        # instantiate only the text stack (Qwen3.8-27B loads text-only, with no vision blocks),
        # and FSDP errors on a wrap target that is not present. Keep only the classes that exist.
        declared = getattr(model, "_no_split_modules", None) or []
        present = {type(m).__name__ for m in model.modules()}
        layers = [name for name in declared if name in present]
        if not layers:
            raise ValueError(
                f"{args.model}: none of its _no_split_modules {declared} are in the loaded model; "
                "name the transformer layer class manually"
            )
        fsdp = "full_shard auto_wrap"
        fsdp_config = {
            "transformer_layer_cls_to_wrap": list(layers),
            "state_dict_type": "FULL_STATE_DICT",  # so save_model writes a normal checkpoint
            # Keep fp32 shards and cast to bf16 for compute (bf16=True drives the mixed-precision
            # policy), which preserves the precision parity with the from-scratch baselines.
            "use_orig_params": True,
            "limit_all_gathers": True,
        }
        if rank == 0:
            print(f"  FSDP             : full_shard auto_wrap over {list(layers)}", flush=True)

    trainer = SudokuTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            run_name=group_name,
            seed=args.seed,
            num_train_epochs=1,  # the dataset already encodes the full budget
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_scheduler,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            adam_beta2=args.beta2,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=25,
            save_strategy="no",  # saved once at the end
            report_to="none",  # SudokuTrainer.log drives W&B, to control the metric names
            remove_unused_columns=False,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            fsdp=fsdp,
            fsdp_config=fsdp_config,
            accelerator_config=accelerator_config,
            optim=args.optim,
            use_liger_kernel=args.liger,
        ),
        train_dataset=train_dataset,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=callbacks,
        codec=codec,
        run=run,
    )
    for cb in callbacks:
        cb.trainer = trainer

    trainer.train()
    trainer.save_model(output_dir)
    if rank == 0:
        # A checkpoint is only meaningful with the prompt it was trained on; record it so
        # eval.py cannot silently evaluate against a different one.
        with open(os.path.join(output_dir, CODEC_META), "w") as f:
            json.dump({"system_prompt": codec.system_prompt}, f, indent=2)
    if rank == 0:
        print(f"[{group_name}] saved to {output_dir}", flush=True)
    if run is not None:
        run.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

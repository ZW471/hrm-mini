# Fine-tuned-LLM Sudoku baseline

Answers "what does a pretrained LLM do on this task?" under conditions matched to the
from-scratch models in `arch/`: the same 1000 training puzzles from `sudoku-extreme-1k`, the
same band/stack/digit augmentation (`dataset.sudoku.shuffle_sudoku`), the same 81-cell
exact-match metric, and the same W&B project (`sudoku`) so the curves overlay.

## Environment

Kept in a separate venv so the pinned reproduction env in `.venv/` is untouched:

```bash
uv venv .venv-llm --python 3.13
VIRTUAL_ENV=.venv-llm uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.14.0' transformers trl peft accelerate datasets huggingface_hub wandb numpy
.venv-llm/bin/hf auth login
```

## Encoding

Each puzzle is a fixed-length 164-token sequence, `[81 puzzle digits | \n | 81 solution digits | EOS]`,
with the loss masked off the prompt. **One token per cell** is the property that makes per-cell
accuracy and exact match mean the same thing they do in `train.py`; `data.build_codec` raises if
the tokenizer violates it. Qwen's pre-tokenizer splits numbers into single digits, so the
sequence we build is byte-identical to what the tokenizer itself produces — pick a different
model family and check this first. Fixed length also means no padding anywhere, and no
left-padding subtleties at generation time.

## Train

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-llm/bin/torchrun --nproc-per-node 8 \
  -m baselines.llm_sft.train --model Qwen/Qwen3-1.7B-Base --run-name llm_sft_qwen3_1.7b
```

LoRA on a larger model, to show that scale alone does not fix it:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-llm/bin/torchrun --nproc-per-node 8 \
  -m baselines.llm_sft.train --model Qwen/Qwen3-8B-Base --lora-r 32 \
  --batch-size 8 --grad-accum 8 --run-name llm_lora_qwen3_8b
```

The training budget is `--num-samples`, in augmented puzzles seen — the unit `train.py` varies
through `data.repeat x epochs`. Note it cannot be matched to the from-scratch runs: `tuned_hrm`
and `ar_param_matched` see ~64M augmented samples, which at 164 tokens is ~10B tokens through a
1.7B model. Report the budget you used rather than claiming a matched comparison.

Per-device batch is capped by the logits tensor (32 x 164 positions x 152k vocab), not by the
weights; use `--grad-accum` for the global batch and `--gradient-checkpointing` if you go bigger.

## Evaluate

Training logs `eval/test_hard_exact_match` on a 1024-puzzle subsample every `--eval-every`
steps, by greedy decoding — *not* teacher-forced token accuracy, which badly overstates ability
(a model can score well per-token and solve nothing). For the final number, run the full 20k:

```bash
HF_HUB_OFFLINE=1 .venv-llm/bin/torchrun --nproc-per-node 8 -m baselines.llm_sft.eval \
  --model checkpoints/llm_sft_qwen3_1.7b/seed_1 --split test_hard
```

Decoding is greedy and constrained to digits 1-9 by default, so a model is never penalised for
unparseable output — this measures Sudoku ability, not format compliance. `--free-decode` drops
the constraint and is worth reporting once, as the gap is the format-compliance cost.

Controls that cost minutes and pre-empt the obvious reviewer questions:

```bash
# no fine-tuning at all, with in-context demonstrations
... -m baselines.llm_sft.eval --model Qwen/Qwen3-1.7B-Base --fewshot 8 --limit 1024
# augmentation ablation
... -m baselines.llm_sft.train --no-augment --run-name llm_sft_qwen3_1.7b_noaug
```

## Files

- `data.py` — token codec (with the one-token-per-cell assertion), augmented dataset, collator
- `train.py` — `Trainer` loop, optional LoRA, periodic generation eval
- `eval.py` — batched greedy decoding, exact-match / cell-accuracy scoring, few-shot control

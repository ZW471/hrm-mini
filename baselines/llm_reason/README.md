# Reasoning-LLM Sudoku baseline

The companion to `baselines/llm_sft`, which fine-tunes an LLM to emit the 81-digit grid directly.
Here the model is allowed to *think first*: up to 32k tokens of free-form reasoning before the
answer. There is no reference reasoning for Sudoku-Extreme, so the trace is the model's own and
only the answer is supervised.

## Protocol

Each round:

1. **Roll out** (`rollout.py`, vLLM): for a fresh batch of augmented training puzzles, sample a
   thinking trace from the current model (chat template, thinking on, the model's recommended
   sampling: T=1.0, top-p 0.95, top-k 20) until `</think>` or the 32768-token budget. Then close
   the think block -- forcibly if the model did not -- and greedily decode the answer. The forced
   close is s1-style budget forcing; without it, a trace that overruns the budget is a non-answer,
   and on Sudoku-Extreme most do.
2. **Train** (`train.py`): SFT on `[prompt | own trace | </think> | gold 81 digits | <|im_end|>]`
   with the loss on the last 82 tokens only. Full fine-tune, fp32 master weights, FSDP, 8-bit Adam,
   gradient checkpointing, one ~33k-token sequence per GPU, global batch 16, lr 1e-5.
3. **Evaluate**: step 1 on a fixed 512-puzzle subset of `test_hard`, logged as
   `eval/test_hard_exact_match` (the metric the other baselines use) plus diagnostics: how often
   the model closed its thinking by itself, trace length, legal-grid rate, clue preservation.

Round 0 evaluates the untouched model under the same protocol, which is the prompting-only control.
Repeating the loop keeps the traces on-policy: the model always trains on reasoning it would
actually produce.

What this learns is p(correct grid | prompt, own reasoning). The reasoning itself gets no direct
gradient -- there is nothing to supervise it with -- and moves only through the shared weights.
The budget is deliberately not matched to the from-scratch runs: 4 rounds x 1024 augmented puzzles
= ~4 views of each training puzzle, against ~64k views for `tuned_hrm`, because each view here
costs a 33k-token rollout and a 33k-token training sequence (the untouched model closes its
thinking within 32k tokens on only ~6% of test puzzles, so nearly every sequence is full length).

## Run

```bash
tmux new -s llm_reason 'bash baselines/llm_reason/run.sh 2>&1 | tee logs/llm_reason.log'
```

`ROUNDS`, `PER_ROUND`, `EVAL_N`, `THINK`, `LR`, `GRAD_ACCUM` are environment overrides. Every stage
leaves a marker, so a killed run resumes from the last finished stage. Only the two newest fp32
checkpoints (150 GB each) are kept.

Rollouts run as two tensor-parallel-4 vLLM replicas in `.venv-vllm`; training runs in
`.venv-llm`. The driver 535 on this box supports CUDA 12.2, so the vLLM venv uses the
`+cu129` wheels rather than the default CUDA 13 build:

```bash
uv venv .venv-vllm --python 3.13
VIRTUAL_ENV=.venv-vllm uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  'vllm==0.29.0+cu129' 'torch==2.13.0+cu129' wandb numpy datasets
VIRTUAL_ENV=.venv-llm uv pip install flash-linear-attention   # fla kernels for the 48 DeltaNet layers
```

## Files

- `rollout.py` — vLLM think-then-answer generation for training data and for evaluation; `--merge` scores shards
- `train.py` — answer-only SFT on rollouts, variable length, batch 1 per device
- `run.sh` — the round loop, W&B bookkeeping, checkpoint rotation

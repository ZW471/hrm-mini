# Status report — LLM baselines for the HRM rebuttal (paused 2026-09-20 05:10 UTC)

Everything is stopped: no tmux sessions, no processes, GPUs free. This note is what you need
to resume. The results write-up with the figure is `RESULTS.md` next to this file.

## Headline

A Qwen3.5-4B student fine-tuned on **993 verified teacher traces** (609 full-difficulty +
384 hinted) reaches **17.8% exact match on `test_hard`** (pass@4 25.8%) with a 64k thinking
budget. Every other LLM configuration tried is at 0–5%. Accuracy still rises roughly linearly
with the number of traces (259 → 993: 6.2% → 17.8%). RLVR on top of that checkpoint was
started and stopped at step ~30 with no improvement visible yet (step-20 eval 14.5% at 64k,
inside the ±2.4-point noise of the 256-puzzle eval).

## What exists on disk

| what | where |
|---|---|
| verified teacher traces (one JSONL record per attempt; `exact: true` marks the usable ones) | `cot/deepseek_deepseek-v4-pro/train.jsonl` (1000 puzzles, ≤3 attempts), `cot/deepseek_deepseek-v4-pro/repair.jsonl` (27), `cot/anthropic_claude-opus-5/train.jsonl` (87), `cot/deepseek_deepseek-v4-pro/train_hinted40.jsonl` (471) |
| SFT checkpoints (fp32, text-only `Qwen3_5ForCausalLM`) | `checkpoints/llm_cot_qwen3.5_4b/final_{259,354,522}traces`, `checkpoints/llm_cot_qwen3.5_4b_full609/final`, **`checkpoints/llm_cot_qwen3.5_4b_all1080/final`** (best) |
| SFT evals (16k / 32k / 64k on the first 256 `test_hard` rows, 4 samples each, plus the 8/16/30/45-blank curve) | `diag/llm_cot_qwen3.5_4b_{259,354,522}traces/`, `diag/llm_cot_qwen3.5_4b_full609/`, `diag/llm_cot_qwen3.5_4b_all1080/` (`sft`, `sft_budget32768`, `sft_budget65536`) |
| RLVR-from-SFT run (stopped) | `checkpoints/llm_rlvr_from_cot_32k/checkpoint-20`, evals `diag/llm_rlvr_from_cot_32k/step{0,20}`, W&B `sudoku/mmgmtuze`, logs `logs/llm_rlvr_from_cot_32k*.log` |
| earlier RLVR-from-base arms (finished, negative) | `diag/llm_rlvr_v2_4b_{8k,16k}/`, W&B `t40xlpd1` / `2ipcf6wn`; v1 `diag/llm_rlvr_qwen3.5_4b/`, W&B `hd3nyhcc` |
| figure + write-up | `baselines/llm_cot/results/{llm_sudoku_results.svg,RESULTS.md,make_figure.py,sft_scaling.json}` |
| code | `baselines/llm_cot/{generate,repair,train}.py`, `run.sh`; `baselines/llm_rlvr/{common,diag,train}.py`, `run.sh` |

Nothing in `baselines/llm_cot/`, `baselines/llm_rlvr/` or the new memory notes is committed
yet (`git status` shows them untracked); `cot/`, `diag/`, `checkpoints/`, `logs/` are gitignored.

## Numbers (test_hard, first 256 rows, pass@1 unless noted)

| arm | 16k | 32k | 64k |
|---|---|---|---|
| Qwen3.5-4B base | 0.1% | – | – |
| RLVR from base (v2, 16k-trained, 160 steps) | 0% | 0% | – |
| Qwen3.8-27B direct-answer SFT | 4.0% (no thinking) | | |
| Qwen3.8-27B prompting | | 5.3% | |
| CoT SFT, 259 traces | 0.9% | 2.5% | 6.2% |
| CoT SFT, 354 | 1.6% | 3.4% | 8.9% |
| CoT SFT, 522 | 1.9% | 4.4% | 12.7% |
| CoT SFT, 609 (+Opus) | 1.8% | 4.6% | 14.2% |
| **CoT SFT, 609 + 384 hinted** | 2.4% | 5.9% | **17.8%** (pass@4 25.8%) |
| RLVR from that, step 20 | – | 5.0% | 14.5% (pass@4 23.4%) |

Teacher yields on the 1000 training puzzles: DeepSeek V4 Pro 35% per first attempt, 52%
after three attempts + repair; Claude Opus 5 46% on the unsolved tail; hinted (40 blanks) 99%.
Solvability follows the dataset rating: rating 0 → 100%, 1–9 → 65%, 10+ → ~30%. Nothing from
`test_hard` was ever sent to a teacher.

## Money spent

* OpenRouter: ~$210 DeepSeek (first pass) + ~$25 probes + **$980 Opus** → credits essentially
  exhausted (the Opus run overshot my $850 cap by ~$130: the cap stops new requests, in-flight
  ones still finish).
* DeepSeek native API: retries, repairs, hinted set — small (balance was ¥38k, cost not itemised
  by that API; roughly ¥300–500).

## Things I learned that are not obvious from the code

* OpenRouter routes `deepseek-v4-pro` across hosts; BaseTen/SiliconFlow cap reasoning at
  32,768 tokens and DigitalOcean/NextBit drop long generations → pin `--providers
  novita,parasail`, or use the native API (500 concurrent max; 400 is safe).
* Claude 5 on OpenRouter returns *summarised* thinking and rejects assistant prefill; the usable
  mode is `--visible-cot` (derivation as output) with the multi-turn "continue where you
  stopped" for outputs cut at 128k. GPT-6 Astra returns encrypted reasoning — unusable.
* Nine models probed on rating-30 puzzles DeepSeek failed: only Opus 5 solved any.
* Liger's fused linear cross-entropy is required for SFT on 100k-token traces (logits would be
  ~100 GB); sequences up to 128k tokens train fine on 8×H100 at batch 1 per GPU.
* The RLVR curriculum must clamp hints: one training puzzle has only 46 blanks (fixed in
  `common.puzzle_with_empties`).
* The 256-puzzle eval has ±2.4-point standard error at ~18%; differences of <3 points between
  single checkpoints are noise.

## Open items, in the order I'd do them

1. **Decide on RLVR-from-SFT.** Resume with the same command (it continues from
   `checkpoint-20`; ~9 min/step at 32k, 90 steps ≈ 14 h + ~5 h of evals). Judge at the step-40
   and step-60 64k evals; stop if not above 17.8%. Alternative that is cheaper per step: budget
   16k with `--empties-start 40` to purely train efficiency, then evaluate at 64k.
   ```
   BASE=checkpoints/llm_cot_qwen3.5_4b_all1080/final NAME=llm_rlvr_from_cot_32k PORT=29850 \
   BUDGET=32768 TOTAL_STEPS=120 CHUNK=20 EVAL_N=256 EXTRA_EVAL='test_hard:0:65536:256' \
   TRAIN_ARGS='--prompts-per-step 16 --per-device-batch 1 --empties-start 30 --empties-floor 16 --replay 0.3 --vllm-gpu-memory-utilization 0.4' \
   tmux new -d -s rlvr_sft "bash baselines/llm_rlvr/run.sh 2>&1 | tee -a logs/llm_rlvr_from_cot_32k.log"
   ```
2. **A larger eval of the best checkpoint for the paper number**: 2,048 puzzles × 1 sample at
   64k (~1 h) — `baselines/llm_rlvr/diag.py --model checkpoints/llm_cot_qwen3.5_4b_all1080/final
   --config test_hard:0:65536:2048 --n 1`, sharded over 8 GPUs as in `run.sh`'s `evaluate`.
3. **More traces** are the surest gain (the curve has not bent): a second hinted level (e.g.
   48 blanks) on the 391 puzzles without a full trace is ~$0.1 each on DeepSeek; more Opus
   traces need more OpenRouter credit (~$11/trace) or the batch API (half price; blocked by the
   account's Zero-Data-Retention setting).
4. `train.py` drops traces >128k tokens (54 of them, mostly Opus continuations); raising
   `--max-len` to 256k would need a memory check.
5. Add the from-scratch reference (HRM / flow) to the figure once the headline number is
   confirmed — `logs/hrm*` contain several runs with very different `test_hard_exact_match`.
6. Commit `baselines/llm_cot`, `baselines/llm_rlvr`, and the results directory.

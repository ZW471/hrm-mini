#!/usr/bin/env bash
# Reasoning-LLM Sudoku baseline: rounds of (roll out own reasoning -> answer-only SFT -> eval).
#
#   tmux new -s llm_reason 'bash baselines/llm_reason/run.sh 2>&1 | tee logs/llm_reason.log'
#
# Resumable: every stage writes a marker under $EXP, and finished stages are skipped on rerun.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=${BASE:-Qwen/Qwen3.8-27B}
NAME=${NAME:-llm_reason_qwen3.8_27b}
ROUNDS=${ROUNDS:-4}
PER_ROUND=${PER_ROUND:-1024}      # augmented training puzzles rolled out per round (~33k tokens each)
EVAL_N=${EVAL_N:-512}             # test_hard puzzles per eval (un-augmented, fixed subset)
THINK=${THINK:-32768}             # reasoning budget, tokens
LR=${LR:-1e-5}
GRAD_ACCUM=${GRAD_ACCUM:-2}       # global batch = 8 x this
SEED=${SEED:-1}
VLLM=.venv-vllm/bin/python
TRAIN=.venv-llm/bin/torchrun

export HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-/sg-pretrain/zhiyu/.cache/huggingface} OMP_NUM_THREADS=1 WANDB_PROJECT=sudoku
EXP=${EXP:-checkpoints/$NAME}
mkdir -p "$EXP" "rollouts/$NAME"

export BASE ROUNDS PER_ROUND THINK LR GRAD_ACCUM SEED
# One W&B run for the whole experiment; each stage resumes it by id.
if [ ! -f "$EXP/wandb_id" ]; then
  .venv-llm/bin/python - "$NAME" "$EXP/wandb_id" <<'PY'
import sys, coolname, wandb
name, out = sys.argv[1], sys.argv[2]
group = f"{name} {coolname.generate_slug(2)}"
run = wandb.init(project="sudoku", name=group, group=group,
                 config=dict(base=__import__("os").environ.get("BASE"), rounds=int(__import__("os").environ["ROUNDS"]),
                             per_round=int(__import__("os").environ["PER_ROUND"]), think_budget=int(__import__("os").environ["THINK"]),
                             lr=float(__import__("os").environ["LR"]), grad_accum=int(__import__("os").environ["GRAD_ACCUM"]),
                             seed=int(__import__("os").environ["SEED"]), loss="answer_only", reasoning="self_generated"),
                 settings=wandb.Settings(x_disable_stats=True))
open(out, "w").write(run.id)
open(out + ".group", "w").write(group)
run.finish()
PY
fi
WANDB_ID=$(cat "$EXP/wandb_id"); GROUP=$(cat "$EXP/wandb_id.group")
echo "[run] $GROUP  wandb id $WANDB_ID"

# rollout <model> <split> <offset> <count> <out-prefix> [--no-augment]: two TP=4 replicas
rollout() {
  local model=$1 split=$2 offset=$3 count=$4 out=$5; shift 5
  if [ -f "$out.done" ]; then echo "[skip] $out"; return; fi
  mkdir -p "$(dirname "$out")"; rm -f "$out".rank*.jsonl
  CUDA_VISIBLE_DEVICES=0,1,2,3 $VLLM -m baselines.llm_reason.rollout --model "$model" --split "$split" \
    --offset "$offset" --count "$count" --seed "$SEED" --max-think-tokens "$THINK" \
    --dp-rank 0 --dp-size 2 --tp 4 --out "$out" "$@" > "$out.rank0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=4,5,6,7 $VLLM -m baselines.llm_reason.rollout --model "$model" --split "$split" \
    --offset "$offset" --count "$count" --seed "$SEED" --max-think-tokens "$THINK" \
    --dp-rank 1 --dp-size 2 --tp 4 --out "$out" "$@" > "$out.rank1.log" 2>&1 &
  wait
  touch "$out.done"
}
merge() {  # merge <out-prefix> <eval|rollout> <step>
  $VLLM -m baselines.llm_reason.rollout --merge "$1" --wandb-id "$WANDB_ID" --wandb-prefix "$2" --wandb-step "$3"
}

STEP=0
# Round 0: the untouched model under the same protocol -- the prompting-only control.
rollout "$BASE" test_hard 0 "$EVAL_N" "rollouts/$NAME/round0/eval" --no-augment
merge "rollouts/$NAME/round0/eval" eval 0

PREV=$BASE
for r in $(seq 1 "$ROUNDS"); do
  R="rollouts/$NAME/round$r"; CK="$EXP/round$r"
  echo "[round $r] rollouts from $PREV"
  rollout "$PREV" train $(( (r - 1) * PER_ROUND )) "$PER_ROUND" "$R/train"
  merge "$R/train" rollout "$STEP"

  if [ ! -f "$CK/train_meta.json" ]; then
    $TRAIN --nproc-per-node 8 -m baselines.llm_reason.train --model "$PREV" --tokenizer "$BASE" \
      --rollouts "$R/train" --output-dir "$CK" --lr "$LR" --grad-accum "$GRAD_ACCUM" --seed "$((SEED + r))" \
      --liger --wandb-id "$WANDB_ID" --wandb-group "$GROUP" --step-offset "$STEP"
  fi
  STEP=$(( STEP + $(python3 -c "import json;print(json.load(open('$CK/train_meta.json'))['steps'])") ))

  rollout "$CK" test_hard 0 "$EVAL_N" "$R/eval" --no-augment
  merge "$R/eval" eval "$STEP"

  # 150GB per fp32 checkpoint: keep only the newest two.
  if [ "$r" -ge 3 ]; then rm -rf "$EXP/round$((r - 2))"; fi
  PREV=$CK
done
echo "[run] done after $STEP optimizer steps"

#!/usr/bin/env bash
# RLVR Sudoku baseline: diagnose the base model, then GRPO in chunks with an eval after each chunk.
#
#   tmux new -s llm_rlvr 'bash baselines/llm_rlvr/run.sh 2>&1 | tee -a logs/llm_rlvr.log'
#
# Resumable: the trainer restarts from its newest checkpoint, every eval leaves a `.done` marker,
# and a finished chunk is recognised by its checkpoint rather than the trainer's exit code (process
# teardown with colocated vLLM sometimes aborts after the checkpoint is safely on disk).
set -uo pipefail
cd "$(dirname "$0")/../.."

BASE=${BASE:-Qwen/Qwen3.5-4B}
NAME=${NAME:-llm_rlvr_qwen3.5_4b}
TOTAL_STEPS=${TOTAL_STEPS:-200}
CHUNK=${CHUNK:-20}                # optimizer steps between evals (= --save-steps)
BUDGET=${BUDGET:-8192}            # completion budget: thinking + answer
EVAL_N=${EVAL_N:-256}             # test_hard puzzles per eval, fixed subset
EVAL_K=${EVAL_K:-4}               # samples per puzzle (pass@1 and pass@K)
CURVE_N=${CURVE_N:-64}            # training puzzles per difficulty level, for the curriculum curve
SEED=${SEED:-1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}     # comma-separated; the run uses exactly these
PORT=${PORT:-29810}
PY=.venv-rl/bin/python
TORCHRUN=.venv-rl/bin/torchrun

export HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-/sg-pretrain/zhiyu/.cache/huggingface} OMP_NUM_THREADS=1 WANDB_PROJECT=sudoku
EXP=${EXP:-checkpoints/$NAME}
mkdir -p "$EXP" "diag/$NAME" logs

export BASE TOTAL_STEPS BUDGET SEED
# One W&B run for the whole experiment: the trainer resumes it by id on every chunk, and the
# evals log into it at the matching optimizer step.
if [ ! -f "$EXP/wandb_id" ]; then
  $PY - "$NAME" "$EXP/wandb_id" <<'PY'
import os, sys, coolname, wandb
name, out = sys.argv[1], sys.argv[2]
group = f"{name} {coolname.generate_slug(2)}"
run = wandb.init(project="sudoku", name=group, group=group,
                 config=dict(base=os.environ["BASE"], total_steps=int(os.environ["TOTAL_STEPS"]),
                             budget=int(os.environ["BUDGET"]), seed=int(os.environ["SEED"]),
                             method="grpo", reward="exact_match", curriculum="blanks_left"),
                 settings=wandb.Settings(x_disable_stats=True))
open(out, "w").write(run.id)
open(out + ".group", "w").write(group)
run.finish()
PY
fi
WANDB_ID=$(cat "$EXP/wandb_id"); GROUP=$(cat "$EXP/wandb_id.group")
IFS=, read -ra GPU_LIST <<< "$GPUS"; NGPU=${#GPU_LIST[@]}
echo "[run] $GROUP  wandb id $WANDB_ID  gpus $GPUS"

# evaluate <model> <step> <tag> <config>...: `diag.py` over the given configs, one vLLM process
# per GPU, merged and logged to W&B at <step>.
evaluate() {
  local model=$1 step=$2 tag=$3; shift 3
  local out="diag/$NAME/$tag/eval"
  if [ -f "$out.done" ]; then echo "[skip] $out"; return; fi
  mkdir -p "$(dirname "$out")"; rm -f "$out".rank*.jsonl
  local configs=(); for c in "$@"; do configs+=(--config "$c"); done
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} $PY -m baselines.llm_rlvr.diag --model "$model" --seed "$SEED" "${configs[@]}" \
      --n "$EVAL_K" --dp-rank "$i" --dp-size "$NGPU" --out "$out" > "$out.rank$i.log" 2>&1 &
  done
  wait
  if $PY -m baselines.llm_rlvr.diag --merge "$out" --wandb-id "$WANDB_ID" --wandb-step "$step"; then
    touch "$out.done"
  else
    echo "[eval] FAILED for $model; see $out.rank*.log"
  fi
}

# test_hard pass@K, plus the curriculum curve on training puzzles at fixed difficulty levels.
STD=("test_hard:0:$BUDGET:$EVAL_N" "train:8:$BUDGET:$CURVE_N" "train:16:$BUDGET:$CURVE_N"
     "train:30:$BUDGET:$CURVE_N" "train:45:$BUDGET:$CURVE_N")

# Step 0: the untouched model.
evaluate "$BASE" 0 step0 "${STD[@]}"

# Train in chunks; eval each checkpoint. The trainer resumes from its own newest checkpoint.
step=0
while [ "$step" -lt "$TOTAL_STEPS" ]; do
  target=$(( step + CHUNK )); [ "$target" -gt "$TOTAL_STEPS" ] && target=$TOTAL_STEPS
  ck="$EXP/checkpoint-$target"
  if [ ! -f "$ck/trainer_state.json" ]; then
    echo "[train] steps $step -> $target"
    for attempt in 1 2; do
      WANDB_RUN_ID=$WANDB_ID WANDB_RESUME=allow CUDA_VISIBLE_DEVICES=$GPUS \
      $TORCHRUN --nproc-per-node "$NGPU" --master-port "$PORT" -m baselines.llm_rlvr.train \
        --model "$BASE" --output-dir "$EXP" --run-name "$GROUP" --seed "$SEED" \
        --max-steps "$target" --save-steps "$CHUNK" --max-completion-length "$BUDGET" ${TRAIN_ARGS:-} \
        > "logs/${NAME}_train_${target}.log" 2>&1
      [ -f "$ck/trainer_state.json" ] && break
      echo "[train] attempt $attempt did not produce $ck; log: logs/${NAME}_train_${target}.log"
    done
    if [ ! -f "$ck/trainer_state.json" ]; then echo "[run] giving up"; exit 1; fi
  fi
  evaluate "$ck" "$target" "step$target" "${STD[@]}"
  step=$target
done

# The last checkpoint with twice the budget: does more room to think help after RL?
evaluate "$EXP/checkpoint-$TOTAL_STEPS" "$TOTAL_STEPS" "step${TOTAL_STEPS}_budget$((BUDGET * 2))" "test_hard:0:$((BUDGET * 2)):$EVAL_N"
echo "[run] done"

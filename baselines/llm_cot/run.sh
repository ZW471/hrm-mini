#!/usr/bin/env bash
# Distilled-CoT Sudoku baseline: generate verified traces from an API model, SFT the student on
# them, evaluate with the RLVR diagnostic (pass@k on test_hard plus the difficulty curve).
#
#   tmux new -s llm_cot 'bash baselines/llm_cot/run.sh 2>&1 | tee -a logs/llm_cot.log'
#
# Resumable: generation appends and skips done indices, SFT is skipped if its `final/` exists,
# evals leave `.done` markers.
set -uo pipefail
cd "$(dirname "$0")/../.."

BACKEND=${BACKEND:-deepseek}          # deepseek (native API) or openrouter
TEACHER=${TEACHER:-deepseek-v4-pro}
STUDENT=${STUDENT:-Qwen/Qwen3.5-4B}
NAME=${NAME:-llm_cot_qwen3.5_4b}
COUNT=${COUNT:-1000}              # training puzzles sent to the teacher (raw rows, no augmentation)
WORKERS=${WORKERS:-400}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
GRAD_ACCUM=${GRAD_ACCUM:-4}
BUDGET=${BUDGET:-16384}           # student eval budget
EVAL_N=${EVAL_N:-256}
EVAL_K=${EVAL_K:-4}
CURVE_N=${CURVE_N:-64}
SEED=${SEED:-1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-29830}
PY=.venv-rl/bin/python
TORCHRUN=.venv-rl/bin/torchrun

export HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-/sg-pretrain/zhiyu/.cache/huggingface} OMP_NUM_THREADS=1 WANDB_PROJECT=sudoku
TRACES=${TRACES:-cot/deepseek_deepseek-v4-pro/train.jsonl}
REPAIRS=${REPAIRS:-cot/deepseek_deepseek-v4-pro/repair.jsonl}   # from repair.py; optional
EXTRA_TRACES=${EXTRA_TRACES:-}    # space-separated extra JSONLs (another teacher, hinted variants)
SKIP_GENERATE=${SKIP_GENERATE:-0}
EXP=checkpoints/$NAME
mkdir -p "$(dirname "$TRACES")" "$EXP" "diag/$NAME" logs
IFS=, read -ra GPU_LIST <<< "$GPUS"; NGPU=${#GPU_LIST[@]}

# 1. Teacher traces (CPU/network only; GPUs are free while this runs).
[ "$SKIP_GENERATE" = 1 ] || $PY -m baselines.llm_cot.generate --backend "$BACKEND" --model "$TEACHER" --count "$COUNT" --workers "$WORKERS" \
  --seed "$SEED" --out "$TRACES" --retry-errors
$PY - "$TRACES" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in recs if r.get("exact")]
tok = [r["usage"].get("completion_tokens", 0) for r in ok if r.get("usage")]
print(f"[traces] {len(ok)}/{len(recs)} correct; completion tokens mean {sum(tok)/max(len(tok),1):.0f}, "
      f"max {max(tok, default=0)}; cost ${sum(float((r.get('usage') or {}).get('cost') or 0) for r in recs):.2f}")
PY

# 2. SFT (redone whenever the traces are newer than the last checkpoint; old evals are cleared).
TRACE_FILES=("$TRACES"); [ -f "$REPAIRS" ] && TRACE_FILES+=("$REPAIRS")
for f in $EXTRA_TRACES; do TRACE_FILES+=("$f"); done
stale=0; for f in "${TRACE_FILES[@]}"; do [ "$f" -nt "$EXP/final/train_meta.json" ] && stale=1; done
if [ ! -f "$EXP/final/train_meta.json" ] || [ "$stale" = 1 ]; then
  rm -rf "$EXP/final" "diag/$NAME"
  CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc-per-node "$NGPU" --master-port "$PORT" -m baselines.llm_cot.train \
    --model "$STUDENT" --traces "${TRACE_FILES[@]}" --output-dir "$EXP" --run-name "$NAME" --seed "$SEED" \
    --epochs "$EPOCHS" --lr "$LR" --grad-accum "$GRAD_ACCUM" > "logs/${NAME}_train.log" 2>&1
  [ -f "$EXP/final/train_meta.json" ] || { echo "[run] SFT failed; see logs/${NAME}_train.log"; exit 1; }
fi

# 3. Eval, same protocol and panels as the RLVR arms.
evaluate() {
  local model=$1 tag=$2; shift 2
  local out="diag/$NAME/$tag/eval"
  if [ -f "$out.done" ]; then echo "[skip] $out"; return; fi
  mkdir -p "$(dirname "$out")"; rm -f "$out".rank*.jsonl
  local configs=(); for c in "$@"; do configs+=(--config "$c"); done
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} $PY -m baselines.llm_rlvr.diag --model "$model" --seed "$SEED" "${configs[@]}" \
      --n "$EVAL_K" --dp-rank "$i" --dp-size "$NGPU" --out "$out" > "$out.rank$i.log" 2>&1 &
  done
  wait
  $PY -m baselines.llm_rlvr.diag --merge "$out" && touch "$out.done"
}
STD=("test_hard:0:$BUDGET:$EVAL_N" "train:8:$BUDGET:$CURVE_N" "train:16:$BUDGET:$CURVE_N"
     "train:30:$BUDGET:$CURVE_N" "train:45:$BUDGET:$CURVE_N")
evaluate "$EXP/final" sft "${STD[@]}"
# The teacher's correct traces are long (median ~50k tokens), so the student also gets 2x and 4x
# the standard budget on test_hard; report the budgets side by side.
evaluate "$EXP/final" "sft_budget$((BUDGET * 2))" "test_hard:0:$((BUDGET * 2)):$EVAL_N"
evaluate "$EXP/final" "sft_budget$((BUDGET * 4))" "test_hard:0:$((BUDGET * 4)):$EVAL_N"
echo "[run] done"

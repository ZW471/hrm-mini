#!/usr/bin/env bash
#
# Extra seeds for the v2 data-scaling runs on GPUs as they become idle (2026-09-22, user request: "since
# they are idle, use them to run different seeds; the W&B run name for different seeds should stay the same
# so they can be grouped").
#
# Per host, a worker loop: every 60 s, find GPUs with no compute process (and nothing launched on them in
# the last 5 min), pop the next `<size> <seed>` line from logs/v2/queue_<ARCH>.txt and launch
# `<ARCH>_<size>_v2` with `seeds=[<seed>]` on that GPU -- the SAME settings as run_scaling_v2.sh (the
# per-size table below must stay identical to it), the same MLP_TASK_NAME (= W&B name and group, so seeds
# group together; config.seed tells them apart), checkpoints in checkpoints/<run>/seed_<seed>/, log
# logs/v2/<run>_s<seed>.log, one tmux window `<run>_s<seed>` in session `v2`.
#
#   ARCH=hrm ./run_v2_seeds.sh     # on Sapient-sg-2 (tmux window `seeds` in session `v2`)
#   ARCH=rt  ./run_v2_seeds.sh     # on Sapient-sg
# Queue lines: `<size> <seed>`; '#' comments; `STOP` ends the worker. Edit the file any time.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:?set ARCH=hrm or ARCH=rt}"
SESSION="${SESSION:-v2}"
QUEUE="$REPO/logs/v2/queue_${ARCH}.txt"
STATE="$REPO/logs/v2/seeds_${ARCH}_launched.txt"    # "<epoch> <gpu> <run> <seed>" per launch
LR=1e-4; MIN_RATIO=0.01; PATIENCE=49920; EVAL=4160; BATCH=768

# size -> "data config | weight decay | max_steps | epochs"   (identical to run_scaling_v2.sh)
declare -A SPEC=(
  [1]="sudoku_1 1.0 104000 30"
  [10]="sudoku_10 1.0 104000 30"
  [100]="sudoku_100 0.3 104000 30"
  [1k]="sudoku 0.3 166400 45"
  [10k]="sudoku_10k 0.1 332800 85"
  [100k]="sudoku_100k 0.1 416000 105"
  [1m]="sudoku_1m 0.1 499200 30"
  [full]="sudoku_full 0.1 582400 10"
)

pop_queue() {
    local line
    line="$(grep -v -m1 -E '^\s*(#|$)' "$QUEUE" || true)"
    [[ -z "$line" ]] && return 1
    awk -v skip="$line" 'BEGIN{done=0} { if (!done && $0 == skip) { done=1; next } print }' "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    printf '%s\n' "$line"
}

free_gpus() {
    # GPU indices with no compute process and no launch recorded in the last 300 s
    local busy_uuids now
    busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)"
    now="$(date +%s)"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r idx uuid; do
        grep -q "$uuid" <<< "$busy_uuids" && continue
        if [[ -f "$STATE" ]] && awk -v g="$idx" -v t="$now" '$2 == g && t - $1 < 300 {found=1} END{exit !found}' "$STATE"; then continue; fi
        echo "$idx"
    done
}

launch() {
    local size="$1" seed="$2" gpu="$3" run data wd steps epochs cmd
    run="${ARCH}_${size}_v2"
    read -r data wd steps epochs <<< "${SPEC[$size]}"
    cmd="cd '$REPO' && CUDA_VISIBLE_DEVICES=$gpu MLP_TASK_NAME='$run' WANDB_TAGS='scaling test,scaling_v2' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
'$REPO/.venv/bin/torchrun' --standalone --nproc-per-node 1 train.py --config-name tuned_${ARCH}_full \
data=$data epochs=$epochs local_batch_size=$BATCH lr=$LR weight_decay=$wd lr_min_ratio=$MIN_RATIO eval_interval=$EVAL \
+max_steps=$steps +early_stop_patience_steps=$PATIENCE seeds=[$seed] run_name=$run > 'logs/v2/${run}_s${seed}.log' 2>&1; \
echo \"${run}_s${seed} exit=\$? \$(date -u '+%F %T')\" >> 'logs/v2/done.txt'; sleep 86400"
    tmux new-window -t "$SESSION" -n "${run}_s${seed}" "$cmd"
    echo "$(date +%s) $gpu $run $seed" >> "$STATE"
    echo "${run} seed $seed START $(date -u '+%F %T') gpu=$gpu (data=$data wd=$wd max_steps=$steps epochs=$epochs)" >> "$REPO/logs/v2/launched.txt"
    echo "$(date -u '+%F %T') launched $run seed $seed on gpu $gpu"
}

if [[ "${1:-}" == "--worker" ]]; then
    cd "$REPO" || exit 1
    touch "$QUEUE"
    while true; do
        for gpu in $(free_gpus); do
            line="$(pop_queue)" || break
            [[ "$line" == "STOP" ]] && { echo "=== STOP reached $(date '+%F %T') ==="; exit 0; }
            read -r size seed <<< "$line"
            [[ -z "${SPEC[$size]:-}" ]] && { echo "unknown size '$size' in queue line '$line', skipped"; continue; }
            launch "$size" "$seed" "$gpu"
        done
        sleep 60
    done
fi

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
tmux has-session -t "=$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -c "$REPO" -n launcher "sleep 3600"
if tmux list-windows -t "$SESSION" -F '#W' | grep -qx seeds; then
    echo "seed worker already running in tmux window '$SESSION:seeds'"; exit 1
fi
tmux new-window -t "$SESSION" -n seeds "ARCH='$ARCH' SESSION='$SESSION' '$REPO/run_v2_seeds.sh' --worker 2>&1 | tee -a '$REPO/logs/v2/seeds_${ARCH}.log'"
echo "seed worker started in tmux window '$SESSION:seeds'; queue: $QUEUE"

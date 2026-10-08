#!/usr/bin/env bash
#
# Data-scaling study, protocol v2 (2026-09-22): ONE run per (architecture, dataset size), ONE GPU each,
# no stages, no restarts, nothing decided by looking at the eval curve except the early stop.
#
#   * 8 sizes: 1 / 10 / 100 / 1k / 10k / 100k / 1M / full (3,831,994 puzzles), 2 architectures (HRM, RT)
#     -> 16 runs = 16 H100s: HRM on host `Sapient-sg-2` (this script with ARCH=hrm), RT on `Sapient-sg`
#     (ARCH=rt), GPU i = size i. Same configuration for HRM and RT at a given size.
#   * global batch 768 on one GPU (local_batch_size 768, the same 96 x 8 the v1 runs used on 8 GPUs),
#     lr 1e-4 with 2k warm-up, cosine to 0.01 x lr over `max_steps` (the run's fixed horizon), weight
#     decay per size, EMA 0.999 (both models; see logs/v2/README.md for the EMA check), eval every 4,160
#     steps on the 20k test_hard split at 16 cycles, best.pt = best eval; early stop when the best has not
#     improved by 0.1 pp for 49,920 steps (12 evals).
#   * `epochs` only has to be large enough to reach `max_steps` (train.py raises otherwise).
#
#   ARCH=hrm ./run_scaling_v2.sh          # on Sapient-sg-2: tmux session `v2`, one window per run
#   ARCH=rt  ./run_scaling_v2.sh          # on Sapient-sg
#   tmux attach -t v2 ; logs in logs/v2/<run>.log ; checkpoints in checkpoints/<run>/seed_1/
#
# W&B: project `sudoku`, run name = checkpoint dir = `<arch>_<size>_v2`, tags `scaling test`, `scaling_v2`
# (the v1 runs were tagged `scaling_v1` on 2026-09-22 so they can be hidden).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:?set ARCH=hrm or ARCH=rt}"
SESSION="${SESSION:-v2}"
LR=1e-4
MIN_RATIO=0.01
PATIENCE=49920
EVAL=4160
BATCH=768

# size | data config | weight decay | max_steps (cosine horizon) | epochs (upper bound, >= max_steps)
# wd: the v1 sweep's per-size winners (1 at <= 10 puzzles where it is irrelevant, 0.3 at 100, 0.1 at >= 10k;
# 1k gets 0.3 by interpolation). Horizons: generous relative to where the v1 chains converged, with the
# early stop as the actual terminator for the sizes that overfit (<= 1k peak within 10-30k steps).
RUNS=(
  "1     sudoku_1     1.0  104000  30"
  "10    sudoku_10    1.0  104000  30"
  "100   sudoku_100   0.3  104000  30"
  "1k    sudoku       0.3  166400  45"
  "10k   sudoku_10k   0.1  332800  85"
  "100k  sudoku_100k  0.1  416000  105"
  "1m    sudoku_1m    0.1  499200  30"
  "full  sudoku_full  0.1  582400  10"
)

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"; exit 1
fi
mkdir -p "$REPO/logs/v2"
tmux new-session -d -s "$SESSION" -c "$REPO" -n launcher "sleep 3600"
gpu=0
for spec in "${RUNS[@]}"; do
    read -r size data wd steps epochs <<< "$spec"
    run="${ARCH}_${size}_v2"
    cmd="cd '$REPO' && CUDA_VISIBLE_DEVICES=$gpu MLP_TASK_NAME='$run' WANDB_TAGS='scaling test,scaling_v2' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
'$REPO/.venv/bin/torchrun' --standalone --nproc-per-node 1 train.py --config-name tuned_${ARCH}_full \
data=$data epochs=$epochs local_batch_size=$BATCH lr=$LR weight_decay=$wd lr_min_ratio=$MIN_RATIO eval_interval=$EVAL \
+max_steps=$steps +early_stop_patience_steps=$PATIENCE seeds=[1] run_name=$run > 'logs/v2/$run.log' 2>&1; \
echo \"$run exit=\$? \$(date -u '+%F %T')\" >> 'logs/v2/done.txt'; sleep 86400"
    echo "$run START $(date -u '+%F %T') gpu=$gpu (data=$data wd=$wd max_steps=$steps epochs=$epochs)" >> "$REPO/logs/v2/launched.txt"
    tmux new-window -t "$SESSION" -n "$run" "$cmd"
    gpu=$((gpu + 1))
done
echo "launched ${#RUNS[@]} $ARCH runs in tmux session '$SESSION' (one per GPU); logs/v2/<run>.log"

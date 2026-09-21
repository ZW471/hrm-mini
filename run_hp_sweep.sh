#!/usr/bin/env bash
#
# LR / weight-decay sweep of HRM (then RT at HRM's winning setting) on the nested Sudoku-Extreme
# subsets of 10k, 100k and 1M puzzles, one seed each, on all 8 GPUs, run back to back in a tmux
# session. Every run keeps the ~83k-step budget of the 1k / full arms (see run_dataset_sizes.sh):
#     10k: repeat 20 x 20 epochs     100k: repeat 2 x 20 epochs     1M: repeat 1 x 4 epochs
# so lr / weight decay (and optionally the lr schedule) are the only knobs that move.
#
# The worker is driven by a queue FILE rather than a fixed list, so runs can be added (the RT arms,
# a refinement around a winner) or removed while the sweep is going:
#     logs/hpsweep/queue.txt    one run per line:  <run name>|<base config>|<hydra overrides>
#                               '#' lines are comments; the first non-comment line is taken next.
#     logs/hpsweep/done.txt     <run name> <exit code> <start> <end>, appended after each run
# The worker sleeps and re-reads the queue when it is empty, so the tmux session stays alive
# until it is killed (or a line reading `STOP` is reached).
#
#   ./run_hp_sweep.sh                  # start the worker in tmux session `hpsweep` (queue as is)
#   tmux attach -t hpsweep             # watch;  logs in logs/hpsweep/<run>.log, runner.log
#   .venv/bin/python experiments/collect_hp_sweep.py     # results table from W&B
#
# Checkpoints land in checkpoints/<run name>/seed_1 (via MLP_TASK_NAME) and W&B runs in project
# `sudoku` under the same name, tagged `hp_sweep` and `scaling test` (WANDB_TAGS is read by wandb.init).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-hpsweep}"
NPROC="${NPROC:-8}"
QUEUE="$REPO/logs/$SESSION/queue.txt"
DONE="$REPO/logs/$SESSION/done.txt"

pop_queue() {
    # Print the first non-comment, non-blank line of the queue and delete it from the file.
    local line
    line="$(grep -v -m1 -E '^\s*(#|$)' "$QUEUE" || true)"
    [[ -z "$line" ]] && return 1
    # Delete exactly that first occurrence (the queue is short; a tmp file keeps it atomic).
    awk -v skip="$line" 'BEGIN{done=0} { if (!done && $0 == skip) { done=1; next } print }' "$QUEUE" > "$QUEUE.tmp" \
        && mv "$QUEUE.tmp" "$QUEUE"
    printf '%s\n' "$line"
}

if [[ "${1:-}" == "--worker" ]]; then
    cd "$REPO" || exit 1
    mkdir -p "logs/$SESSION"; touch "$QUEUE" "$DONE"
    while true; do
        line="$(pop_queue)" || { sleep 60; continue; }
        [[ "$line" == "STOP" ]] && { echo "=== STOP reached $(date '+%F %T') ==="; break; }
        IFS='|' read -r run config overrides <<< "$line"
        run="$(echo "$run" | xargs)"; config="$(echo "$config" | xargs)"
        start="$(date '+%F %T')"
        echo "=== $run START $start   ($config $overrides)"
        WANDB_TAGS="hp_sweep,scaling test" MLP_TASK_NAME="$run" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node "$NPROC" train.py --config-name "$config" \
                $overrides "seeds=[1]" "run_name=$run" > "logs/$SESSION/$run.log" 2>&1
        rc=$?
        echo "=== $run END exit=$rc $(date '+%F %T') ==="
        echo "$run $rc $start $(date '+%F %T')" >> "$DONE"
        [[ $rc -ne 0 ]] && tail -n 15 "logs/$SESSION/$run.log" | tr '\r' '\n' | tail -n 15 | sed 's/^/    | /'
    done
    exit 0
fi

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"; exit 1
fi
mkdir -p "$REPO/logs/$SESSION"; touch "$QUEUE"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "SESSION='$SESSION' NPROC='$NPROC' '$REPO/run_hp_sweep.sh' --worker 2>&1 | tee -a '$REPO/logs/$SESSION/runner.log'"
echo "Worker started in tmux session '$SESSION' on $NPROC GPUs; queue: $QUEUE"
echo "  tmux attach -t $SESSION   |   tail -f logs/$SESSION/runner.log"

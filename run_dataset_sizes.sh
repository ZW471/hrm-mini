#!/usr/bin/env bash
#
# Train HRM, the Recurrent Transformer, the FLOP-matched MAE and MAE + deep supervision on nested Sudoku-Extreme training
# sets of 10k, 100k and 1M puzzles (plus the MAE on the full set), one seed each, sequentially in a
# tmux session. The 1k and full arms of HRM and RT already exist. Every run keeps the ~83k-step
# budget of the 1k / full configs, so the only variable across a row is the number of unique
# puzzles. Checkpoints land in checkpoints/<arch>_<size>/seed_1 (no random slug, via
# MLP_TASK_NAME) so outputs/failure_mode/rescore_sizes.sh can find them by name.
#
#   ./run_dataset_sizes.sh                  # everything, in order: 10k -> 100k -> 1m -> mae full
#   ./run_dataset_sizes.sh hrm_10k rt_1m    # any subset, by run name
#   SEEDS='[2,3]' LOG_SUFFIX=_s23 ./run_dataset_sizes.sh mae_10k   # more seeds into the same dir
#
# Watch:  tmux attach -t sizes     Logs: logs/sizes/<run>.log, logs/sizes/runner.log
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-sizes}"
NPROC="${NPROC:-8}"
SEEDS="${SEEDS:-[1]}"          # hydra list; SEEDS='[2,3]' adds seeds to an existing run dir
LOG_SUFFIX="${LOG_SUFFIX:-}"   # e.g. _s23, so a re-seed does not overwrite the seed-1 log

# run name -> base config + hydra overrides. Sizes take their repeat from config/data/sudoku_<size>
# and their epoch count here (see the note in each data config).
declare -A RUNS=(
    [hrm_10k]="tuned_hrm_full data=sudoku_10k epochs=20"
    [rt_10k]="tuned_rt_full data=sudoku_10k epochs=20"
    [mae_10k]="mae_flops_matched_full data=sudoku_10k epochs=20"
    [hrm_100k]="tuned_hrm_full data=sudoku_100k epochs=20"
    [rt_100k]="tuned_rt_full data=sudoku_100k epochs=20"
    [mae_100k]="mae_flops_matched_full data=sudoku_100k epochs=20"
    [hrm_1m]="tuned_hrm_full data=sudoku_1m epochs=4"
    [rt_1m]="tuned_rt_full data=sudoku_1m epochs=4"
    [mae_1m]="mae_flops_matched_full data=sudoku_1m epochs=4"
    [mae_full]="mae_flops_matched_full"
    # MAE + deep supervision takes one gradient step per batch (cycles_per_data 1) and makes up
    # the step budget with 16x the repeat of the plain MAE at the same size.
    [mae_ds_10k]="mae_ds_flops_matched_full data=sudoku_10k data.repeat=320 epochs=20"
    [mae_ds_100k]="mae_ds_flops_matched_full data=sudoku_100k data.repeat=32 epochs=20"
    [mae_ds_1m]="mae_ds_flops_matched_full data=sudoku_1m data.repeat=16 epochs=4"
    [mae_ds_full]="mae_ds_flops_matched_full"
    # Diagnostic: the plain MAE at 10k with ONE optimizer step per augmented batch (the convention
    # its 1k checkpoints and the MAE+DS arm use) instead of 16, same 83k-step budget. Tests whether
    # the ln 9 plateau lottery is caused by re-stepping the same batch 16x from init.
    [mae_10k_c1]="mae_flops_matched_full data=sudoku_10k data.repeat=320 cycles_per_data=1 epochs=20"
)
ORDER=(hrm_10k rt_10k mae_10k hrm_100k rt_100k mae_100k hrm_1m rt_1m mae_1m mae_full
       mae_ds_10k mae_ds_100k mae_ds_1m mae_ds_full)

if [[ "${1:-}" == "--worker" ]]; then
    shift; cd "$REPO" || exit 1
    mkdir -p "logs/$SESSION"; failed=0
    for run in "$@"; do
        read -r config overrides <<< "${RUNS[$run]}"
        echo "=== $run START $(date '+%F %T')   ($config $overrides seeds=$SEEDS)"
        MLP_TASK_NAME="$run" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            uv run torchrun --nproc-per-node "$NPROC" train.py --config-name "$config" \
                $overrides "seeds=$SEEDS" "run_name=$run" > "logs/$SESSION/$run$LOG_SUFFIX.log" 2>&1
        rc=$?
        echo "=== $run END exit=$rc $(date '+%F %T') ==="
        [[ $rc -ne 0 ]] && { failed=1; tail -n 15 "logs/$SESSION/$run.log" | sed 's/^/    | /'; }
    done
    [[ $failed -eq 0 ]] && echo "All runs finished successfully. $(date '+%F %T')" || echo "One or more runs FAILED. $(date '+%F %T')"
    exit $failed
fi

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
SELECTED=("$@"); [[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=("${ORDER[@]}")
for run in "${SELECTED[@]}"; do [[ -n "${RUNS[$run]:-}" ]] || { echo "unknown run: $run  (known: ${ORDER[*]})"; exit 1; }; done
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; kill it first: tmux kill-session -t $SESSION"; exit 1
fi
mkdir -p "$REPO/logs/$SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "SESSION='$SESSION' NPROC='$NPROC' SEEDS='$SEEDS' LOG_SUFFIX='$LOG_SUFFIX' '$REPO/run_dataset_sizes.sh' --worker ${SELECTED[*]} 2>&1 | tee -a '$REPO/logs/$SESSION/runner.log'"
echo "Launched ${#SELECTED[@]} run(s) sequentially in tmux session '$SESSION' on $NPROC GPUs:"
printf '  - %s\n' "${SELECTED[@]}"
echo "  tmux attach -t $SESSION   |   tail -f logs/$SESSION/runner.log"

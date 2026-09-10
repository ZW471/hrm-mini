#!/usr/bin/env bash
#
# The two 113M conditional flow-matching runs (`flow_113m_cfg_full`, `flow_113m_cfg_1k`) over
# seeds 1/2/3, sequentially inside a detached tmux session so all 8 GPUs go to one run at a time.
#
#   ./run_flow_seeds.sh                       # all 6 runs, in order
#   ./run_flow_seeds.sh full:1 1k:1           # just those two
#   SESSION=flow2 ./run_flow_seeds.sh         # other session name
#
# Every run keeps the hyperparameters of checkpoints/flow_113m_cfg_*/args.json, with the batch and
# the schedule matched to config/tuned_hrm.yaml as it is run on 8 GPUs:
#
#   local_batch_size 96 x 8 GPUs   = 768 global   (tuned_hrm.yaml's local_batch_size)
#   16 cycles x 260 batches x 20 epochs = 83,200 steps
#   768 x 83,200 = 63,897,600 boards seen, with HRM's band/stack/digit augmentation
#
# Seeds share a `--run-name`, so wandb groups them; checkpoints land in
# checkpoints/<run_name>/seed_<seed>/, leaving the earlier single-seed checkpoints alone.
#
# Logs land in logs/$SESSION/. Watch with:  tmux attach -t flowseeds   (Ctrl-b then d to detach)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-flowseeds}"
NPROC="${NPROC:-8}"

# Interleaved so that stopping early still leaves matched full/1k pairs.
DEFAULT_RUNS=(full:1 1k:1 full:2 1k:2 full:3 1k:3)

# Shared with checkpoints/flow_113m_cfg_full/args.json apart from --train-steps.
COMMON_ARGS=(
    --repr onehot
    --num-layers 16 --hidden-size 768 --intermediate-size 3072 --head-dim 64
    --norm-eps 1e-6 --rope-theta 10000.0 --qk-norm --pos-embed rope2d
    --forward-dtype bfloat16
    --t-schedule uniform --sample-steps 64 --sampler heun --noise-scale 10.0
    --conditional --eval-split test_hard --eval-samples 512
    --cond-dropout 0.1 --guidance 2.0 --clamp-givens
    --train-steps 83200 --local-batch-size 96
    --lr 1e-4 --lr-warmup-steps 2000 --lr-min-ratio 0.1
    --weight-decay 0.1 --grad-clip 0.5 --beta1 0.9 --beta2 0.99 --ema 0.999
    --log-interval 50 --eval-interval 2500
    --wandb-project sudoku
    --wandb-keys eval/test_hard_exact_match train/exact_match train/per_position_accuracy
)

# ---------------------------------------------------------------- worker
if [[ "${1:-}" == "--worker" ]]; then
    shift
    cd "$REPO" || exit 1

    logdir="logs/$SESSION"
    mkdir -p "$logdir"

    # `uv` is not always on a non-login shell's PATH; the project venv's launcher always is.
    if command -v uv >/dev/null; then
        LAUNCH=(uv run torchrun)
    else
        LAUNCH=("$REPO/.venv/bin/torchrun")
    fi

    failed=0
    for run in "$@"; do
        data="${run%%:*}"
        seed="${run##*:}"

        if [[ "$data" == "full" ]]; then
            # Trained on all 3.8M sudoku-extreme solutions, scored on the same test_hard as HRM.
            train_set=./downloaded-datasets/sudoku-extreme
        else
            train_set=./downloaded-datasets/sudoku-extreme-1k
        fi

        echo "=== flow_113m_cfg_$data seed=$seed START $(date '+%F %T') ==="
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "${LAUNCH[@]}" --nproc-per-node "$NPROC" experiments/flow_sudoku.py \
                --dataset-name "$train_set" \
                --eval-dataset-name ./downloaded-datasets/sudoku-extreme-1k \
                --run-name "flow_113m_cfg_$data" --seed "$seed" \
                "${COMMON_ARGS[@]}" \
            > "$logdir/flow_113m_cfg_${data}_seed${seed}.log" 2>&1
        rc=$?
        echo "=== flow_113m_cfg_$data seed=$seed END exit=$rc $(date '+%F %T') ==="
        if [[ $rc -ne 0 ]]; then
            failed=1
            echo "    FAILED -- full output in $logdir/flow_113m_cfg_${data}_seed${seed}.log:"
            tail -n 15 "$logdir/flow_113m_cfg_${data}_seed${seed}.log" | sed 's/^/    | /'
        fi
    done

    echo
    if [[ $failed -eq 0 ]]; then
        echo "All runs finished successfully. $(date '+%F %T')"
    else
        echo "One or more runs FAILED. $(date '+%F %T')"
    fi
    exit $failed
fi

# ---------------------------------------------------------------- launcher
command -v tmux >/dev/null || { echo "tmux is not installed (apt-get install -y tmux)"; exit 1; }

RUNS=("$@")
[[ ${#RUNS[@]} -eq 0 ]] && RUNS=("${DEFAULT_RUNS[@]}")

for run in "${RUNS[@]}"; do
    [[ "$run" =~ ^(full|1k):[0-9]+$ ]] || { echo "bad run spec: $run (want full:<seed> or 1k:<seed>)"; exit 1; }
done

if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  tmux attach -t $SESSION          # see what it is doing"
    echo "  tmux kill-session -t $SESSION    # stop it first"
    exit 1
fi

mkdir -p "$REPO/logs/$SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "SESSION='$SESSION' NPROC='$NPROC' '$REPO/run_flow_seeds.sh' --worker ${RUNS[*]} 2>&1 | tee '$REPO/logs/$SESSION/runner.log'"

echo "Launched ${#RUNS[@]} run(s) sequentially in tmux session '$SESSION' on $NPROC GPUs:"
printf '  - %s\n' "${RUNS[@]}"
cat <<EOF

  tmux attach -t $SESSION                                # watch live (Ctrl-b then d to detach)
  tail -f logs/$SESSION/runner.log                       # which run is going / how each ended
  tail -f logs/$SESSION/flow_113m_cfg_full_seed1.log     # training output of one run

The session closes itself when the last run finishes; logs/$SESSION/runner.log keeps the summary.
EOF

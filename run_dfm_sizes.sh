#!/usr/bin/env bash
#
# Size sweep of the best discrete-flow recipe (uniform prior, self-cond x2, tuned sampler) on
# Sudoku-Extreme 1k: how small can the backbone get and still match HRM (12.59M, 80.65 %)?
#
# Unlike run_dfm_seeds.sh (one 8-GPU run at a time), every run here is a *single-GPU* job at the
# same global batch (768) and the same LR schedule, so up to 8 runs train side by side. A run spec is
#     L<layers>d<hidden>[:<tag>][:<seed>]
# e.g. L8d512, L16d768:soft, L4d256:lr3:2. Tags (flags in `variant_flags`):
#     soft   --givens soft   (the given cells are generated, not clamped -- puzzle is a hint only)
#     lr3    --lr 3e-4
#     lr10   --lr 1e-3
#     long   (no flag) names a full-schedule run so it does not overwrite the early-stopped one
#
#   ./run_dfm_sizes.sh                                # default queue below, 8 GPUs
#   ./run_dfm_sizes.sh L8d512 L4d512 L16d256          # just those, spread over GPUs
#   GPUS="4 5 6 7" ./run_dfm_sizes.sh L8d256:lr3      # restrict to some GPUs
#   STOP_STEP=40000 ./run_dfm_sizes.sh ...            # default 30000; schedule length is unchanged
#
# Runs are dealt round-robin to the GPUs in $GPUS; each GPU works through its share sequentially.
# Run names: dfm_L<l>d<d>_unif_sc2t[_<tag>]_cfg_1k; logs in logs/$SESSION/.
#   tail -f logs/$SESSION/runner.log
#   grep -o "\[step [0-9]*\] test_hard_exact_match=[0-9.]*" logs/$SESSION/dfm_L8d512_unif_sc2t_cfg_1k_seed1.log

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-dfmsizes}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
STOP_STEP="${STOP_STEP:-30000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

DEFAULT_RUNS=(L16d512 L8d512 L4d512 L16d256 L8d384 L8d256 L4d256 L16d768:soft
              L6d384 L12d384 L12d512 L6d512 L16d384 L8d512:soft L12d256)

variant_flags() {
    case "$1" in
        "")     echo "" ;;
        soft)   echo "--givens soft" ;;
        lr3)    echo "--lr 3e-4" ;;
        lr10)   echo "--lr 1e-3" ;;
        lr3long) echo "--lr 3e-4" ;;         # lr3 under a distinct name, for STOP_STEP=60000 reruns
        long)   echo "" ;;                  # name-only tag: same recipe, run with STOP_STEP=83200
        *)      echo "unknown variant tag: $1" >&2; return 1 ;;
    esac
}

# Everything but the size is the sc2t recipe from run_dfm_seeds.sh: uniform prior, self-cond with 2
# refinement passes, and the tuned sampler (128 steps, eta 10, guidance 5) for the in-training eval
# and best.pt selection, eta 0 as a secondary curve.
COMMON_ARGS=(
    --dataset-name ./downloaded-datasets/sudoku-extreme-1k
    --eval-dataset-name ./downloaded-datasets/sudoku-extreme-1k
    --prior uniform --self-cond --self-cond-passes 2
    --head-dim 64 --norm-eps 1e-6 --rope-theta 10000.0 --qk-norm --pos-embed rope2d
    --forward-dtype bfloat16
    --sample-steps 128 --noise-scale 10 --eval-noise-scales 10 0 --guidance 5.0
    --conditional --eval-split test_hard --eval-samples 512
    --cond-dropout 0.1
    --train-steps 83200 --local-batch-size 768
    --lr 1e-4 --lr-warmup-steps 2000 --lr-min-ratio 0.1
    --weight-decay 0.1 --grad-clip 0.5 --beta1 0.9 --beta2 0.99 --ema 0.999
    --log-interval 50 --eval-interval 2500
    --wandb-project sudoku
    --wandb-keys eval/test_hard_exact_match eval/test_hard_exact_match_eta0 train/exact_match train/per_position_accuracy
)

# ---------------------------------------------------------------- worker (one GPU, runs in sequence)
if [[ "${1:-}" == "--worker" ]]; then
    gpu="$2"; shift 2
    cd "$REPO" || exit 1
    logdir="logs/$SESSION"
    mkdir -p "$logdir"
    failed=0
    for run in "$@"; do
        IFS=: read -r size tag seed <<< "$run"
        seed="${seed:-1}"
        [[ "$size" =~ ^L([0-9]+)d([0-9]+)$ ]] || { echo "bad size: $size" >&2; failed=1; continue; }
        layers="${BASH_REMATCH[1]}"; hidden="${BASH_REMATCH[2]}"
        tag_flags="$(variant_flags "${tag:-}")" || { failed=1; continue; }
        name="dfm_${size}_unif_sc2t${tag:+_$tag}_cfg_1k"
        # shellcheck disable=SC2206
        extra=($tag_flags $EXTRA_ARGS)
        [[ -n "$STOP_STEP" ]] && extra+=(--stop-step "$STOP_STEP")

        echo "=== gpu$gpu $name seed=$seed START $(date '+%F %T') ==="
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$PYTHON" experiments/dfm_sudoku.py \
                --num-layers "$layers" --hidden-size "$hidden" --intermediate-size $((4 * hidden)) \
                --run-name "$name" --seed "$seed" \
                "${COMMON_ARGS[@]}" "${extra[@]}" \
            > "$logdir/${name}_seed${seed}.log" 2>&1
        rc=$?
        echo "=== gpu$gpu $name seed=$seed END exit=$rc $(date '+%F %T') ==="
        if [[ $rc -ne 0 ]]; then
            failed=1
            echo "    FAILED -- full output in $logdir/${name}_seed${seed}.log:"
            tail -n 15 "$logdir/${name}_seed${seed}.log" | sed 's/^/    | /'
        fi
    done
    echo "=== gpu$gpu queue done, failed=$failed $(date '+%F %T') ==="
    exit $failed
fi

# ---------------------------------------------------------------- launcher
command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
RUNS=("$@")
[[ ${#RUNS[@]} -eq 0 ]] && RUNS=("${DEFAULT_RUNS[@]}")
for run in "${RUNS[@]}"; do
    [[ "$run" =~ ^L[0-9]+d[0-9]+(:[a-z0-9]*)?(:[0-9]+)?$ ]] || { echo "bad run spec: $run (want L<layers>d<hidden>[:<tag>][:<seed>])"; exit 1; }
    variant_flags "$(cut -d: -f2 -s <<< "$run")" >/dev/null || exit 1
done
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists.  tmux attach -t $SESSION  /  tmux kill-session -t $SESSION"
    exit 1
fi
mkdir -p "$REPO/logs/$SESSION"
read -r -a gpu_list <<< "$GPUS"
declare -A queue
for i in "${!RUNS[@]}"; do
    g="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    queue[$g]+=" ${RUNS[$i]}"
done
# One tmux window per GPU; the session's first window is created with the first GPU's queue.
first=1
for g in "${gpu_list[@]}"; do
    [[ -z "${queue[$g]:-}" ]] && continue
    cmd="SESSION='$SESSION' STOP_STEP='$STOP_STEP' EXTRA_ARGS='$EXTRA_ARGS' PYTHON='$PYTHON' '$REPO/run_dfm_sizes.sh' --worker $g ${queue[$g]} 2>&1 | tee -a '$REPO/logs/$SESSION/runner.log'"
    if [[ $first -eq 1 ]]; then
        tmux new-session -d -s "$SESSION" -n "gpu$g" -c "$REPO" "$cmd"; first=0
    else
        tmux new-window -t "$SESSION" -n "gpu$g" -c "$REPO" "$cmd"
    fi
    echo "gpu$g:${queue[$g]}"
done
echo "Launched ${#RUNS[@]} run(s) in tmux session '$SESSION'.  tail -f logs/$SESSION/runner.log"

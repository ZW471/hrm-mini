#!/usr/bin/env bash
#
# Discrete flow matching runs (`dfm_113m_{mask,unif}_cfg_{1k,full}`), the categorical counterpart
# of run_flow_seeds.sh: same 113m backbone, same 768 x 83,200 HRM-matched budget, same conditioning
# and guidance. Only the state space (tokens, not whitened one-hots) and the loss (cross-entropy on
# p(x_1 | x_t), not velocity MSE) differ.
#
#   ./run_dfm_seeds.sh                          # default queue below
#   ./run_dfm_seeds.sh mask:1k:1 unif:full:1    # just those
#   SESSION=dfm2 ./run_dfm_seeds.sh             # other session name
#   STOP_STEP=35000 ./run_dfm_seeds.sh unif:1k:1:sc unif:1k:1:wd1
#       -> tagged variants (flags per tag in `variant_flags` below), stopped early with the LR
#          schedule unchanged; run names dfm_113m_unif_sc_cfg_1k, dfm_113m_unif_wd1_cfg_1k.
#          EXTRA_ARGS adds flags to every run in the queue.
#   Tuned-sampler eval curves, 1k early-stopped then full schedule, chained in one session:
#       tmux new-session -d -s dfmtuned -c "$PWD" \
#           'STOP_STEP=25000 SESSION=dfmtuned ./run_dfm_seeds.sh --worker unif:1k:1:sc2t unif:1k:2:sc2t unif:1k:3:sc2t;
#            SESSION=dfmtuned ./run_dfm_seeds.sh --worker unif:full:1:sc2t unif:full:2:sc2t unif:full:3:sc2t'
#
# Logs land in logs/$SESSION/. Watch with:  tmux attach -t dfmseeds   (Ctrl-b then d to detach)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-dfmseeds}"
NPROC="${NPROC:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
STOP_STEP="${STOP_STEP:-}"

DEFAULT_RUNS=(mask:1k:1 unif:1k:1 mask:full:1 unif:full:1)

variant_flags() {
    case "$1" in
        "")     echo "" ;;
        sc)     echo "--self-cond" ;;
        wd1)    echo "--weight-decay 1.0" ;;
        elbo)   echo "--loss-weight elbo" ;;
        scwd1)  echo "--self-cond --weight-decay 1.0" ;;
        scp8)   echo "--self-cond --self-cond-p 0.8" ;;
        sc2)    echo "--self-cond --self-cond-passes 2" ;;
        scg3)   echo "--self-cond --guidance 3.0" ;;
        # sc2 trained identically, but the in-training eval uses the tuned sampler from the
        # eval_dfm_sudoku.py sweep (128 steps, eta 10, guidance 5) instead of the untuned default,
        # with eta 0 kept as a secondary curve. Training itself is unchanged (guidance and eta are
        # sampling-only); only the eval curve and best.pt selection differ.
        sc2t)   echo "--self-cond --self-cond-passes 2 --sample-steps 128 --noise-scale 10 --eval-noise-scales 10 0 --guidance 5.0" ;;
        *)      echo "unknown variant tag: $1" >&2; return 1 ;;
    esac
}

COMMON_ARGS=(
    --num-layers 16 --hidden-size 768 --intermediate-size 3072 --head-dim 64
    --norm-eps 1e-6 --rope-theta 10000.0 --qk-norm --pos-embed rope2d
    --forward-dtype bfloat16
    --sample-steps 64 --noise-scale 0 --eval-noise-scales 0 3 10
    --conditional --eval-split test_hard --eval-samples 512
    --cond-dropout 0.1 --guidance 2.0
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
    if command -v uv >/dev/null; then
        LAUNCH=(uv run torchrun)
    else
        LAUNCH=("$REPO/.venv/bin/torchrun")
    fi

    failed=0
    for run in "$@"; do
        IFS=: read -r prior data seed tag <<< "$run"
        tag_flags="$(variant_flags "${tag:-}")" || { failed=1; continue; }
        tag="${tag:+_$tag}"
        prior_flag=$([[ "$prior" == "unif" ]] && echo uniform || echo mask)
        if [[ "$data" == "full" ]]; then
            train_set=./downloaded-datasets/sudoku-extreme
        else
            train_set=./downloaded-datasets/sudoku-extreme-1k
        fi
        name="dfm_113m_${prior}${tag}_cfg_${data}"
        # shellcheck disable=SC2206
        extra=($tag_flags $EXTRA_ARGS)
        [[ -n "$STOP_STEP" ]] && extra+=(--stop-step "$STOP_STEP")

        echo "=== $name seed=$seed START $(date '+%F %T') ==="
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "${LAUNCH[@]}" --nproc-per-node "$NPROC" experiments/dfm_sudoku.py \
                --dataset-name "$train_set" \
                --eval-dataset-name ./downloaded-datasets/sudoku-extreme-1k \
                --prior "$prior_flag" \
                --run-name "$name" --seed "$seed" \
                "${COMMON_ARGS[@]}" "${extra[@]}" \
            > "$logdir/${name}_seed${seed}.log" 2>&1
        rc=$?
        echo "=== $name seed=$seed END exit=$rc $(date '+%F %T') ==="
        if [[ $rc -ne 0 ]]; then
            failed=1
            echo "    FAILED -- full output in $logdir/${name}_seed${seed}.log:"
            tail -n 15 "$logdir/${name}_seed${seed}.log" | sed 's/^/    | /'
        fi
    done
    echo
    if [[ $failed -eq 0 ]]; then echo "All runs finished successfully. $(date '+%F %T')"
    else echo "One or more runs FAILED. $(date '+%F %T')"; fi
    exit $failed
fi

# ---------------------------------------------------------------- launcher
command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
RUNS=("$@")
[[ ${#RUNS[@]} -eq 0 ]] && RUNS=("${DEFAULT_RUNS[@]}")
for run in "${RUNS[@]}"; do
    [[ "$run" =~ ^(mask|unif):(full|1k):[0-9]+(:[a-z0-9]+)?$ ]] || { echo "bad run spec: $run (want mask|unif:full|1k:<seed>[:tag])"; exit 1; }
    variant_flags "$(cut -d: -f4 <<< "$run")" >/dev/null || exit 1
done
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists.  tmux attach -t $SESSION  /  tmux kill-session -t $SESSION"
    exit 1
fi
mkdir -p "$REPO/logs/$SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "SESSION='$SESSION' NPROC='$NPROC' EXTRA_ARGS='$EXTRA_ARGS' STOP_STEP='$STOP_STEP' '$REPO/run_dfm_seeds.sh' --worker ${RUNS[*]} 2>&1 | tee -a '$REPO/logs/$SESSION/runner.log'"
echo "Launched ${#RUNS[@]} run(s) sequentially in tmux session '$SESSION' on $NPROC GPUs:"
printf '  - %s\n' "${RUNS[@]}"
echo "  tail -f logs/$SESSION/runner.log"

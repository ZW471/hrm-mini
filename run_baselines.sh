#!/usr/bin/env bash
#
# Run the HRM baselines sequentially inside a detached tmux session, one config at a time
# so the GPUs are never contended.
#
#   ./run_baselines.sh                              # all four baselines, in order
#   ./run_baselines.sh ar_param_matched             # just one (or any subset)
#   EXTRA="epochs=1 seeds=[1]" ./run_baselines.sh   # extra hydra overrides for every run
#   SESSION=hrm2 NPROC=4 ./run_baselines.sh         # other session name / GPU count
#
# Logs land in logs/$SESSION/. Watch with:  tmux attach -t hrm   (Ctrl-b then d to detach)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-hrm}"
NPROC="${NPROC:-8}"
EXTRA="${EXTRA:-}"

DEFAULT_CONFIGS=(mae_param_matched mae_flops_matched ar_param_matched ar_flops_matched)

# ---------------------------------------------------------------- worker
# This branch is what actually runs inside the tmux session.
if [[ "${1:-}" == "--worker" ]]; then
    shift
    cd "$REPO" || exit 1

    logdir="logs/$SESSION"   # per session, so two launches never clobber each other's logs
    mkdir -p "$logdir"

    failed=0
    for config in "$@"; do
        echo "=== $config START $(date '+%F %T') ==="
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            uv run torchrun --nproc-per-node "$NPROC" train.py --config-name "$config" $EXTRA \
            > "$logdir/$config.log" 2>&1
        rc=$?
        echo "=== $config END exit=$rc $(date '+%F %T') ==="
        if [[ $rc -ne 0 ]]; then
            failed=1
            echo "    FAILED -- full output in $logdir/$config.log:"
            tail -n 15 "$logdir/$config.log" | sed 's/^/    | /'
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

CONFIGS=("$@")
[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=("${DEFAULT_CONFIGS[@]}")

for config in "${CONFIGS[@]}"; do
    [[ -f "$REPO/config/$config.yaml" ]] || { echo "no such config: config/$config.yaml"; exit 1; }
done

# `-t =name` forces an exact match; a bare name would also match e.g. "hrm" against "hrmtest"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  tmux attach -t $SESSION          # see what it is doing"
    echo "  tmux kill-session -t $SESSION    # stop it first"
    echo "  SESSION=other ./run_baselines.sh # or launch under a different name"
    exit 1
fi

mkdir -p "$REPO/logs/$SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "SESSION='$SESSION' NPROC='$NPROC' EXTRA='$EXTRA' '$REPO/run_baselines.sh' --worker ${CONFIGS[*]} 2>&1 | tee '$REPO/logs/$SESSION/runner.log'"

echo "Launched ${#CONFIGS[@]} run(s) sequentially in tmux session '$SESSION' on $NPROC GPUs:"
printf '  - %s\n' "${CONFIGS[@]}"
[[ -n "$EXTRA" ]] && echo "  hydra overrides: $EXTRA"
cat <<EOF

  tmux attach -t $SESSION                # watch live (Ctrl-b then d to detach)
  tail -f logs/$SESSION/runner.log       # which config is running / how each ended
  tail -f logs/$SESSION/<config>.log     # training output of one run
  tmux kill-session -t $SESSION          # stop everything

The session closes itself when the last run finishes; logs/$SESSION/runner.log keeps the summary.
EOF

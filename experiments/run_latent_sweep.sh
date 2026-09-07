#!/usr/bin/env bash
#
# Launch the latent flow-matching sweep: one configuration per GPU, all in parallel, inside a
# detached tmux session. Each run trains a VAE and then a flow in its latent space.
#
#   experiments/run_latent_sweep.sh                      # all 8 configs on GPUs 0..7
#   EXTRA="--vae-steps 5000 --flow-steps 5000" experiments/run_latent_sweep.sh   # quick version
#   SESSION=latent2 experiments/run_latent_sweep.sh      # another session name
#
# Logs: logs/$SESSION/<run name>.log.  Watch with: tmux attach -t latent

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-latent}"
EXTRA="${EXTRA:-}"

# "<latent_dim> <kl_beta>", one per GPU. The two axes that decide whether this works at all:
# how much room the latent has, and how hard the KL pushes it towards the prior (too hard ->
# posterior collapse, too soft -> a latent distribution the flow has to work harder to match).
# "<latent_tokens> <latent_dim> <kl_beta>". Token counts are deliberately generic (1, 4, 16, 64) --
# never 9 or a 3x3 layout, which would hand the model the board's block structure.
# beta is the whole ballgame. At 1e-5..1e-3 the autoencoder reconstructs perfectly (exact match
# 1.000) and the latent still carries thousands of nats -- a decoder that sharp rejects anything
# the flow produces, and every such run generated exactly zero valid boards. These betas compress
# the latent towards the ~50 nats the board distribution actually needs, which is what gives the
# flow a target it can hit.
# "<latent_tokens> <latent_dim> <kl_beta> <decoder_noise>".
# Tying the decoder's error budget to the posterior width failed in both directions: a sharp
# posterior (beta<=1e-3) reconstructs perfectly but rejects every latent the flow produces, while
# a wide one (beta 0.03-0.1) makes 75% of the latent variance pure noise and the flow collapses to
# the trivial N(0, I) solution (loss 1.57 ~= the Gaussian optimum, zero valid boards).
# So keep the posterior narrow (small beta -> high-SNR means for the flow to model) and buy the
# decoder's robustness separately with decoder-side noise augmentation.
#
# sigma is the flow's error budget: the decoder is trained on per-dim noise sigma, so it absorbs
# per-dim latent error of about that size. Treating the latent as a Gaussian channel, it carries
# (D/2)*log2(1 + 1/sigma^2) bits, and the board distribution needs ~50 -- so the right move is the
# LARGEST sigma whose capacity still clears that, which means a wide latent, not a narrow one.
# The latent is pinned to unit RMS per token, so sigma is a real signal-to-noise ratio: capacity
# (D/2)*log2(1 + 1/sigma^2) is 594 bits at sigma=0.5, 256 at sigma=1, 82 at sigma=2 for D=512, all
# comfortably above the ~50 the board distribution needs. sigma=0 is the no-augmentation control.
# Probing the gap between sigma=1.0 (decoder reconstructs, flow lands outside the basin) and
# sigma=1.5 (flow lands inside, decoder no longer reconstructs). Is there a sigma where both hold?
# The last field is --vae-steps: sigma>=1.5 also gets a doubled VAE budget, since its capacity
# (~136 bits at D=512) is far above the ~50 the task needs, so its failure may be optimisation.
CONFIGS=(
    "16 32 1e-3 1.1 30000"
    "16 32 1e-3 1.2 30000"
    "16 32 1e-3 1.3 30000"
    "16 32 1e-3 1.4 30000"
    "16 32 1e-3 1.2 60000"
    "16 32 1e-3 1.5 60000"
    "16 64 1e-3 1.2 30000"
    "16 32 1e-3 1.0 60000"
)

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attach, or kill it first:"
    echo "  tmux attach -t $SESSION ; tmux kill-session -t $SESSION"
    exit 1
fi

mkdir -p "$REPO/logs/$SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" "sleep infinity"

gpu=0
for cfg in "${CONFIGS[@]}"; do
    read -r tokens dim beta dnoise vsteps <<< "$cfg"
    vsteps="${vsteps:-30000}"
    name="latent_t${tokens}x${dim}_kl${beta}_dn${dnoise}_v${vsteps}"
    cmd="cd '$REPO' && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        uv run python experiments/latent_flow_sudoku.py \
        --latent-tokens $tokens --latent-dim $dim --kl-beta $beta --decoder-noise $dnoise \
        --vae-steps $vsteps \
        --run-name $name $EXTRA \
        > 'logs/$SESSION/$name.log' 2>&1; echo \"=== $name exit=\$? \$(date '+%F %T') ===\""
    tmux new-window -t "$SESSION" -n "gpu$gpu" -c "$REPO" "$cmd; sleep infinity"
    echo "  GPU $gpu -> $name"
    gpu=$((gpu + 1))
done

cat <<EOF

Launched ${#CONFIGS[@]} runs in tmux session '$SESSION' (one per GPU).

  tmux attach -t $SESSION                       # watch (Ctrl-b then n/p to switch windows)
  grep -h "^\[vae\]" logs/$SESSION/*.log        # VAE diagnostics once stage 1 finishes
  grep -h "flow step" logs/$SESSION/*.log       # latent-flow eval metrics
  tmux kill-session -t $SESSION                 # stop everything
EOF

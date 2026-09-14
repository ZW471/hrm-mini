# Cycle FF layers on a recurrent transformer

## The question

On Sudoku-Extreme 1k, HRM beats a recurrent transformer that is matched on **both** parameters
(12.59M) and block-forwards per pass (28):

| | best test_hard | at epoch | final |
|---|---|---|---|
| `tuned_hrm` | 81.3 – 82.3 % | 11 – 19 | ~80.6 % |
| `tuned_rt`  | 70.5 ± 0.5 % (n=4) | 6 | ~58.5 % |

The structural difference is that HRM runs a second, slower block (its "H level") between groups of
fast cycles, while the RT applies one identical core every cycle. **Does adding such a layer to a
recurrent transformer recover any of that 11-point gap?** We call the added component a
**cycle FF layer**.

Note what the baseline curves actually say: the RT reaches 70.7 % at epoch 6 with train
exact-match already at 1.000, then overfits downhill for 14 epochs. HRM never fully fits the train
set and keeps climbing. So on 1k the gap is a **generalisation** gap, not a capacity gap — which is
why adding parameters is not obviously the right medicine, and why `best` (not `last`) is the
number to compare.

## The architecture

`arch/rt_cff.py`, class `RecurrentTransformerCFF`:

```
z_H = 0
for i in 1..cycles:                       # cycles = 7
    z   = core(z + z_H + x)               # the RT core: 4 shared layers
    if i % cff_period == 0:
        z_H = cff_i(z_H + z)              # cff_gated: z_H + g * cff_i(z_H + z)
logits = lm_head(z)
```

`z_H` is the slow state; it is fed back into the core's input on the next cycle and carried across
recurrent steps (detached, like the RT's `z`).

| config key | meaning |
|---|---|
| `cff_layers` | layers per cycle FF layer. **0 disables it**, leaving the plain RT (used for controls) |
| `cff_type` | `mlp` = post-norm feed-forward, no attention. `block` = full transformer block, i.e. what HRM's H level actually is |
| `cff_intermediate_size` | FF width of the cycle FF layer (default: the core's) |
| `cff_tied` | `True` = one layer shared across all cycles (HRM ties its H level). `False` = one layer per cycle |
| `cff_period` | run a cycle FF layer every N core cycles (HRM's `L_cycles` per H cycle) |
| `cff_mode` | `inject` = maintain the slow state `z_H`. `inline` = write straight into the fast state `z`, carrying nothing across cycles |
| `cff_gated` | wrap in a zero-init per-channel gate. Makes a run resuming from a pretrained core an **exact no-op at init** — verified `max\|logit diff\| = 0`. Pointless from scratch |
| `cff_lr_mult` | LR multiplier for the cycle FF parameters vs the rest of the model |
| `pretrained_ckpt` / `freeze_core` | resume from a trained RT; optionally freeze core + embed + lm_head |

## Results so far (Sudoku-Extreme 1k, `test_hard` exact match)

### Trained from scratch — all seed 1 only

| arm | params | vs RT | best | @ep | last |
|---|---|---|---|---|---|
| `tuned_rt` (baseline, n=4) | 12.59M | — | **70.50 ± 0.46** | 6 | 58.5 |
| `cff_mlp_tied` (n=1) | 13.64M | +8.3 % | 68.89 | 7 | *(killed at ep9)* |
| `cff_block_tied` (n=1) | 14.69M | +16.7 % | 70.00 | 6 | 57.9 |
| `cff_mlp_untied` (n=1) | 19.93M | +58.3 % | 69.71 | 5 | 38.7 |
| `cff_block_untied` (n=3) | 27.27M | +116.6 % | 69.95 ± 2.57 | 6 | 31.7 |
| `tuned_hrm` (reference) | 12.59M | — | 81.3 – 82.3 | 11–19 | 80.6 |

**No from-scratch arm beats the RT baseline.** `cff_block_untied` looked like the exception at n=1
(71.93, +1.43) — replicated at 3 seeds it gives 71.63 / 71.23 / **66.99**, i.e. 69.95 ± 2.57, which
is 0.55 *below* the RT mean. The original run was a lucky seed. What the extra parameters actually
buy is variance: a 2.57 spread against the RT's own 0.46, from an arm whose seeds otherwise agree on
peaking at epoch 6. Pooling all four runs of this arm gives 70.45 ± 2.32 — the same null.

The late collapse replicated on every seed (~30 % at ep20 vs the RT's 58 %), so it is a property of
the arm, not of one run. That is what +117 % parameters on 1000 puzzles predicts. Untying costs
parameters but **not** FLOPs — `block_tied` and `block_untied` do the same arithmetic per pass.

The 1k conclusion for the from-scratch setting is therefore negative: on a generalisation gap,
bolting a cycle FF layer onto a randomly-initialised RT does not close any of it.

### Resumed from a trained RT

Zero-gated, so each arm provably starts at exactly its checkpoint's accuracy. **Two different
checkpoints are in play** — the original runs used a 70.87 % `tuned_rt` (`nonchalant-malamute`,
since deleted); the 3-seed replication used the surviving `malachite-saluki`, which evals to
**69.77 %**. Rows are marked accordingly and `vs init` is against each row's own init, so the
`vs init` column compares across rows but the raw `best` column does not.

| arm | core | init | best | @ep | last | vs init |
|---|---|---|---|---|---|---|
| `rt_sft` — **no cycle FF** (control, n=3) | trained | 69.77 | 69.38 ± 0.10 | 1 | 63.5 | −0.39 |
| `rt_sft` (original, n=1) | trained | 70.87 | 70.02 | 1 | 61.7 | −0.85 |
| graft, `inject`, frozen core (n=1) | frozen | 70.87 | 64.23 | 1 | 57.3 | −6.64 |
| graft, `inline`, frozen core (n=1) | frozen | 70.87 | 69.80 | 8 | 69.4 | −1.07 |
| `cff_sft` — `inject`, core fine-tuned (n=3) | trained | 69.77 | **72.82 ± 0.50** | 3 | 66.0 | **+3.05** |
| `cff_sft` (original, n=1) | trained | 70.87 | 72.31 | 4 | 58.7 | +1.44 |
| `cff_sft_lr10` (n=1) | trained | 69.77 | 69.87 | 1 | 63.7 | +0.10 |
| **`cff_sft_untied`** (n=3) | trained | 69.77 | **73.53 ± 0.31** | 2 | 66.3 | **+3.76** |

**This is the one setting where the cycle FF layer clearly earns its place, and it replicates.**
Against a matched control resumed from the same checkpoint for the same budget, `cff_sft` wins by
**+3.44** (72.82 ± 0.50 vs 69.38 ± 0.10, n=3 each). The two arms do not overlap at any seed, and
each is internally tight — an order of magnitude tighter than the from-scratch arm. The shapes
differ as much as the peaks: every `cff_sft` seed climbs above its init and peaks at epoch 3, every
`rt_sft` seed peaks at epoch 1 *below* its init and only decays. The second training budget on its
own makes the model worse; the cycle FF layer is what converts it into a gain.

`cff_sft_lr10` answers the "is it just undertrained?" question with a clear **no**: training the
layer 10x faster than the core erases the entire gain (69.87, +0.10) and reproduces the no-layer
control's shape exactly — peak at epoch 1, monotone decay. The layer only pays off when it
co-adapts *with* the core at the same rate. This also rules out the reading that the graft works
by adding fresh capacity, since capacity trained faster should have helped more, not less.

**`cff_sft_untied` is the best configuration found: 73.53 ± 0.31 (73.61 / 73.19 / 73.80), i.e.
+4.15 over the matched control.** Untying beats tied `cff_sft` by +0.71, but that margin is *not*
established at n=3 — the seed ranges still touch (untied's worst 73.19 vs tied's best 73.27) and a
Welch t-test gives t≈2.1, p≈0.12. Treat it as suggestive, not proven, and note the price: 14.7M
added parameters against tied's 2.1M, for a gain roughly the size of one seed's noise. It does
consistently peak an epoch earlier (2 vs 3), so the per-cycle layers mostly buy faster adaptation.

HRM ties its H level, and nothing here contradicts that choice.

Freezing the core fails both ways: `inject` degrades from the first eval, `inline` sits flat near
its init for 20 epochs without ever exceeding it. Those two rows remain n=1.

## Setup on a new machine

```bash
pip install -r requirements.txt          # or: uv sync
wandb login

mkdir -p downloaded-datasets
hf download --repo-type dataset --local-dir ./downloaded-datasets/sudoku-extreme-1k sapientinc/sudoku-extreme-1k
# only needed for full-data arms:
hf download --repo-type dataset --local-dir ./downloaded-datasets/sudoku-extreme    sapientinc/sudoku-extreme
```

### The pretrained checkpoint (required for every `*_sft` arm)

`checkpoints/` is gitignored, so it does **not** survive a clone. Either copy a `tuned_rt`
checkpoint across, or train one (~1 h on 8 GPUs):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run torchrun --nproc-per-node 8 \
    train.py --config-name tuned_rt seeds=[1]
```

Then point the resume arms at it. `config/cff_sft.yaml` reads `$RT_CKPT`:

```bash
export RT_CKPT="checkpoints/tuned_rt <coolname>/seed_1/best.pt"
```

Use **`best.pt`, not `last.pt`** — the RT peaks at epoch 6 (70.9 %) and overfits down to 59 % by
epoch 20, so `last.pt` would start the graft 12 points lower. `best.pt` holds EMA weights taken at
the best eval. Sanity-check whatever you point at:

```bash
uv run python eval.py --ckpt "$RT_CKPT" --split test_hard   # expect 0.698-0.709
```

**Which checkpoint you land on matters and is not fixed.** `tuned_rt` seeds peak anywhere in
69.8-70.8, so a fresh one can start the graft a point either way. The 3-seed replication used
`malachite-saluki` (**0.6977**, the weakest of the four RT seeds) because the 70.87 %
`nonchalant-malamute` used by the original runs no longer exists. This does not affect the
`cff_sft` vs `rt_sft` comparison — both resume from the same weights — but it does shift every
absolute `best` and every `vs init`, so record the init alongside any new resume result.

## Running the experiments

`run_baselines.sh` runs configs **sequentially** in a detached tmux session, so the GPUs are never
contended:

```bash
SESSION=cff NPROC=8 ./run_baselines.sh cff_block_untied cff_sft rt_sft
tail -f logs/cff/runner.log        # which arm is running, how each ended
tmux kill-session -t cff           # stop everything
```

To kill just the current arm and let the queue advance to the next:

```bash
pkill -f "config-name cff_block_untied"
```

### Priority order

Priorities 1–3 are **done** (2026-09-10): `cff_block_untied` failed to replicate, `cff_sft` and its
control replicated at 3 seeds each, `cff_sft_lr10` ran at n=1, and `cff_sft_untied` — the best
configuration found — ran at 3 seeds. The queue took 3h40m on 8 H100s, plus 45 min for the
3-seed `cff_sft_untied` follow-up:

```bash
SESSION=cff NPROC=8 ./run_baselines.sh cff_block_untied cff_sft rt_sft cff_sft_lr10 cff_sft_untied
```

What is left, now that the graft is the only surviving positive result:

1. **The graft on full Sudoku-Extreme.** Every 1k arm peaks by epoch 6 and then overfits, so the
   whole sweep is measuring a generalisation gap on 1000 puzzles. On full data nothing overfits and
   the question becomes capability rather than memorisation — the setting where a +3.44 graft would
   actually matter.
2. **`cff_period > 1`** — nothing here has real timescale *separation*: the cycle FF layer fires
   every cycle, so the "slow" state is not slow. This is the structural difference from HRM that
   the sweep has not yet tested. Also `cff_layers=2`.
3. **An LR sweep between 1x and 10x** on the cycle FF layer. 10x destroys the gain and 1x gives
   +3.05; nobody has looked in between, and the fact that the layer must move *with* the core is
   the most mechanistically suggestive thing the sweep has produced.

Any knob can be overridden without a new config:

```bash
uv run torchrun --nproc-per-node 8 train.py --config-name cff_sft \
    arch.cff_lr_mult=10 arch.cff_period=7 seeds=[1] epochs=8
```

### When to kill a run

Every arm on 1k peaks and then overfits, so a run is finished as soon as it has peaked. Kill it when
it has declined for ~3 consecutive evals **and** its peak is below the relevant baseline
(70.5 % from scratch, 70.87 % for a resume). For reference, per-epoch trajectories:

```
tuned_rt   2.3 23.9 60.8 68.3 70.7 70.7 69.5 68.7 67.5 66.7 65.9 64.6 ...   (peaks ep6)
tuned_hrm  0.6 16.7 34.1 56.5 63.8 69.2 73.1 76.3 78.5 79.0 80.5 81.0 ...   (still climbing)
```

An arm tracking the first line is a null result; only one tracking the second is interesting.

## Collecting results

```bash
uv run python experiments/collect_cff_results.py
uv run python experiments/collect_cff_results.py --arms cff_sft rt_sft
uv run python experiments/collect_cff_results.py --project <entity>/<project>   # if not the default
```

Prints mean ± stdev of `best` across seeds plus every per-epoch curve. W&B project is `sudoku`
(taken from `data.name`); pass `--project` if your entity differs.

## Gotchas

- **Eval recurrence is `cycles_per_data`.** Inference runs that many recurrent passes. Overriding it
  for a quick smoke test (`cycles_per_data=1`) drops accuracy from ~71 % to ~6 % — that is the
  override, not a broken model.
- **8-epoch runs are not a different schedule.** `lr_min_ratio: 1.0` makes the post-warmup LR
  constant and warmup is step-based, so an 8-epoch run is exactly the first 8 epochs of a 20-epoch
  one. Safe to shorten any arm that peaks early.
- **Compare `best`, not `last`.** Everything overfits 1k; `last` measures the fall, not the learning.
- **Two checkpoint dirs can share an arm name** (`checkpoints/<arm> <coolname>/`), including ones
  left by smoke tests. Sort by mtime before reading weights, or you will analyse the wrong run.
- **`cff_layers: 0`** turns the module into the plain RT — that is how the `rt_sft` control is built,
  so control and treatment differ in exactly one thing.
- Old `graft_rt_*` checkpoints predate the rename to `cff_*` and no longer load; their curves live
  in W&B.
- **A long-lived tmux server does not inherit your shell's env.** `run_baselines.sh` launches into
  tmux, so if the server was started before you exported `RT_CKPT` (or before `uv` was on `PATH`),
  the resume arms silently fall back to the default checkpoint path and die on `FileNotFoundError`.
  Either `tmux kill-server` first, or launch the worker with the env set explicitly:
  ```bash
  tmux new-session -d -s cff -c "$PWD" \
      "PATH=/path/to/uv:\$PATH RT_CKPT='checkpoints/tuned_rt <coolname>/seed_1/best.pt' \
       SESSION=cff NPROC=8 bash run_baselines.sh --worker cff_sft rt_sft | tee logs/cff/runner.log"
  ```
  Confirm what a run actually loaded before trusting it:
  `grep pretrained_ckpt "checkpoints/<arm> <coolname>/seed_1/model_config.json"`.

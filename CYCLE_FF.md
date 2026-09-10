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
| `cff_mlp_tied` | 13.64M | +8.3 % | 68.89 | 7 | *(killed at ep9)* |
| `cff_block_tied` | 14.69M | +16.7 % | 70.00 | 6 | 57.9 |
| `cff_mlp_untied` | 19.93M | +58.3 % | 69.71 | 5 | 38.7 |
| `cff_block_untied` | 27.27M | +116.6 % | **71.93** | 6 | 34.0 |
| `tuned_hrm` (reference) | 12.59M | — | 81.3 – 82.3 | 11–19 | 80.6 |

`cff_block_untied` is the only from-scratch arm above the RT baseline: **+1.43 over the RT mean**,
about 3x the RT's own seed spread — but it is **n=1**, so this is the headline number that needs
replication. It also collapses hardest late (34 % vs the RT's 58 %), which is what +117 % parameters
on 1000 puzzles predicts. Untying costs parameters but **not** FLOPs — `block_tied` and
`block_untied` do the same arithmetic per pass.

### Resumed from a trained RT (all start from the same 70.87 % checkpoint)

Zero-gated, so each provably starts at exactly 70.87 %.

| arm | core | best | @ep | last | vs init |
|---|---|---|---|---|---|
| `rt_sft` — **no cycle FF** (control) | trained | 70.02 | 1 | 61.7 | −0.85 |
| graft, `inject`, frozen core | frozen | 64.23 | 1 | 57.3 | −6.64 |
| graft, `inline`, frozen core | frozen | 69.80 | 8 | 69.4 | −1.07 |
| `cff_sft` — `inject`, core fine-tuned | trained | **72.31** | 4 | 58.7 | **+1.44** |

**This is the one setting where the cycle FF layer clearly earns its place.** `cff_sft` beats the
matched no-cycle-FF control by **+2.29**, so the gain is not merely the second training budget.
Freezing the core fails both ways: `inject` degrades from the first eval, `inline` sits flat near
its init for 20 epochs without ever exceeding it. All n=1 — replication is the priority.

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
uv run python eval.py --ckpt "$RT_CKPT" --split test_hard   # expect ~0.709
```

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

1. **`cff_block_untied`, 3 seeds** — replicate the only from-scratch arm above baseline (+1.43,
   n=1). 20 epochs, ~55 min/seed.
   ```bash
   SESSION=cff NPROC=8 ./run_baselines.sh cff_block_untied
   ```
2. **`cff_sft` and `rt_sft`, 3 seeds each** — replicate the +2.29 and its matched control. This is
   the strongest signal in the whole sweep. 8 epochs, ~22 min/seed.
   ```bash
   SESSION=sft NPROC=8 ./run_baselines.sh cff_sft rt_sft
   ```
3. **`cff_sft_lr10`** (`cff_lr_mult=10`) — is a randomly-initialised layer behind a zero gate simply
   undertrained at a converged backbone's LR? **`cff_sft_untied`** — per-cycle layers in the setting
   that works.
4. Untried and worth a look: `cff_period > 1` (nothing here has real timescale *separation* — the
   cycle FF layer fires every cycle, so the "slow" state is not slow); `cff_layers=2`; the graft on
   full Sudoku-Extreme, where nothing overfits and the question becomes capability rather than
   memorisation.

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

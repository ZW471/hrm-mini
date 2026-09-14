# What a cycle FF layer is, and what it actually bought

A companion to `CYCLE_FF.md`. That file is the lab notebook — configs, commands, per-epoch curves.
This one explains the idea, the code that implements it, and what the experiments concluded.

---

## 1. The gap this is trying to explain

On Sudoku-Extreme 1k, HRM beats a recurrent transformer that is matched on **both** parameters
(12.59M) and block-forwards per pass (28):

| | best `test_hard` | at epoch | final |
|---|---|---|---|
| `tuned_hrm` (n=6) | 80.65 ± 2.40 | 11–19 | 79.4 |
| `tuned_rt` (n=4) | 70.50 ± 0.46 | 6 | 58.5 |

Eleven points, with no parameter or FLOP advantage on either side. The remaining difference is
*structural*: HRM runs a second, slower block (its "H level") between groups of fast cycles, while
the RT applies one identical core every cycle.

**The critical detail is what kind of gap it is.** The RT hits train exact-match 1.000 by epoch 6
and then overfits downhill for 14 epochs. HRM never fully fits the 1000 training puzzles and keeps
climbing. So this is a **generalisation** gap, not a capacity gap. That single fact drives
everything below: it is why adding parameters is not obviously the right medicine, and why every
comparison uses `best` rather than `last`.

## 2. The idea

Give the recurrent transformer the one thing it structurally lacks — a second state that updates on
a slower schedule than the core — and see how much of the 11 points comes back. The added component
is a **cycle FF layer**.

`arch/rt_cff.py`, class `RecurrentTransformerCFF`. The plain RT is `z = core(z + x)` repeated
`cycles` times. The modified loop is:

```
z_H = 0                                   # the slow state
for i in 1..cycles:                       # cycles = 7
    z   = core(z + z_H + x)               # fast state: 4 shared layers, as before
    if i % cff_period == 0:
        z_H = cff_i(z_H + z)              # slow state update
logits = lm_head(z)
```

Two things make `z_H` a genuinely separate timescale rather than just extra depth:

1. It is **fed back into the core's input** on the next cycle, so it steers the fast computation
   instead of just post-processing it.
2. It is **carried across recurrent steps** (detached, like the RT's own `z`), so it persists beyond
   a single forward pass.

The `inline` mode is the ablation that removes both properties — it writes straight into the fast
state (`z = z + g * cff(z)`) and carries nothing — which separates "second timescale" from "more
depth per cycle".

## 3. The knobs

| config key | meaning |
|---|---|
| `cff_layers` | layers per cycle FF layer. **0 disables it**, leaving the plain RT — this is how the controls are built |
| `cff_type` | `mlp` = post-norm feed-forward, no attention. `block` = full transformer block, i.e. what HRM's H level actually is |
| `cff_intermediate_size` | FF width of the cycle FF layer (default: the core's) |
| `cff_tied` | `True` = one layer shared across all cycles (HRM ties its H level). `False` = one per cycle |
| `cff_period` | run the layer every N core cycles (HRM's `L_cycles` per H cycle) |
| `cff_mode` | `inject` = maintain the slow state `z_H`. `inline` = write into the fast state, carry nothing |
| `cff_gated` | wrap in a zero-init per-channel gate |
| `cff_lr_mult` | LR multiplier for the cycle FF parameters vs the rest of the model |
| `pretrained_ckpt` / `freeze_core` | resume from a trained RT; optionally freeze core + embed + lm_head |

Two of these deserve more than a table row.

**`cff_gated` is what makes the graft measurable.** With the gate zero-initialised and `z_H`
starting at exactly zero, a run resuming from a pretrained RT begins *bit-for-bit identical* to that
checkpoint — verified `max|logit diff| = 0`. So "did the cycle FF layer help?" is answerable
directly against the checkpoint's own accuracy, with no confound from a perturbed starting point.
Training from scratch there is nothing to preserve, so the gate is off by default.

**`cff_tied` costs parameters but not FLOPs.** `block_tied` and `block_untied` do exactly the same
arithmetic per pass; untying only stops the seven cycles from sharing weights. That makes the pair a
clean test of whether the slow state wants a cycle-specific update schedule.

## 4. Two settings, and why they are not the same experiment

**From scratch.** Train the whole thing — core and cycle FF layer together — from random init, with
everything else matched to `tuned_rt`. Asks: does the structure help a model that has to learn the
task from nothing?

**Grafted (`*_sft`).** Take a *converged* RT checkpoint, splice in a zero-gated cycle FF layer, and
fine-tune. Asks a narrower question: given a model that has already plateaued and started
overfitting, does adding a slow timescale give it somewhere new to go?

The graft gets a second training budget the from-scratch arms do not, so it is **not** comparable to
`tuned_rt` directly. That is the entire reason `rt_sft` exists: the same checkpoint, resumed for the
same budget, with `cff_layers: 0` so the model is a plain RT again. Treatment and control differ in
exactly one thing. Any gain over that control is the layer, not the extra epochs.

## 5. What happened

### From scratch — nothing works

| arm | params | vs RT | best | @ep | last |
|---|---|---|---|---|---|
| `tuned_rt` (baseline, n=4) | 12.59M | — | **70.50 ± 0.46** | 6 | 58.5 |
| `cff_mlp_tied` (n=1) | 13.64M | +8.3 % | 68.89 | 7 | — |
| `cff_block_tied` (n=1) | 14.69M | +16.7 % | 70.00 | 6 | 57.9 |
| `cff_mlp_untied` (n=1) | 19.93M | +58.3 % | 69.71 | 5 | 38.7 |
| `cff_block_untied` (n=3) | 27.27M | +116.6 % | 69.95 ± 2.57 | 6 | 31.7 |

`cff_block_untied` looked like the exception at n=1 (71.93, +1.43 over the RT mean). **Replicated at
three seeds it gives 71.63 / 71.23 / 66.99 — 69.95 ± 2.57, which is 0.55 *below* baseline.** The
original run was a lucky seed.

What the extra parameters reliably buy is variance: a 2.57 seed spread against the RT's own 0.46,
from an arm whose seeds otherwise agree closely on peaking at epoch 6. The late collapse replicated
on every seed (~30 % at ep20 vs the RT's 58 %). On a generalisation gap, more capacity behaves
exactly as you would expect it to.

### Grafted — this one works

All six runs below resume from the same checkpoint (0.6977 on `test_hard`), zero-gated, so each
provably starts there:

| arm (n=3 each) | best | @ep | last | vs init |
|---|---|---|---|---|
| **`cff_sft`** — `inject`, core fine-tuned | **72.82 ± 0.50** | 3 | 66.0 | **+3.05** |
| `rt_sft` — no cycle FF (control) | 69.38 ± 0.10 | 1 | 63.5 | −0.39 |

**+3.44 for the layer over its matched control.** The two arms do not overlap at any seed, and each
is internally tight — an order of magnitude tighter than the from-scratch arm. The curve *shapes*
differ as much as the peaks: every `cff_sft` seed climbs above its init and peaks at epoch 3, every
`rt_sft` seed peaks at epoch 1 *below* its init and only decays from there.

That last point is the one worth holding onto: **the second training budget on its own makes the
model worse.** Resuming a converged, already-overfitting RT and training it further is a losing
move. The cycle FF layer is what converts that same budget into a gain.

### The mechanism, narrowed by two ablations

| arm | best | @ep | vs init |
|---|---|---|---|
| **`cff_sft_untied`** (n=3) | **73.53 ± 0.31** | 2 | **+3.76** |
| `cff_sft` (n=3) | 72.82 ± 0.50 | 3 | +3.05 |
| `cff_sft_lr10` (n=1) | 69.87 | 1 | +0.10 |

`cff_sft_lr10` trains the cycle FF layer 10× faster than the core it is grafted onto. The
hypothesis was benign: a randomly-initialised layer sitting behind a zero gate might simply be
undertrained at a converged backbone's learning rate.

**Instead, 10× erases the entire gain** — 69.87, and the curve collapses onto the no-layer control's
shape exactly (peak at epoch 1, monotone decay). Two readings fall out:

- It is **not** undertraining. Training it harder is strictly worse.
- It is **not** simply added capacity, either. Capacity trained faster should have helped more, not
  less. What the layer needs is to **co-adapt with the core at the same rate** — the core has to
  move to meet it. A slow state the backbone has not adjusted to is worth nothing.

`cff_sft_untied` gives each cycle its own layer instead of sharing one across all seven — 14.7M
added parameters against tied's 2.1M, at **identical FLOPs**. At three seeds (73.61 / 73.19 / 73.80)
it is **the best configuration found: 73.53 ± 0.31, +4.15 over the matched control.**

But the margin over tied `cff_sft` is +0.71 and is *not* established at this sample size: the seed
ranges still touch (untied's worst 73.19 against tied's best 73.27), and a Welch t-test gives
t≈2.1, p≈0.12. What untying does change reliably is *speed* — every untied seed peaks at epoch 2,
every tied seed at epoch 3. The read is that per-cycle layers mostly buy faster adaptation rather
than a higher ceiling, which is a weak reason to pay 7× the parameters. HRM ties its H level, and
nothing here contradicts that choice.

## 6. So how much of the 11 points came back?

About four, and only in the graft setting. The best arm, `cff_sft_untied` at 73.53, leaves roughly
seven of the eleven points unexplained against HRM's 80.65. The honest summary:

- **The slow timescale is worth something real** — +4.15 against a matched control, replicated, with
  non-overlapping seeds and a mechanism constrained by the LR ablation.
- **It is not what makes HRM work.** If the H level were the whole story, bolting one on should have
  closed most of the gap, and from scratch it closes none of it.
- **Everything here is measured on a generalisation gap on 1000 puzzles.** Every arm peaks by epoch
  6 and then overfits. That is a narrow regime to draw architectural conclusions from, and it is why
  the full Sudoku-Extreme graft is the most valuable untried experiment: on full data nothing
  overfits, and the question becomes capability rather than memorisation.

Also unresolved and structurally interesting: **nothing tested so far has real timescale
*separation*.** With `cff_period: 1` the cycle FF layer fires every single cycle, so the "slow"
state is not actually slow — it just has its own weights and its own feedback path. `cff_period > 1`
is the knob that would make it slow, and it has never been run.

## 7. What changed in this session (2026-09-10)

Ran the replication queue — `cff_block_untied`, `cff_sft`, `rt_sft`, `cff_sft_lr10`,
`cff_sft_untied` — 3h40m on 8 H100s, then `cff_sft_untied` at 3 seeds (45 min). All arms exit 0.

- **`CYCLE_FF.md`** — results tables rewritten with n=3 numbers. The from-scratch headline was
  reversed (the +1.43 did not replicate); the graft result was confirmed and strengthened
  (+2.29 → +3.44). Priority list updated to what is actually left.
- **`experiments/collect_cff_results.py`** — `SFT_INIT` corrected 0.7087 → 0.6977. It was
  understating every new `cff_sft` result by 1.1 points.

Two setup facts that cost time and are now documented in `CYCLE_FF.md`:

- **The original 70.87 % checkpoint (`nonchalant-malamute`) no longer exists.** The replication used
  `malachite-saluki` at **0.6977**, the weakest of the four `tuned_rt` seeds. This does not affect
  the `cff_sft` vs `rt_sft` comparison — both resume from the same weights — but it shifts every
  absolute number, so record the init alongside any new resume result.
- **A long-lived tmux server does not inherit your shell's environment.** `run_baselines.sh`
  launches into tmux, so `RT_CKPT` and `uv`'s `PATH` silently fail to propagate and the resume arms
  die on `FileNotFoundError`. Verify what a run actually loaded before trusting it:
  `grep pretrained_ckpt "checkpoints/<arm> <coolname>/seed_1/model_config.json"`.

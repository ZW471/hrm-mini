# What the flow-matching model is, and what it says about HRM

A companion to `outputs/flow_matching/README.md` (the results log) and `outputs/flow_trace/README.md`
(the mechanism study). This one explains the idea, the code that implements it, and what the two
HRM-matched runs — `flow_113m_cfg_1k` and `flow_113m_cfg_full` — concluded. §7 adds the **discrete**
counterpart (`experiments/dfm_sudoku.py`, runs `dfm_113m_*`), which on 1k puzzles goes from
50 % to **84.5 ± 1.2 % single-shot on the full `test_hard` split (n=3)** and overtakes HRM's
80.65 ± 2.40; §8 shrinks it to **HRM's own parameter count (13M), where it ties HRM — 81.5 ± 0.5 —
in a twentieth of the training compute and at 3.5× the inference wall clock**, and works out why.
The sampler's step count then turns out to be a test-time-compute dial: **the same 13M model gives
63.5 % at HRM's wall clock, 90 % at 7×, 95 % at 14× and 98.2 ± 0.6 % at 28×**, single sample, on
the full split — while HRM given more segments gains ~2 points per doubling. The visual abstract is
`flow_matching_architecture.{svg,png}`.

---

## 1. The question this is asking

Every other model in this repo maps a puzzle to a solution with a *latent* state: HRM carries
`z_L`/`z_H`, the recurrent transformer carries `z`, and both read the answer off that state with an
`lm_head` trained by cross-entropy. The flow model removes the latent altogether. **The board is
the state.** A solved Sudoku is a point in a *continuous* 81 × 9 space (this is continuous flow
matching — the digits are relaxed to whitened one-hot channels, the prior is Gaussian, the sampler
integrates a real-valued SDE; nothing is categorical until the final argmax), the model is a
velocity field over that space, and solving a puzzle means integrating from Gaussian noise to a
board while the given cells are held fixed. §7 is what happens when the board is kept discrete.

That makes it a very different kind of baseline from the ones in `CYCLE_FF_EXPLAINED.md`. It has
no recurrence, no solver, no autoregressive factorisation, and nothing in it knows the rules of
Sudoku. What it *does* have is iteration at inference time — 64 integrator steps — which is the
one property it shares with HRM. So the comparison asks: how much of what HRM does is "iterate on
a state", and how much is specifically HRM's way of iterating?

## 2. The idea

`experiments/flow_sudoku.py`, class `SudokuFlowTransformer`. Three pieces.

**The network.** A plain bidirectional encoder — the same `arch.layers.Transformer` stack that
`arch/mae.py` uses — with a timestep added:

```
h = x_proj(x_t) + cond_embed(puzzle)      # 81 cells x 9 channels -> 81 x 768
h = h + t_mlp(sinusoid(t))                # one vector, broadcast to every cell
v = out_proj(block_16(... block_1(h)))    # back to 81 x 9: the predicted velocity
```

16 post-norm blocks, 768 wide, 12 heads, qk-norm, GELU MLPs of width 3072. Positions enter through
**axial 2D RoPE**: half of each head is rotated by the cell's row index and half by its column
index, so the model sees a 9 × 9 grid rather than a length-81 sequence. Nothing encodes the 3 × 3
boxes. `cond_embed` is one learned vector per digit with 0 for "blank", so a puzzle is just "some
cells are pinned" and an all-blank puzzle *is* the unconditional case — no null token needed.

The name says 113M; the model has 117.99M parameters. The 16 blocks are 113.2M (7.08M each); the
timestep MLP adds another 4.7M and the I/O projections are negligible.

**Training: regress the straight path.** Take a solved board, encode it as a one-hot image whitened
to zero mean and unit variance (`BoardCodec`), and call it `x_1`. Draw `x_0 ~ N(0, I)` and a time
`t ~ U(0, 1)`. The point on the straight segment between them is

```
x_t = (1 - t) * x_0 + t * x_1
```

and the model is asked to predict the segment's constant velocity: `loss = ‖v(x_t, t | puzzle) −
(x_1 − x_0)‖²`. That is the whole objective — a rectified-flow / conditional-OT regression, one
network evaluation per training sample, no cross-entropy anywhere. With probability 0.1 the puzzle
is blanked out so the same weights also learn the unconditional field; that is what makes
classifier-free guidance available at sampling time.

The data path is HRM's exact augmentation group — transpose, band/stack and within-band row/column
permutations, digit relabelling — vectorised on the GPU (`augment_grids`) so it never bottlenecks
the batch. The puzzle and its solution are augmented by the same group element so they stay a pair.

**Solving: integrate a stochastic ODE from noise to board.** The learned velocity is enough to
recover the score of the marginal at time `t`:

```
score(x, t) = (t * v(x, t) − x) / (1 − t)
```

so the deterministic probability-flow ODE `dx = v dt` can be traded for any SDE with the same
marginals, `dx = [v + ½ g² score] dt + g dW`. `sde_sample` uses `g(t) = σ (1 − t)`, which cancels the
`1/(1 − t)` and makes both the drift correction and the injected noise vanish as `t → 1`. On top of
that:

- **Classifier-free guidance**, `v_cfg = v(∅) + 2 (v(puzzle) − v(∅))`, which costs a second,
  blank-puzzle evaluation per drift.
- **Replacement inpainting** (`--clamp-givens`): at every step the given cells are re-noised to the
  current `t` and written back, so the model only ever has to fill in the blanks.
- **Heun** integration, 64 steps. Two drift evaluations per step, each with two network calls for
  guidance: **256 network evaluations per solve**.
- **Decode and verify.** `argmax` over the 9 channels gives a board; the 27 rows/columns/boxes are
  checked. A well-posed Sudoku has one valid completion, so a board that passes is correct — the
  answer key is never consulted. That is what makes "redraw until it verifies" a legitimate solver
  protocol rather than a leak.

## 3. The knobs

Every flag is in `flow_sudoku.py`'s `main()`. The values are the ones `run_flow_seeds.sh` fixes
for both runs.

| flag | value | meaning |
|---|---|---|
| `--num-layers / --hidden-size / --intermediate-size` | 16 / 768 / 3072 | the encoder |
| `--pos-embed` | `rope2d` | axial 2D RoPE. `rope1d` treats the board as a sequence (~11 points worse); `learned` is a free `[81, 768]` table (loses to RoPE from step 4000 on) |
| `--qk-norm` | on | RMS-normalise q/k per head. Without it three learning rates all collapsed mid-training from attention-logit growth |
| `--adaln` | off | DiT-style adaLN-Zero conditioning instead of the additive timestep. Implemented, not used here |
| `--repr` | `onehot` | 9 whitened channels per cell; `scalar` is 1 channel holding the digit |
| `--t-schedule` | `uniform` | `logit_normal` concentrates training on the middle of the path |
| `--conditional` | on | feed the puzzle. Off = unconditional board generation |
| `--cond-dropout` | 0.1 | blank the puzzle this often during training; enables guidance |
| `--guidance` | 2.0 | CFG scale at sampling; 1.0 is plain conditional |
| `--clamp-givens` | on | replacement inpainting of the given cells during sampling |
| `--sampler / --sample-steps` | `heun` / 64 | integrator and step count |
| `--noise-scale` | 10 | the `σ` in `g(t) = σ (1 − t)`. 0 is the deterministic ODE |
| `--eval-dataset-name` | `sudoku-extreme-1k` | scores the full-data run on the same `test_hard` split HRM uses |
| `--local-batch-size / --train-steps` | 96 × 8 GPUs / 83,200 | matched to `config/tuned_hrm.yaml`, see below |
| `--ema` | 0.999 | evaluation and checkpoints use the EMA weights |

Two of these deserve more than a table row.

**`--noise-scale` is not a sampling nicety.** The same weights give 55 % on `test_hard` with the
deterministic ODE and 72 % with `σ = 10` at 64 steps (`outputs/flow_trace/`); on the long-trained
checkpoint the sweep goes 72 → 90 % from `σ = 0` to 20 and then falls off a cliff at 25. Section 6
explains why: the noise is the mechanism by which the model revises cells it has already written.

**The step count and the noise are coupled.** The Wiener increment is `σ (1 − t) √dt`, so a sweep
over step counts at fixed `σ` is confounded — at 1 step the increment is `σ` itself and the state is
pure noise. Sweep steps at `σ = 0` to measure integration accuracy, and sweep `σ` at fixed steps to
measure the repair. `noise 10` is the calibrated value for 64 steps; the 89.9 % headline in
`outputs/flow_matching/` used 250 steps at `noise 20`.

## 4. The two runs, and why they are set up the way they are

`run_flow_seeds.sh` trains both arms at **exactly `tuned_hrm`'s optimisation budget**: a global
batch of 768 (96 × 8 GPUs), and `16 cycles × 260 batches × 20 epochs = 83,200` optimizer steps,
which is what HRM's `cycles_per_data: 16, epochs: 20` comes to on the 1k split with its 200-fold
augmentation. 4,160 steps is one HRM epoch, and the x-axis of the curve in the figure is labelled
that way. Three seeds each, `flow_113m_cfg_full` and `flow_113m_cfg_1k` interleaved so an early stop
still leaves matched pairs.

- **`flow_113m_cfg_1k`** trains on the same 1000 puzzles as HRM. This is the like-for-like
  comparison: same data, same augmentation, same number of steps at the same batch size.
- **`flow_113m_cfg_full`** trains on all 3.83M `sudoku-extreme` puzzles, 63.9M draws in total, so
  each puzzle is seen ~17 times in augmented form. It is scored on the *same* `test_hard` split as
  the 1k run (`--eval-dataset-name`). It asks what the architecture can do when data is not the
  constraint.

The per-step training compute is roughly matched too: one flow step is one pass through 16 blocks at
768 wide (113M MACs per cell); one HRM step is 28 block-forwards at 512 wide (88M). Inference is
not matched at all — 256 evaluations × 16 blocks against HRM's 16 recurrent steps × 28 blocks is
about 20× more compute per solve, and the flow model has 9.4× the parameters. Section 8 comes back
to that with a model of HRM's size.

Provenance, since the checkpoint directory is easy to misread:

- `checkpoints/flow_113m_cfg_{1k,full}/seed_{1,2,3}/` are the runs above (`logs/flowseeds/`).
- `checkpoints/flow_113m_cfg_{1k,full}/{best,last}.pt` at the top level are an earlier
  single-seed pair at 384 batch that was cut off at step 87,500 of 166,400 (`logs/final/`). The
  mechanism study in `outputs/flow_trace/` was run on that `full/best.pt`.
- The 89.94 % / 100 %-at-64-restarts numbers in `outputs/flow_matching/README.md` are from a
  different run, `cond_full_113m_cfg_long` — same architecture, 2048 batch, 150k steps, sampled with
  250 Heun steps at noise 20. They are the ceiling of the recipe, not the matched-budget result.

## 5. What happened

In-training evaluation, 512 `test_hard` puzzles, EMA weights, heun 64 / noise 10 / guidance 2,
single sample per puzzle. Mean ± sample std over 3 seeds.

| arm | data | best | at step (epoch) | final (step 83,200) |
|---|---|---|---|---|
| `flow_113m_cfg_1k` | 1k | **44.21 ± 2.18** | 17.5–20k (4–5) | 24.93 ± 1.41 |
| `flow_113m_cfg_full` | 3.83M | **74.54 ± 0.49** | 82.5–83.2k (20) | 74.22 ± 1.03 |
| `tuned_hrm` (reference, n=6) | 1k | 80.65 ± 2.40 | ep 11–19 | 79.4 |

Seeds: 1k 43.36 / 42.58 / 46.68, full 75.00 / 74.02 / 74.61.

The mean curves say more than the peaks. **The two arms are indistinguishable for the first 20,000
steps** — 41.5 % vs 43.9 % at step 20k, and within a point of each other at every evaluation before
that. Then they split: the 1k run falls to ~30 % within 10k steps and drifts down to 25 %, while the
full run keeps climbing and is still gaining a few tenths of a point per 2,500 steps at the end.

```
step    5k   10k   15k   20k   25k   30k   40k   50k   60k   70k   80k   83k
1k     6.1  20.4  30.9  41.5  31.8  30.3  28.3  26.2  25.7  24.8  24.9  24.9
full   6.4  20.1  28.1  43.9  56.1  59.8  66.1  69.1  71.3  72.9  73.6  74.2
```

That shape is the same generalisation failure the RT shows in `CYCLE_FF_EXPLAINED.md` §1 —
peak by epoch ~5, then overfit — and it is worth being precise about what is being overfitted. With
augmentation, each of the 1000 puzzles appears in ~10¹² forms, so this is not memorised strings.
`outputs/flow_matching/README.md` measured it directly on a 1k checkpoint of this architecture: edit
one given digit of a training puzzle (which keeps it uniquely solvable and leaves the augmentation
orbit) and the 1k model emits the *old* orbit's answer 48 % of the time on the cells the edit moved,
against 2.8 % for the full-data model and 9.8 % for HRM. The 1k model has learned to recognise which
orbit a puzzle belongs to; the full-data model has learned to solve.

Three things the table does **not** say, to head off the obvious misreadings:

- The full-data number is not the architecture's ceiling. At this budget it is still climbing, and
  the same recipe trained longer and sampled harder reaches 89.9 % single-shot and 100 % with
  verified restarts. What the matched budget shows is the *rate*: 74.5 % after 20 HRM-epochs' worth
  of steps, on 3,800× the data.
- The 1k number is not the architecture failing at Sudoku. The same 1k-trained weights generate a
  valid *unconditional* board 99.98 % of the time. 1000 boards plus HRM's augmentation is entirely
  enough to learn what a Sudoku looks like; it is not enough for this model to learn to solve one.
- Scale does not rescue the 1k regime. `outputs/flow_matching/` has the 27M version of the same
  model at 21.8 % on 1k; 118M moves that to 44–50 % depending on protocol. It is data efficiency,
  not capacity.

## 6. How it actually solves

`outputs/flow_trace/` records the model's endpoint estimate `x̂_1 = x_t + (1 − t) v` at every
integrator step for 1024 trajectories on `test_hard`. The short version:

1. **One evaluation is most of the answer.** A single forward pass on pure noise, one Euler step
   to `t = 1`, gets 83.4 % of cells right and solves 29.5 % of `test_hard` outright. Integrating the
   deterministic ODE more accurately adds nothing after ~4 evaluations; it saturates at 55 %.
2. **The rest is stochastic repair.** With `σ = 10` the same weights reach 72–74 % single-shot.
   Two thirds of successful trajectories get *worse* before they get better — a monotone
   fill-in-what-you-know process cannot do that. The model is tearing up a region of the board and
   re-laying it.
3. **The repair is aimed.** A cell inside a broken row/column/box is revised on the next step with
   probability 26.7 %; a cell whose three groups are all legal, 0.20 %. A factor of 130. Each
   revising step removes 1.87 broken groups, and the population count of broken groups falls
   monotonically to zero by step ~20.
4. **It is not constraint propagation.** 71 % of blank cells are written once and never touched;
   the order in which cells settle is uncorrelated with how many candidates they have (Spearman
   0.08); and naked/hidden singles only reach 15 % of these puzzles' blanks anyway.
5. **Restarts differ in route, not destination.** Two trajectories for the same puzzle agree on
   88 % of cells at step 1 and 100 % at the end whenever both verify. Failures land on a legal-looking
   board 35 cells away, never a near miss. That is why verified restarts converge to 100 % instead
   of plateauing, and why "valid ⇒ correct" is safe to lean on.

So the model is **amortised global guessing plus constraint-directed stochastic repair**, with
verification on the outside. Not search, not propagation, not autoregression.

## 7. The discrete variant: keep the board as tokens

`experiments/dfm_sudoku.py`, class `SudokuDiscreteFlowTransformer`, launched by `run_dfm_seeds.sh`.
Same 16-block backbone, same 2D RoPE, same puzzle conditioning and CFG, same augmentation, same
768 × 83,200 budget. What changes is the state space and the loss.

**Discrete flow matching** (Campbell et al. 2024; Gat et al. 2024). Each cell is a token in
{1..9}. The corruption is per cell: at time `t`, a cell is its true digit with probability `t` and
a draw from the prior otherwise. The model is a per-cell classifier `p(x_1 | x_t, t, puzzle)`,
trained with cross-entropy on the non-given cells. Sampling is Euler on the continuous-time Markov
chain: at each step draw `x̂_1 ~ p(x_1 | x_t)`, and every cell that disagrees with its draw jumps to
it with probability `dt / (1 − t)`; a detailed-balance term re-noises cells at rate `η` without
changing the marginals — the discrete analogue of the SDE's `g(t)`. The given cells are never
noised and never in the loss, so there is no clamping. One network call per step.

Two priors were tried. **`mask`** starts every cell at a MASK token and can only fill in; the loss
is on masked cells only. **`uniform`** starts at random digits, so the model cannot tell which
cells are wrong and has to predict `x_1` for all of them — the generator then keeps *correcting*
cells, which is the discrete version of the continuous model's guess-then-repair.

**Self-conditioning** (`--self-cond`, Chen et al. 2022 "Analog Bits"). A hard token sample throws
away exactly what the continuous `x_t` carries between steps: soft information about undecided
cells. Self-conditioning gives it back. The model also receives its own previous posterior over
every cell, through a zero-initialised projection `W_sc · p_prev`. At sampling time `p_prev` is
the softmax from the previous step; at training time it is a detached posterior from a no-grad
pass on the same `x_t` half the time and zeros the other half. `--self-cond-passes 2` runs two
no-grad refinement passes before the graded one.

### What happened on 1k

In-training evaluation, 512 `test_hard` puzzles, 64 CTMC steps, guidance 2, single seed unless marked. Every
run peaks by epoch 3–6 and overfits afterwards, so runs were stopped at 20–30k steps.

| arm | best in-training (η=0 / η=3) | at step |
|---|---|---|
| `dfm_113m_mask_cfg_1k` | 25.4 / 29.1 | 27.5k |
| `dfm_113m_unif_cfg_1k` | 24.0 / 24.4 | 10k |
| `dfm_113m_mask_sc_cfg_1k` (mask + self-cond) | 39.6 / 43.4 | 22.5k |
| `dfm_113m_unif_sc_cfg_1k` (uniform + self-cond) | 52.7 / 55.3 | 10k |
| `dfm_113m_unif_scwd1_cfg_1k` (+ weight decay 1.0) | 51.8 / 52.7 | 10k |
| `dfm_113m_unif_scp8_cfg_1k` (self-cond on 80 % of steps) | 58.2 / 61.3 (65.2 at η=10) | 12.5k |
| **`dfm_113m_unif_sc2_cfg_1k`** (self-cond, 2 refinement passes; n=3) | **66.6 ± 2.5 (74.5 ± 1.5 at η=10)** | 12.5k, every seed |
| `flow_113m_cfg_1k` (continuous, reference, n=3) | 44.21 ± 2.18 | 17.5–20k |

Three things fall out. Without self-conditioning, discrete is *worse* than continuous (24–25
against 44). With it, uniform beats mask everywhere, and every extra dose helps: one pass 53,
80 %-of-the-time 58, two passes 64. Weight decay does nothing. So the ingredient is not "discrete"
per se — it is discrete *plus* a soft channel between steps.

The tuned-sampler numbers are where the comparison to HRM is made (`eval_dfm_sudoku.py`, 512
`test_hard` puzzles, the `best.pt` at step 12.5k, three seeds):

| | continuous flow, 1k (n=3) | **discrete unif + sc×2, 1k** (n=3) | HRM, 1k (n=6) |
|---|---|---|---|
| in-training eval, 64 steps (512-puzzle subset) | 44.2 ± 2.2 | 66.6 ± 2.5 | — |
| tuned sampler, 1 sample, 512-puzzle subset | 50.4 ± 1.2 (heun 250, σ 20, guidance 3) | 86.5 ± 0.8 (128 steps, η 10, guidance 5) | — |
| **tuned sampler, 1 sample, full `test_hard` (20,000)** | — | **84.5 ± 1.2** (84.8 / 83.2 / 85.5) | **80.65 ± 2.40** (this is the split HRM is always scored on) |
| 8 verified restarts (512 subset) | 81.6 ± 1.3 | **97.7 ± 0.2** | — |
| 32 verified restarts (512 subset) | 91.3 ± 1.0 | **99.3 ± 0.3** | — |

Both flow rows are the `best.pt` of each seed scored on the same 512 `test_hard` puzzles. The
sampler cell was chosen on seed 1 (sweep: steps {64, 128} × η {10, 20, 30} × guidance {3, 4, 5})
and then held fixed for seeds 2 and 3, so the ± is a seed spread, not a selection artefact; the
per-seed values are 86.3 / 85.7 / 87.3. The continuous sweep was heun {64, 250} × σ {10, 20} ×
guidance {2, 3}, likewise fixed at its best cell across seeds. (An earlier draft of this table quoted
34.3 / 85.2 @64 for the continuous model, taken from `outputs/flow_matching/results.csv`; those
numbers belong to `cond_1k_113m_cfg`, a different, batch-2048 run whose own peak was 32.4 %, not to
the matched-budget seeds.)

**84.5 ± 1.2 % single-shot on the full `test_hard` split over three seeds, against HRM's
80.65 ± 2.40 on the same split**, with the same 1000 puzzles and HRM's own optimisation budget (it is
reached at 3 epochs of it). The seed ranges do not overlap (discrete 83.2–85.5, HRM's six seeds top
out at 82.3 in `CYCLE_FF.md`). The 512-puzzle subset on which `best.pt` and the sampler cell were
chosen reads 2 points higher (86.5 ± 0.8) — that is the selection bias, and the reason the full-split
number is the one to quote against HRM, whose in-training evaluation (`train.py`) has always been the
whole split. The sampler sweep
is flat near the top — everything at 128 steps with η ∈ {10, 20} and guidance ∈ {3, 4, 5} sits
within 1.5 points of the best cell on every seed — so this is not a lucky sampler cell either. Guidance matters more here than for the continuous
model (guidance 1 → 3 → 5 is roughly 55 → 70 → 80 at 64 steps), and η > 0 is worth 5–10 points,
the same repair story as §6.

### On full data

`dfm_113m_unif_sc2_cfg_full`, one seed, the full 83,200-step schedule (no early stop — on 3.8M
puzzles nothing overfits):

| step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 83.2k |
|---|---|---|---|---|---|---|---|---|---|
| in-training, η = 0 | 52.9 | 73.4 | 79.1 | 81.3 | 82.2 | 83.8 | 85.9 | 86.5 | **86.7** |
| in-training, η = 10 | 59.8 | 80.5 | 85.9 | 85.7 | 89.3 | 88.5 | 90.6 | 89.8 | **91.6** |
| continuous `flow_113m_cfg_full` (mean of 3) | 20.1 | 43.9 | 59.8 | 66.1 | 69.1 | 71.3 | 72.9 | 73.6 | 74.2 |

The discrete model passes the continuous model's *final* number (74.2) at step 20k and finishes
12–17 points above it at the same budget, still climbing. Its sampler has not been tuned yet; the
continuous full-data recipe gained ~15 points from tuning (74 → 89.9), so the comparison here is
in-training to in-training only.

Caveats that the figure states and that should be carried with the numbers:

- **n = 3 on 1k, n = 1 on full data.** The full-data seed-2 run is the only full-schedule one
  (seed 1 was early-stopped at 25k by mistake and reached 73.4 / 81.6 there, consistent with seed
  2's curve). Two 1k seeds had to be re-run after CUDA launch failures on GPUs that were also
  serving eval jobs; the reruns are clean.
- **On 1k it still overfits.** The 1k curve peaks at epoch 3 and halves by epoch 5. `best.pt` is an
  early-stopped checkpoint selected on 512 `test_hard` puzzles; the subset numbers are ~2 points
  above the full-split ones for that reason (`outputs/dfm_sizes/full_split/`).
- **Compute per solve** is 128 network calls × 2 for guidance = 256 evaluations × 16 blocks, the
  same as the continuous model and ≈20× HRM.

### How small can it be?

`outputs/dfm_sizes/README.md` sweeps the same recipe over backbone size on 1k (one GPU per run,
`run_dfm_sizes.sh`; single seed except where an ± is given). Two findings. **Depth is what matters**:
every 12–16-block model from 13M to 52M peaks at ~80 % in-training at LR 1e-4, while 4–8-block
models of any width stay at 35–64 % — `L4d512`, which has exactly HRM's 12.58M of transformer
weights, gets 35 %, and the same count arranged as `L16d256` gets 79.5 %. **Small models want a
higher LR**: at 3e-4 the 13M `L16d256` reaches 81.5 ± 0.5 % on the full `test_hard` split over three
seeds (84.1 ± 1.2 on the 512-puzzle selection subset); a 7.4M `L16d192` is at 80.8 ± 1.1 (n=3, full
split) and a 3.3M `L16d128` (single seed, subset) at 78.7 %; the recipe gives out below ~2.5M. §8
takes the 13M model as the like-for-like comparison with HRM.

The sweep also tried **soft givens** (`--givens soft`): the given cells are noised, predicted and
scored like every other cell and never written back, so the puzzle is a hint through `cond_embed`
rather than a clamp. At 113M it costs nothing (85.4 % on the 512 subset with the givens entirely
free, 86.3 % if clamped at sampling, vs 86.5 ± 0.8 hard on the same subset); at 13M it is 1–3 points
behind. It never beats hard
givens, so it is an ablation: the model can be trusted to rewrite the givens itself, but there is no
gain from letting it.

### Why it works

The continuous model on 1k learns *which orbit* a puzzle belongs to and emits that orbit's answer
(§5). The discrete model with self-conditioning does not, or does much less of it — and the
ablation says why. Neither the categorical state nor the cross-entropy on their own help (24 %).
What helps is being able to see a soft posterior over the whole board and correct it: uniform
prior over mask (so every cell is up for revision), self-conditioning (so the revision is informed
by the previous estimate rather than a hard resample), and more of both. That is a description of
an iterative solver with a readable working state, which is closer to what HRM's `z_L`/`z_H`
provide than to what a velocity field provides. The §9 candidate — that a classification readout
is the key — is half right: the cross-entropy is necessary but it needs the self-conditioned state
to pay off.

## 8. Same size as HRM: what the 13M discrete flow model costs and why it works

`dfm_L16d256_unif_sc2t_lr3_n3_cfg_1k`, three seeds, against `tuned_hrm`, six seeds. Same 1000
puzzles, same augmentation, same global batch (768), both reported at their best evaluation (HRM:
best epoch on the full `test_hard` split; DFM: best checkpoint and sampler cell chosen on a
512-puzzle subset, then scored on the full split — see the caveat at the end). MACs are per cell per
block-forward, `12·d²` (attention projections `4d²` + MLP `8d²`; the `81 × d` attention scores are
< 5 % and ignored); FLOPs are 2 × MACs. Wall clocks are one H100, `torch.compile`d, steady state
(a warm-up batch excluded), measured by `eval.py --batch-size` and `eval_dfm_sudoku.py --compile`
(`outputs/dfm_sizes/timing/`).

### Size, FLOPs, training time

| | **HRM** (`tuned_hrm`) | **DFM 13M** (`L16d256`, lr 3e-4) | DFM 113M (`L16d768`) |
|---|---|---|---|
| parameters | 12.59M | 13.12M (1.04×) | 117.99M (9.4×) |
| transformer weights | 4 blocks × 3.15M, 512 wide | 16 blocks × 0.79M, 256 wide | 16 blocks × 7.08M, 768 wide |
| **`test_hard` single-shot, full split (20,000)** | **80.65 ± 2.40** (n=6) | **81.5 ± 0.5** (n=3: 81.8 / 81.7 / 81.0) | **84.5 ± 1.2** (n=3: 84.8 / 83.2 / 85.5) |
| same, on the 512-puzzle selection subset | — | 84.1 ± 1.2 | 86.5 ± 0.8 |
| + 8 verified restarts (512 subset) | — (deterministic) | 98.0 | 97.7 |
| **inference**: block-forwards per solve | 16 segments × 28 = 448 | 128 steps × 2 (CFG) × 16 = 4,096 | 4,096 |
| inference MACs per cell / per board | 1.41 G / 114 G | 3.22 G / 261 G (**2.3×**) | 29.0 G / 2.35 T (20.6×) |
| **wall clock, whole `test_hard` (20,000 puzzles), one H100** | **15 s** (0.75 ms/puzzle; batch 512) | **52 s** (2.6 ms/puzzle; batch 2048; **3.5×**) — 175 s without `torch.compile`; 26 s at 64 steps but 72.7 % | 236 s (11.8 ms/puzzle; 16×) |
| **training**: MACs per cell per optimizer step | 264 M (28 fwd + BPTT bwd) | 50 M (1 graded fwd+bwd, ½ × 2 no-grad fwd) | 453 M |
| steps to best | 46–79k (epoch 11–19; mean ≈ 62k) | 15k (epoch 3.6) | 12.5k (epoch 3) |
| training FLOPs to best (batch 768 × 81 cells) | 2.1 × 10¹⁸ | 9.4 × 10¹⁶ (**22× less**) | 7.0 × 10¹⁷ (2.9× less) |
| wall clock to best | ~28 min on 8 H100 for the full 83k steps → ≈ 2.9 GPU-h | 10.4 min on **one** H100 (≈ 0.17 GPU-h, 17× less) | 9 min on 8 H100 (≈ 1.2 GPU-h) |
| learning rate | 1e-4, constant (`lr_min_ratio 1.0`) | 3e-4, cosine (stopped at 30k of 83k) | 1e-4, cosine |
| test-time knobs | none | steps, η, guidance | steps, η, guidance |

So at the same parameter count the discrete flow model **ties HRM** on the full split — 81.5 ± 0.5
against 80.65 ± 2.40, +0.9 with a far tighter seed spread, every DFM seed above HRM's mean but inside
HRM's range — gets there with 22× less training compute and in a seventeenth of the GPU-hours on one
card, and pays for it with 2.3× the inference MACs, which is **3.5× the wall clock** (52 s against
15 s for the 20,000-puzzle split; the narrow 256-wide blocks and the 256 sequential network calls
per solve run at lower arithmetic intensity than HRM's 448 calls at 512 wide). At 128 steps, that
is. The next subsection shows the step count is a dial: 63.5 % at HRM's wall clock, 95 % at 14× it,
98 % at 28×. The 113M model at 128 steps beats HRM, 84.5 ± 1.2, +3.9 with non-overlapping seed
ranges — for 9× the parameters and 16× the wall clock — but per second it is the worst of the three
sizes.

### Accuracy against inference compute: the step count is a test-time-compute dial

The 128-step sampler above was tuned on a {64, 128} grid and is far from saturated. Sweeping the
CTMC step count on the full split (`outputs/dfm_sizes/steps/`; three seeds of the 13M model, the
best η per step count, guidance 3; one H100, batch 2048, compiled), against HRM given more
recurrent segments at inference (`eval.py --cycles`; the one local HRM checkpoint, `precise-ant`
epoch 19, whose 16-cycle score is 73.6 — a weak seed, so read HRM's *slope* from it, not its level):

| wall clock, 20k puzzles | DFM 13M, n=3 | DFM 7.4M (seed 1) | DFM 113M (seed 1) | HRM `precise-ant` (cycles) | HRM n=6 reference |
|---|---|---|---|---|---|
| 6.5–7 s | 54.7 ± 2.4 (16 steps) | | | 69.8 (8) | |
| 13–15 s | **63.5 ± 1.4 (32 steps)** | | | 73.6 (16) | **80.65 ± 2.40 (16)** |
| 26–30 s | 72.2 ± 0.9 (64) | | | 76.5 (32) | |
| 53–59 s | 81.9 ± 0.2 (128) | 39 s: 81.8 (128) | | 78.6 (64) | |
| 105–118 s | 90.0 ± 0.5 (256, η 20) | 79 s: 90.4 (256) | | 80.4 (128) | |
| 211 s | 95.0 ± 0.8 (512, η 20) | 157 s: 95.8 (512) | | | |
| 421 s | **98.2 ± 0.6 (1024, η 40)** | | 470 s: 91.6 (256) | | |
| 946 s | | | 94.8 (512) | | |

Three readings of this table.

- **At HRM's wall clock the DFM loses.** 32 steps take the same 13 s as HRM's 16 segments and give
  63.5 %, seventeen points below HRM; the DFM needs ~4× HRM's wall clock (128 steps) to draw level.
  HRM is the better solver per second up to about 50 s per 20k puzzles. That is the answer to "how
  does the 13M model do at 32 steps".
- **Past that the DFM keeps buying accuracy and HRM barely does.** Every doubling of steps is worth
  +8–10 points for the DFM from 16 to 512 steps, then +3 to 1024: 90 % at 256 steps, 95 % at 512,
  98.2 % at 1024 — single sample, no verification, no restarts. HRM given more segments than it was
  trained with gains about +2 per doubling (69.8 → 80.4 from 8 to 128 cycles) and is still 15
  points short of the DFM at the same 118 s. The 7.4M model is *more* efficient than the 13M one per
  second (95.8 at 157 s), and the 113M model, at 9× the cost per step, is the worst deal above 50 s.
- **Why steps help this much.** `η` sets the *rate* of re-noising per unit time, so the expected
  amount of repair is the same at every step count; what the step count changes is how many cells
  jump *at once* on a stale posterior. Euler on a CTMC moves every disagreeing cell independently
  given the same `x_t`; at 32 steps that is dozens of simultaneous, mutually unaware jumps per step,
  at 1024 a handful, each seen by the next self-conditioned evaluation before the following one.
  Larger η also becomes usable as the steps grow (η 40 at 1024 steps beats η 20 by 0.8), because
  more re-noising is only useful if the model gets enough evaluations to repair it. So the curve is
  the §6 mechanism — guess, then constraint-directed repair — given more turns.

The consequence for the comparison: the HRM-sized flow model is not "as good as HRM"; it is a
different trade. HRM is a fixed-cost solver at 80 % that does not improve much with more compute.
The DFM at the same size is a dial from 55 % at half HRM's cost to 98 % at 28× it. The verified
restarts of §7 (97.7 % at 8 draws of 128 steps ≈ 8× cost) sit *below* this curve — 512 steps of one
trajectory (14× cost) reaches 95 % without any verifier at all.

### Inductive bias

Both are bidirectional transformers over 81 cells that iterate. Everything else about *how* they
iterate differs, and the ablations in §7 and `outputs/dfm_sizes/` say which differences carry the
result.

| | HRM | discrete flow model |
|---|---|---|
| what is iterated | two latents, `z_L` (every step) and `z_H` (every 6th) | the board itself, as 81 tokens |
| where the iteration lives | inside the network: 28 block-forwards per segment, 16 segments, one carry | in the sampler: 128 CTMC steps; the network is a plain 16-block feed-forward encoder |
| weight reuse | 4 blocks, each applied 7× per segment (2 L-blocks × 12 + 2 H-blocks × 2 = 28) | 16 distinct blocks, each applied once per evaluation |
| timescales | two (H/L) | one, the corruption level `t`; no slow/fast split |
| what the intermediate state looks like in training | whatever the network makes it (latents; no gradient across the carry, BPTT within a segment) | an explicit noisy board *sampled from the forward process* — the model is teacher-forced on its own working state at every corruption level |
| supervision | solution cross-entropy at the end of every segment, from the same puzzle → solution pair | solution cross-entropy on every non-given cell at a random `t`; each draw is a different partial board |
| revision | latents can change; the readout is recomputed each segment | any cell can jump to a new digit (uniform prior); η re-noises cells; a failed board is redrawn (restarts) |
| stochasticity | none | prior, jumps, η, restarts |
| carry-over between steps | full-width latents (`z_L`, `z_H`) | the hard tokens *plus* the previous posterior (self-conditioning) |
| puzzle | as the input, injected into every step (`z_L + z_H + x`) | as an additive embedding, plus the givens pinned in the state; dropped 10 % for CFG |
| Sudoku-specific structure | none (1D RoPE; the 2D variant `tuned_hrm_rope2d` exists) | none beyond axial 2D RoPE (row/col; no 3×3-box prior); `--givens soft` removes even the pinning at no cost at 113M. **With HRM's 1D RoPE the 13M model drops to 77.5 ± 3.6 on the full split (75.0 / 81.6 / 75.8)** — it learns ~2× slower and one seed in three still reaches the 2D number, so the geometry is learnable but the prior buys speed and stability |
| output check | none | a valid board is a correct one, so it can verify and restart |

### Why it works — what the evidence supports

Two things need explaining: how a plain encoder with no recurrence learns Sudoku from 1000 puzzles
as well as HRM does at the same size (and better at 9× the size), and why it does so in a twentieth
of the compute.

1. **It is trained on its own intermediate states.** HRM has to invent a trajectory through latent
   space that ends at the solution, supervised only by the readout at the end of each segment, with
   no gradient across segments. The flow model never has to learn what a good intermediate state is:
   the forward process *hands it one* — a board with a random fraction `t` of the cells right and
   the rest random — and asks for the solution from there. One puzzle becomes a whole family of
   "finish this partially-right board" problems, one per corruption draw, each with a per-cell
   target. This is the same trick as teacher forcing, applied to the working state instead of the
   output sequence, and it is where the 22× compute-to-peak comes from: the supervision is denser
   and every step of it is on-distribution for what the sampler will see. The continuous model gets
   this too and still fails on 1k (§5), so it is necessary, not sufficient.
2. **The state is categorical and revised under cross-entropy, with a soft carry-over.** That is
   the §7 ablation: uniform prior + self-conditioning is the whole difference between 24 % and
   86 % at 113M. A hard token board with no self-conditioning cannot remember that a cell was
   uncertain; a continuous board can, but its velocity regression rewards being close on average
   rather than exactly right. The DFM has both a hard readout and a soft memory, which is the same
   pair HRM has (`lm_head` over `z_H`, `z_L`/`z_H` carried). HRM's version is learned end to end
   through 28 blocks of BPTT; the DFM's is one zero-initialised linear map over its own posterior.
3. **The compute per evaluation goes into depth, not width.** HRM spends its 12.6M parameters on
   four 512-wide blocks and gets depth by reusing them 28 times per segment. The size sweep says
   that arrangement is a bad denoiser: `L4d512` as a DFM gets 35 %, `L8d512` 64 %, while the same
   parameter count as 16 distinct 256-wide blocks gets 81.5 % on the full split. Whether HRM would also gain from
   untied depth is a different experiment (the RT in `CYCLE_FF_EXPLAINED.md` reuses blocks too,
   and lands at 70.5); what the sweep establishes is that once the per-step function has ~16
   distinct blocks, width from 256 to 768 is worth only ~2–5 points.
4. **The failure mode is one that restarts fix.** The sampler is stochastic and a Sudoku certifies
   itself, so the ~18 % of single-shot failures — which land on legal-looking boards far from the
   answer, not near misses (§6) — are re-rolled: 84 → 98 % at 8 draws on the 512 subset, 99.8 % at
   32 on the 113M model. HRM is deterministic; its 19 % of failures are final. This is not a training advantage,
   but it is a property of the model class, and it is why "valid ⇒ correct" makes the comparison to
   HRM's single pass the *conservative* one for the flow model.
5. **It converges in 3–4 epochs and would rather stop there.** The same fact that makes it
   data-efficient (dense per-cell supervision on 1000 puzzles × 10¹² augmentations × corruption
   draws) makes it overfit by epoch 5–7 — every 1k DFM run halves from its peak by 30k steps. HRM
   peaks at epoch 11–19 and degrades gently. So the DFM needs early stopping and HRM does not,
   which is a real operational cost; it is also why the DFM's peak is reached in a tenth of HRM's
   steps rather than being a faster learner that then keeps going.

What it is **not**: it is not a smarter solver. §6's mechanism — amortised guess, constraint-directed
stochastic repair, no propagation, no search — is the 113M continuous model's, but nothing in the
discrete model's design adds search, and its guidance/η dependence (guidance 1 → 5 is worth 25
points at 64 steps) is the same "repair needs a push" signature. The advantage is that its training
problem is easier to learn from 1000 examples, not that it reasons more.

### Caveats specific to this comparison

- `best.pt` and the sampler cell (η ∈ {10, 20} × guidance ∈ {3, 5}, 128 steps) are selected on
  512 `test_hard` puzzles; HRM's "best epoch" is selected on the full split. Both are test-selected,
  but the 512-subset DFM numbers carry a measurable selection bias: scoring the same checkpoints and
  cells on all 20,000 puzzles gives 81.5 vs 84.1 at 13M, 84.5 vs 86.5 at 113M, 80.8 vs 82.5 at 7.4M
  — about 2–2.6 points. Every headline in this document is now the full-split number; the 512
  numbers are kept only where the full split was not run (restarts, the single-seed sweep).
- The in-training peaks of the five 13M runs ranged 81–87 % and identical seeds do not reproduce
  (`torch.compile`, GPU multinomial), so the ± 1.2 offline is the number to carry, not any single
  seed.
- The 2D RoPE is the one piece of Sudoku geometry the DFM is told and HRM is not. Removing it
  (`--pos-embed rope1d`, three seeds) costs 4 points on average and most of the seed stability:
  77.5 ± 3.6 on the full split (75.0 / 81.6 / 75.8), i.e. 3 points *below* HRM's 80.65 ± 2.40. So
  the tie at 13M is "with the row/column prior"; on HRM's own positional encoding the DFM is slightly
  behind at this size. (`tuned_hrm_rope2d` exists in `config/` for the converse experiment but has
  no seeds logged in this repo.)
- LR was tuned for the DFM ({1e-4, 3e-4, 1e-3}) and not re-tuned for HRM; `tuned_hrm` is the
  repo's tuned configuration, but at 13M the DFM's result depends on the LR (79.5 at 1e-4 vs 84.1
  at 3e-4 on the subset), so the comparison is best-known recipe against best-known recipe, not
  matched tuning effort.

## 9. So what does this say about HRM?

The figure's last two rows lay out the differences; the ones that matter for the argument are these.

**HRM's advantage over the continuous model on 1k is data efficiency, and it is large.** 80.65
against 44.21 at the same budget, on the same 1000 puzzles, with a 9.4× smaller model. The
continuous flow model needs about three-and-a-half orders of magnitude more puzzles to get within
6 points, and even then it is 20× more expensive per solve.

**But the advantage is not over flow models as a class.** The discrete variant with
self-conditioning (§7) reaches 84.5 ± 1.2 % on the full `test_hard` split from the same 1000
puzzles at the same budget, three seeds — above HRM's 80.65 ± 2.40, non-overlapping ranges — with
no recurrence, no latent, no Sudoku-specific structure beyond a row/column positional encoding, and
3 epochs of training. At 113M it is 9.4× larger and 16× slower per puzzle; shrunk to HRM's own 13M
(§8) it ties HRM (81.5 ± 0.5) at 128 sampler steps, reaches its peak with 22× less training
compute, and costs 3.5× HRM's wall clock per puzzle at that setting. That leaves HRM its
inference-efficiency claim — it is the fastest 80 % solver here by a factor of ~4, and at HRM's own
wall clock the DFM is at 63.5 % — and makes the parameter- and *data*-efficiency claims a tie
against this comparison class rather than a win. What HRM does not have is the dial: the same 13M
flow model reaches 95 % at 14× HRM's cost and 98 % at 28×, single sample, where HRM given more
segments stalls in the low 80s.

**Iteration at inference is not, by itself, what makes a model data-efficient.** The continuous
flow model iterates 64 times and revises its own answer, and on 1k it still memorises the orbit.
Nor is "a slow state" — `CYCLE_FF_EXPLAINED.md` found that grafting one onto an RT recovers about
four of HRM's eleven points over the RT and none from scratch. What the discrete ablation isolates
is narrower and more useful: a categorical state that is revised under cross-entropy, *with a soft
carry-over between revisions*. Remove the carry-over (no self-conditioning) and the discrete model
falls to 24 %; add more of it and it climbs monotonically.

**The cross-entropy readout was the candidate, and it is half the answer.** HRM and the RT are
trained with a per-cell classification loss at every recurrent step; the continuous flow model is
trained with an MSE on a velocity at a random `t`. A velocity regression rewards being close on
average; a classification loss rewards being exactly right. The discrete variant tests exactly
this swap, and on its own it does nothing (24 %). Paired with a self-conditioned soft state it is
worth 40 points. The two together look a lot like one HRM cycle: a categorical readout, trained
hard, over a state that remembers the last readout.

**On full data the picture inverts.** The full-data flow model out-solves HRM on the one-digit-edit
probe (93.6 % solve / 2.8 % copy against HRM's 83.7 % / 9.8 %, `outputs/flow_matching/README.md`),
reaches 100 % on `test_hard` with restarts, and has no Sudoku-specific structure to point at.
Nothing in HRM's design is *necessary* for Sudoku; it is what makes Sudoku learnable from 1000
examples.

## 10. Reproducing

```
./run_flow_seeds.sh                    # continuous: all six runs, sequential, in tmux
./run_flow_seeds.sh full:1 1k:1        # one matched pair
grep -o "\[step [0-9]*\] test_hard_exact_match=[0-9.]*" logs/flowseeds/flow_113m_cfg_full_seed1.log

STOP_STEP=25000 ./run_dfm_seeds.sh unif:1k:1:sc2      # discrete: best config, early-stopped
uv run python experiments/eval_dfm_sudoku.py --ckpt checkpoints/dfm_113m_unif_sc2_cfg_1k/seed_1/best.pt \
    --sample-steps 64 128 --noise-scale 10 20 --guidance 3 4 5 --restarts 32
```

Discrete run names are `dfm_113m_<prior>[_<tag>]_cfg_<data>`; the tags (`sc`, `sc2`, `scp8`,
`wd1`, `scwd1`, `elbo`) map to flags in `run_dfm_seeds.sh`'s `variant_flags`. Logs are in
`logs/dfmseeds/`; the sampler sweeps quoted in §7 are `eval_dfm_sudoku.py` output.

The size sweep and the 13M comparison in §8 (`run_dfm_sizes.sh`: one single-GPU run per spec, batch
768, the tuned sampler for the in-training eval):

```
./run_dfm_sizes.sh                                                   # the exploratory sweep, 8 GPUs
NAME_SUFFIX=_n3 STOP_STEP=30000 ./run_dfm_sizes.sh L16d256:lr3:1 L16d256:lr3:2 L16d256:lr3:3   # the 13M, 3 seeds
uv run python experiments/eval_dfm_sudoku.py --ckpt checkpoints/dfm_L16d256_unif_sc2t_lr3_n3_cfg_1k/seed_1/best.pt \
    --sample-steps 128 --noise-scale 10 20 --guidance 3 5 --restarts 8
uv run python experiments/collect_dfm_sizes.py logs/dfmsizes* logs/dfmlr* logs/dfmtiny* logs/dfmn3   # in-training table

# full split + wall clock (one H100), and the step-count dial
uv run python experiments/eval_dfm_sudoku.py --ckpt checkpoints/dfm_L16d256_unif_sc2t_lr3_n3_cfg_1k/seed_1/best.pt \
    --num-puzzles 20000 --batch 2048 --compile --sample-steps 32 128 512 1024 --noise-scale 3 10 20 40 --guidance 3
uv run python eval.py --ckpt "checkpoints/tuned_hrm precise-ant/seed_1/epoch_19.pt" --split test_hard --batch-size 512 --cycles 16
```

The figure is generated by `experiments/make_flow_figure.py` (writes the SVG); the PNG is the SVG
rendered at 2880 px wide with cairosvg (the same pipeline as `cycle_ff_architectures.png`). The continuous
curves in the results panel are 3-seed means of the in-training evaluations, read from
`logs/flowseeds/`; the discrete curves are seed 1 from `logs/dfmseeds/` (1k, ablations) and `logs/dfmfull/` (full data); 1k seeds 2–3 are in `logs/dfm1k/`.
The size panel plots each shape's best in-training exact match at its best LR (`outputs/dfm_sizes/in_training_table.txt`),
with the 3-seed offline numbers annotated.

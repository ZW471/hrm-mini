# What the flow-matching model is, and what it says about HRM

A companion to `outputs/flow_matching/README.md` (the results log) and `outputs/flow_trace/README.md`
(the mechanism study). This one explains the idea, the code that implements it, and what the two
HRM-matched runs — `flow_113m_cfg_1k` and `flow_113m_cfg_full` — concluded. The visual abstract is
`flow_matching_architecture.{svg,png}`.

---

## 1. The question this is asking

Every other model in this repo maps a puzzle to a solution with a *latent* state: HRM carries
`z_L`/`z_H`, the recurrent transformer carries `z`, and both read the answer off that state with an
`lm_head` trained by cross-entropy. The flow model removes the latent altogether. **The board is
the state.** A solved Sudoku is a point in a continuous 81 × 9 space, the model is a velocity field
over that space, and solving a puzzle means integrating from Gaussian noise to a board while the
given cells are held fixed.

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
about 20× more compute per solve, and the flow model has 9.4× the parameters. Section 7 comes back
to that.

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
  model at 21.8 % on 1k; 118M moves that to 34–44 % depending on protocol. It is data efficiency,
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

## 7. So what does this say about HRM?

The figure's bottom row lays out the differences; the ones that matter for the argument are these.

**HRM's advantage on 1k is data efficiency, and it is large.** 80.65 against 44.21 at the same
budget, on the same 1000 puzzles, with a 9.4× smaller model. The flow model needs about
three-and-a-half orders of magnitude more puzzles to get within 6 points, and even then it is 20×
more expensive per solve. Whatever HRM's latent recurrence is buying, it is buying it *here*.

**Iteration at inference is not, by itself, what makes HRM data-efficient.** The flow model
iterates 64 times and revises its own answer, and on 1k it still memorises the orbit. Nor is
"a slow state" — `CYCLE_FF_EXPLAINED.md` found that grafting one onto an RT recovers about four of
HRM's eleven points over the RT and none from scratch. The property that separates HRM from both is
still unnamed.

**The cross-entropy readout is one candidate.** HRM and the RT are trained with a per-cell
classification loss at every recurrent step; the flow model is trained with an MSE on a velocity at
a random `t`, and its train-time accuracy is a one-step endpoint estimate averaged over noise
levels — `flow_sudoku.py` warns that the shared wandb key `train/exact_match` does not mean the same
thing. A velocity regression rewards being close on average; a classification loss rewards being
exactly right. On 1000 puzzles that may be the difference between learning a lookup and learning a
rule. This is untested.

**On full data the picture inverts.** The full-data flow model out-solves HRM on the one-digit-edit
probe (93.6 % solve / 2.8 % copy against HRM's 83.7 % / 9.8 %, `outputs/flow_matching/README.md`),
reaches 100 % on `test_hard` with restarts, and has no Sudoku-specific structure to point at.
Nothing in HRM's design is *necessary* for Sudoku; it is what makes Sudoku learnable from 1000
examples.

## 8. Reproducing

```
./run_flow_seeds.sh                    # all six runs, sequential, in tmux
./run_flow_seeds.sh full:1 1k:1        # one matched pair
grep -o "\[step [0-9]*\] test_hard_exact_match=[0-9.]*" logs/flowseeds/flow_113m_cfg_full_seed1.log
```

The figure is regenerated from `flow_matching_architecture.svg`; the PNG is the SVG rendered at
2880 px wide with cairosvg (the same pipeline as `cycle_ff_architectures.png`). The curve in the
bottom panel is the 3-seed mean of the in-training evaluations above, read from `logs/flowseeds/`.

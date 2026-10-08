# Handover: Sudoku data-scaling study (HRM vs RT) -- protocol v2, 16 single-GPU runs in flight

*Rewritten 2026-09-22 07:10 UTC by the local Claude Code session that took over at 04:10 UTC. This file
is the complete context; nothing else needs to be read first. Everything below the "v1" section is
history kept for reference; the live work is v2.*

## 0. Where things run

Your local Claude Code reaches **two GPU servers by plain `ssh <name>`**, both 8 x H100 80GB, both
`root`, both with the project at the **same shared path** `/sg-pretrain/zhiyu/hrm-rebuttal/hrm-mini`
(`/sg-pretrain` is a shared filesystem: logs, checkpoints, queue files and this file are visible on both
hosts within a second). The hosts do **not** share a process space (`tmux ls`, `nvidia-smi`, `ps` are per
host). The local directory `/Users/dricpro/PycharmProjects/hrm-mini` holds only this file.

| ssh name | hostname | v2 runs | tmux |
|---|---|---|---|
| `Sapient-sg-2` | `di-20250904173933-mrc4z` | the 8 **HRM** runs, GPU i = size i | `v2` (windows `hrm_<size>_v2`) |
| `Sapient-sg` | `di-20260902143015-kw9qc` | the 8 **RT** runs, GPU i = size i | `v2` (windows `rt_<size>_v2`) |

Work over ssh: `ssh <host> "cd /sg-pretrain/zhiyu/hrm-rebuttal/hrm-mini && <cmd>"`. Python is
`.venv/bin/python` / `.venv/bin/torchrun` (uv-managed; never `pip install`). W&B creds are in
`/root/.netrc` (entity `zhiyuwang-university-of-cambridge`, project `sudoku`). Git: `master`, last
commit `e6a158d` (the user's); uncommitted: `train.py` (+ `train.py.bak_v1`), `run_scaling_v2.sh`,
`experiments/{collect_v2,collect_converged,collect_hp_sweep}.py`, `outputs/`, `logs/`, this file,
`note.md`, `submission/`. Commit only if the user asks.

Two ssh gotchas: (1) `ssh host "... pkill -f '<pattern>' ..."` kills the ssh shell itself (its command
line contains the pattern) -> exit 255 / no output; the target dies too. Verify in a second call with a
different substring. Patterns containing `+` match nothing (regex). (2) The jump host sometimes drops
a connection ("Connection closed by 45.78.201.249") when several ssh calls come in quick succession;
just retry.

## 1. The study and the user's rules (v2, in the user's words where it matters)

Question: accuracy on Sudoku-Extreme (exact match on the 20k `test_hard` split of the 1k repo, 16
inference cycles) vs number of unique training puzzles, for HRM (2 H-layers + 2 L-layers, hidden 512,
H_cycles 2 / L_cycles 6, 12.59M params) and RT (recurrent transformer, 4 layers x 7 cycles, same 12.59M
params, same 28 block applications per forward), at 1 / 10 / 100 / 1k / 10k / 100k / 1M / full (3,831,994).

The user's rules:
1. **Never** change the model or the inference cycle count. Only data size and trainer hyperparameters.
2. (2026-09-22 06:40 UTC) The staged protocol "is somehow overfitting the validation set since we are
   adjusting hyperparams according to validation results and sft again and again". So: **fixed lr, wd
   and steps per size, chosen a priori from current understanding; one single consistent run per
   (model, size), no stages, no warm restarts; cosine annealing everywhere; early stop when there is
   no validation improvement within ~50k steps; one GPU per experiment; HRM and RT at the same scale
   use the same configuration.** "Be aware that EMA is also applied to HRM -- this might be
   problematic; if so fix any problems" (checked, see section 3; kept on for both).
3. Expectation: **monotonically increasing curves from 1 to full for both models, with HRM better.**
4. Always report `best`, never last. One seed per point is fine. Tag every run `scaling test`.
5. Do not make schedule decisions by looking at the eval curve (that is what v2 fixes). The only
   eval-driven mechanism left is the early stop, which is part of the pre-registered protocol.
6. Never upload copied / replayed results to W&B (the colleague's runs live only as files under
   `checkpoints/*_copied/`).

## 2. Protocol v2 (what is running now)

Launched 2026-09-22 06:54 UTC by `run_scaling_v2.sh` (`ARCH=hrm` on `Sapient-sg-2`, `ARCH=rt` on
`Sapient-sg`). Per run: `torchrun --standalone --nproc-per-node 1 train.py --config-name
tuned_<arch>_full data=<cfg> epochs=<E> local_batch_size=768 lr=1e-4 weight_decay=<wd> lr_min_ratio=0.01
eval_interval=4160 +max_steps=<H> +early_stop_patience_steps=49920 seeds=[1] run_name=<arch>_<size>_v2`,
`CUDA_VISIBLE_DEVICES=<gpu>`, `MLP_TASK_NAME=<run>` (-> `checkpoints/<run>/seed_1/`),
`WANDB_TAGS='scaling test,scaling_v2'`, log `logs/v2/<run>.log`, exit code appended to
`logs/v2/done.txt`, launch record `logs/v2/launched.txt`. Full description: `logs/v2/README.md`.

* Global batch 768 on one GPU (v1 used 96 x 8 GPUs; same batch, ~8.9 it/s per run, 33 GB), AdamATan2
  (0.9, 0.95), lr 1e-4, 2k warm-up, cosine to 0.01 x lr over the horizon `max_steps` (new option in
  `train.py`; the run stops there after a final eval), EMA 0.999, eval every 4,160 steps, `best.pt` =
  best eval (EMA weights), `best_raw.pt` = raw weights at the same step, early stop after 49,920 steps
  (12 evals) without a >= 0.1 pp gain over the best.
* Per size: wd 1.0 at 1 / 10, 0.3 at 100 and 1k, 0.1 at >= 10k (the v1 sweep's per-size winners; 1k by
  interpolation -- the repo's own 1k config used wd 1). Horizons `max_steps`: 104,000 (1 / 10 / 100),
  166,400 (1k), 332,800 (10k), 416,000 (100k), 499,200 (1M), 582,400 (full). ETAs from the 06:54 start:
  1-100 by ~10:10 (earlier if early-stopped), 1k ~12:00, 10k ~17:15, 100k ~20:00, 1M ~22:40, full
  ~01:00 UTC 09-23.
* Data configs: `config/data/sudoku_{1,10,100}.yaml` (200k samples/epoch of the tiny nested subsets),
  `sudoku.yaml` (1k, repeat 200), `sudoku_{10k,100k,1m}.yaml`, `sudoku_full.yaml`. 4,160 steps/epoch
  for <= 100k, 20,833 for 1M, 79,824 for full; `epochs` in the launcher is just an upper bound.

**Monitoring** (from the laptop; the table is the thing to look at):
```
ssh Sapient-sg-2 "cd /sg-pretrain/zhiyu/hrm-rebuttal/hrm-mini && .venv/bin/python experiments/collect_v2.py"   # both hosts' runs, from the logs
ssh Sapient-sg-2 "cd /sg-pretrain/zhiyu/hrm-rebuttal/hrm-mini && tr '\r' '\n' < logs/v2/hrm_full_v2.log | grep -oE '\[(eval|early-stop)\].*' | tail -5"
ssh Sapient-sg   "cd /sg-pretrain/zhiyu/hrm-rebuttal/hrm-mini && cat logs/v2/done.txt; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader"
```
`collect_v2.py` states: `running` / `early-stopped` / `finished` / `crashed` / `stalled?` (log not
written for 10 min). A crashed run is re-run from scratch with its exact line from `logs/v2/launched.txt`
(there is no resume in v2 by design): `tmux kill-window -t v2:<run>` on its host, delete
`checkpoints/<run>`, and start it again in a new tmux window with the same env and overrides (the
launcher's loop body). The W&B run of a crashed attempt should be tagged `crashed` (`experiments/
tag_wandb_runs.py --regex '^<run>$' --tag crashed`) so the re-run is the one that counts.

**Do not** kill a v2 run because it looks flat -- the early stop handles that. Do not add stages.

**Extra seeds on idle GPUs** (user, 09:00 UTC: "use idle GPUs to run different seeds; the W&B run name for
different seeds should stay the same so they group"). `run_v2_seeds.sh` runs as tmux window `v2:seeds` on
each host (`ARCH=hrm` / `ARCH=rt`): every 60 s it finds GPUs with no compute process (and nothing launched
on them in the last 5 min), pops the next `<size> <seed>` line from `logs/v2/queue_<arch>.txt` and launches
`<arch>_<size>_v2` with `seeds=[<seed>]` -- same settings and same `MLP_TASK_NAME` (= W&B name + group; the
runs differ by `config.seed`), checkpoints `checkpoints/<run>/seed_<seed>/`, log `logs/v2/<run>_s<seed>.log`,
tmux window `<run>_s<seed>`, launch records in `logs/v2/launched.txt` and `logs/v2/seeds_<arch>_launched.txt`,
worker log `logs/v2/seeds_<arch>.log`. Queue order as installed: tiny sizes seed 2 (1, 10, 100), then
full / 1M / 100k / 10k / 1k seed 2, then the same for seed 3; edit the queue file any time (`STOP` line
ends the worker; restart with `ARCH=<arch> ./run_v2_seeds.sh`). The per-size table in `run_v2_seeds.sh`
must stay identical to the one in `run_scaling_v2.sh`. `collect_v2.py` lists every seed and a per-point
mean over the seeds that have stopped; the curve should use the mean over seeds (state the n per point).
Seed 1 of each point is the one launched 06:54; seed-2 tiny runs started 09:01 on the GPUs freed by the
early stops (hrm_1 2.37, hrm_10 10.60, rt_1 2.09, rt_10 9.41, rt_100 45.11 -- all stopped at 58-62k steps
with bests at 8-12k).

## 2b. v2 results as of 2026-09-23 03:10 UTC (seed 1 complete everywhere)

Mean over the seeds that have finished; individual seeds in brackets. Full table:
`.venv/bin/python experiments/collect_v2.py`; write-up and figure: `outputs/scaling_v2/`
(`README.md`, `scaling_v2.svg/.png`, `make_figure_v2.py`, `results_v2.csv`).

| N | HRM | RT |
|---|---|---|
| 1 | 2.37 (2.37, 2.50, 2.23) | 2.43 (2.09, 2.51, 2.69) |
| 10 | **9.82** (10.60, 8.97, 9.89) | 9.05 (9.41, 8.07, 9.68) |
| 100 | 38.33 (30.02, 42.50, 42.46) | **41.82** (45.11, 40.34, 40.00) |
| 1k | **81.88** (82.67, 80.53, 82.44) | 71.56 (70.90, 71.50, 72.27) |
| 10k | **96.50** (96.65, 96.34) | 95.79 (95.20, 96.38) |
| 100k | 97.59 (97.22, 97.97) | **97.80** (97.94, 97.66) |
| 1M | 97.00 (1 seed) | **98.07** (97.86, 98.27) |
| full | 94.66 (97.29, 92.03) | **98.12** (1 seed) |

HRM wins at 1k (+10.3) and 10k (+0.7); RT wins from 100k up. Seed noise dominates at N = 100
(HRM 30.0-42.5) and at the full set (HRM 92.0-97.3), so the >= 100k rows are not settled.
Still running: seed 3 of 10k / 100k / 1M / full for both architectures (launched 03:04-03:05, the
large ones finish ~13:00-21:00 UTC); queued next: seed 4 of 100, full, 1k, 10k (`logs/v2/queue_<arch>.txt`).

**The early stop, not the cosine horizon, ended most runs** (1k at ~92k of 166k; HRM full seed 2 at
~129k of 582k, scoring 92.03 against seed 1's 97.29 over the full 582k). If the user wants a v2.1,
the a-priori fix is a longer patience at the large sizes, or a shorter horizon so the decay completes
-- applied identically to both architectures, decided before the runs, never from the eval curves.

## 3. The EMA check (done 2026-09-22, `logs/v2/emacheck_*.log`)

`adam_atan2.py` keeps `param_ema.lerp_(param, 1 - 0.999)` per step for both models and swaps it in for
every eval/checkpoint. Scoring EMA vs raw weights at the EMA-selected best step of v1 runs
(`eval.py --ckpt ... --split test_hard --batch-size 500`): constant lr 1e-4 -> HRM 100 puzzles 46.98
vs 3.96, RT 42.76 vs 15.45, HRM 100k 91.34 vs 85.85; end of an anneal -> RT 100k 97.70 vs 97.58. So
EMA is what generalises at high lr (for both, more for HRM) and coincides with the raw weights once the
lr is low, which a cosine-to-0.01x run reaches by its end. Not a bug; kept for both. If the user wants
the raw numbers, `best_raw.pt` exists for every v2 run.

## 4. When the runs finish

1. `experiments/collect_v2.py` -> `outputs/scaling_v2/results_v2.csv` (best %, step, state per run).
2. Figure: write `outputs/scaling_v2/make_figure_v2.py` (start from `outputs/scaling_tuned/make_figure.py`,
   which already handles log-x sizes, provisional markers and step-count labels): one panel, best
   test_hard vs unique puzzles, HRM and RT, one seed, label each point with its step count and whether
   it early-stopped; a second panel with the per-run eval curves is useful. Keep "best of run, one
   seed, 16 cycles" wording.
3. Write-up `outputs/scaling_v2/README.md`: protocol (section 2 above), table, the EMA check, and an
   honest comparison with v1 (v1 numbers are optimistic by construction). Also report raw-weight
   scores if the user asks (`eval.py` on `best_raw.pt`, on a **free** GPU only -- never on a GPU a run
   is using; that crashed a v1 run once with `CUDA error: unspecified launch failure`).
4. If a point disappoints, the v2 rule is: do not tune on the eval curve. Any change (a different wd,
   horizon or lr for a size) must be argued a priori, applied to **both** models, and run as a fresh
   single run; say so in the write-up.

## 5. v1 (superseded, kept for reference; all its W&B runs are tagged `scaling_v1`)

v1 = (A) a fixed 83,200-step lr/wd sweep per size (`run_hp_sweep.sh`, `logs/hpsweep/`, results in
`outputs/scaling_tuned/results.csv` / `README.md` / `EXPLANATION.md`), (B) train-to-convergence chains
of warm restarts (constant lr -> plateau-triggered anneal -> chained slow decays at 1e-5, 5e-6,
2.5e-6) whose cut points were chosen by watching `test_hard` (`logs/hpsweep/` for HRM, `logs/rtconv/`
for RT; `experiments/collect_converged.py` -> `outputs/scaling_tuned/converged.csv`), and (C) the
1 / 10 / 100-puzzle downscale at the fixed budget (`logs/downscale/`). Figure of A + B:
`outputs/scaling_tuned/scaling_tuned.{png,svg}` (`make_figure.py`, refreshed 06:31 UTC).

Final v1 numbers (best %, one seed): fixed budget HRM 3.48 / 8.94 / 45.66 / 80.65 / 90.59 / 92.07 /
92.51 / 92.95 and RT 2.59 / 10.45 / 43.00 / 69.78 / 88.29 / 90.86 / 91.45 / 90.45 at 1 / 10 / 100 /
1k / 10k / 100k / 1M / full; converged (10k..full) HRM 96.41 / 97.21 / 98.13 / 98.20 and RT 96.23 /
98.47 / 98.31 / 97.85 (HRM 1M and RT full stage 3s were killed mid-run at 06:44 when v2 started).
What v1 taught, and what v2's settings rest on: wd 1 is far too strong beyond ~100 puzzles (0.1 wins
at >= 10k, 0.3 at 100); lr 2e-4 diverges; gains come from lr decay, not from sitting at constant lr;
the tiny sizes peak at 8-21k steps and then overfit (train exact match -> 100 %), and without
augmentation they solve 0 test puzzles; at convergence the two architectures land within ~0.3 pp of
each other except RT ahead at 100k, after RT's constant-lr phases ran 3-7x longer than HRM's -- and
HRM's own constant-lr phase at 100k is genuinely flat at ~91 (checked twice on 09-22), at 1M it
plateaus at ~94.3 within 200k steps while RT creeps to 94.65 over 400k.

v1 files that still matter: `checkpoints/<v1 run>/seed_1/{best,last,best_raw,last_raw}.pt`,
`logs/hpsweep/wandb_ids.json`, `experiments/tag_wandb_runs.py` (tagging by name regex),
`experiments/collect_hp_sweep.py` (W&B table, slow), `dataset/build_sudoku_{subsets,tiny_subsets}.py`,
`config/data/*.yaml`, `eval.py` (`--ckpt <pt> --split test_hard --batch-size 500`; never pass `--cycles`).

"""Copy the full-Sudoku-Extreme HRM and RT runs from a colleague's scaling study into this repo's
checkpoint layout, explicitly labelled as copied (local files only, nothing goes to W&B).

Source: /sg-pretrain/hex/hrm-mini/cl_agi_hrm_mini (scaling handover of 2026-09-20). Their
full-set lr/wd search picked lr 2.5e-5 / wd 0.03 for HRM (95.81 % on the 20k test_hard, epoch 18 of
20) and the same setting transferred to RT (97.365 %, epoch 15, early-stopped after 17). Both are
seed 1, 8 x H100, local batch 96, 16 cycles per batch, EMA 0.999, constant lr after 2k warm-up --
the same recipe as `config/tuned_{hrm,rt}_full.yaml` apart from lr, weight decay and, crucially,
the budget: 20 epochs over the full set is up to 1,596,480 optimizer steps, 20x the ~83k of the
fixed-budget arms in run_dataset_sizes.sh / run_hp_sweep.sh. They are NOT budget-matched points.

Their model code carries muP hooks (`mup_base_width: 512`) that are identities at hidden 512
(width multiplier 1, same embedding scale and init), and the state-dict keys are this repo's
`HRM` / `RecurrentTransformer` keys, so the weights load unchanged; `epoch_N.pt` are EMA weights
(N zero-based). Verify with
    .venv/bin/python eval.py --ckpt checkpoints/<run>/seed_1/best.pt --split test_hard

What this writes, per run:
    checkpoints/<run>/seed_1/{best.pt,last.pt,model_config.json,metrics.jsonl,train_metrics.jsonl,
                              source_config.yaml,source_arch/,COPIED_FROM.md}
Copied runs are deliberately NOT logged to W&B: the `sudoku` project holds only runs trained in
this repo. Their curves live in `metrics.jsonl` / `train_metrics.jsonl` next to the weights.

    .venv/bin/python experiments/copy_hex_full_runs.py            # both
    .venv/bin/python experiments/copy_hex_full_runs.py --only hrm
"""
import argparse
import hashlib
import json
import os
import shutil

import yaml

HEX = "/sg-pretrain/hex/hrm-mini/cl_agi_hrm_mini/experiment"
RUNS = {
    "hrm": dict(
        run="hrm_full_lr2.5e-5_wd0.03_copied",
        base_config="config/tuned_hrm_full.yaml",
        exp=f"{HEX}/hrm_full_lr_wd_20260917",
        ckpt_dir=f"{HEX}/hrm_full_lr_wd_20260917/source/checkpoints/lr25e6_wd003/seed_1",
        config=f"{HEX}/hrm_full_lr_wd_20260917/source/config/lr25e6_wd003.yaml",
        best_epoch=17, last_epoch=19, result_kind="complete_20_epochs",
    ),
    "rt": dict(
        run="rt_full_lr2.5e-5_wd0.03_copied",
        base_config="config/tuned_rt_full.yaml",
        exp=f"{HEX}/rt_full_hrm_best_20260919",
        ckpt_dir=f"{HEX}/rt_full_hrm_best_20260919/source/checkpoints/rt_transfer_seed1/seed_1",
        config=f"{HEX}/rt_full_hrm_best_20260919/source/config/rt_transfer.yaml",
        best_epoch=14, last_epoch=16, result_kind="early_stopped_after_17_epochs",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_run(arch, spec):
    src_cfg = yaml.safe_load(open(spec["config"]))
    metrics = [json.loads(l) for l in open(os.path.join(spec["ckpt_dir"], "metrics.jsonl"))]
    best = max(metrics, key=lambda m: m["exact_match"])
    assert best["epoch"] == spec["best_epoch"], (best, spec)
    last = metrics[-1]
    assert last["epoch"] == spec["last_epoch"], (last, spec)

    out = os.path.join("checkpoints", spec["run"], "seed_1")
    os.makedirs(out, exist_ok=True)
    src_best = os.path.join(spec["ckpt_dir"], f"epoch_{spec['best_epoch']}.pt")
    src_last = os.path.join(spec["ckpt_dir"], f"epoch_{spec['last_epoch']}.pt")
    for src, dst in [(src_best, "best.pt"), (src_last, "last.pt"),
                     (os.path.join(spec["ckpt_dir"], "metrics.jsonl"), "metrics.jsonl"),
                     (os.path.join(spec["ckpt_dir"], "train_metrics.jsonl"), "train_metrics.jsonl"),
                     (spec["config"], "source_config.yaml")]:
        shutil.copy2(src, os.path.join(out, dst))
    arch_dst = os.path.join(out, "source_arch")
    shutil.rmtree(arch_dst, ignore_errors=True)
    shutil.copytree(os.path.join(spec["exp"], "source", "arch"), arch_dst, ignore=shutil.ignore_patterns("__pycache__"))

    # model_config.json in this repo's TrainConfig layout so eval.py / examine_failure_mode.py load it
    # like any local run: their arch block minus the muP field, their lr / wd / epochs, and the eval
    # pointed at the 1k repo's test_hard exactly as config/tuned_*_full.yaml does.
    base = yaml.safe_load(open(spec["base_config"]))
    base.pop("defaults", None)
    cfg = base | {
        "run_name": spec["run"],
        "arch": {k: v for k, v in src_cfg["arch"].items() if k != "mup_base_width"},
        "data": {"name": "sudoku", "dataset_name": "./downloaded-datasets/sudoku-extreme", "repeat": 1, "augment": True},
        "seeds": [1], "epochs": src_cfg["epochs"], "lr": src_cfg["lr"], "weight_decay": src_cfg["weight_decay"],
        "eval_interval": None,
        "copied_from": spec["ckpt_dir"],
    }
    assert cfg["arch"] == base["arch"], "architecture differs from this repo's full-set config"
    for k in ("cycles_per_data", "local_batch_size", "lr_warmup_steps", "lr_min_ratio", "beta1", "beta2", "ema"):
        assert cfg[k] == src_cfg[k], (k, cfg[k], src_cfg[k])
    yaml.dump(cfg, open(os.path.join(out, "model_config.json"), "w"))

    note = f"""# COPIED RUN -- not trained in this repo

`{spec['run']}` is a copy of a colleague's run from the scaling study at
`{spec['exp']}` (handover of 2026-09-20). Nothing here was trained or evaluated by this repo's
`train.py`; the numbers below are theirs, read from `metrics.jsonl`.

| | |
|---|---|
| source checkpoint dir | `{spec['ckpt_dir']}` |
| best.pt | `epoch_{spec['best_epoch']}.pt` (zero-based; epoch {spec['best_epoch'] + 1}), sha256 `{sha256(src_best)}` |
| last.pt | `epoch_{spec['last_epoch']}.pt` (epoch {spec['last_epoch'] + 1}), sha256 `{sha256(src_last)}` |
| best test_hard exact match | {best['correct']}/{best['total']} = {100 * best['exact_match']:.3f} % at step {best['step']} |
| last test_hard exact match | {last['correct']}/{last['total']} = {100 * last['exact_match']:.3f} % at step {last['step']} |
| result | {spec['result_kind']} |
| lr / weight decay | {src_cfg['lr']} / {src_cfg['weight_decay']} (their full-set search winner, seed 1) |
| budget | up to 20 epochs x 79,824 steps = 1,596,480 optimizer steps on the full 3,831,994-puzzle set; **not** the ~83k-step budget of the fixed-budget arms |
| training data | `{src_cfg['data']['dataset_name']}` (their normalised copy of the full Sudoku-Extreme train split, '.' -> '0'; same puzzles as `downloaded-datasets/sudoku-extreme/train.csv`) |
| eval data | the same 20k `test_hard` split as this repo (sha256 5f9549fb...); best selected on it (exploratory, no held-out set) |

`source_config.yaml` is their config verbatim, `source_arch/` their model code (muP hooks that are
identities at `mup_base_width: 512`); `model_config.json` is this repo's layout for `eval.py`.
Their per-epoch metrics are in `metrics.jsonl` / `train_metrics.jsonl`. Their early-stopping rule
(from epoch 4: stop if the last 3 epochs improved best by < 0.1 pp, or 2 consecutive epochs sit
>= 0.5 pp under best) applied to RT; HRM ran all 20 epochs.
"""
    open(os.path.join(out, "COPIED_FROM.md"), "w").write(note)
    print(f"{spec['run']}: best {100 * best['exact_match']:.3f} % (epoch {best['epoch'] + 1}, step {best['step']}), "
          f"last {100 * last['exact_match']:.3f} % -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(RUNS))
    args = p.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    for arch, spec in RUNS.items():
        if args.only and arch != args.only:
            continue
        copy_run(arch, spec)

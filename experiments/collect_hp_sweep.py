"""Tabulate the lr / weight-decay sweep (run_hp_sweep.sh) from W&B: best and final test_hard exact
match per run, grouped by architecture and dataset size.

    .venv/bin/python experiments/collect_hp_sweep.py             # every run tagged hp_sweep
    .venv/bin/python experiments/collect_hp_sweep.py --size 10k --arch hrm
    .venv/bin/python experiments/collect_hp_sweep.py --csv outputs/hp_sweep/results.csv
    .venv/bin/python experiments/collect_hp_sweep.py --size 1,10,100 --csv results.csv --merge   # only re-scan these

"best" is the maximum of `eval/test_hard_exact_match` over the run's eval points (every 4160 steps),
i.e. the checkpoint saved as `best.pt`; "final" is the last eval. Both are on the 20k `test_hard`
split the 1k runs are scored on. Runs still training are included and marked.
"""
import argparse
import csv
import os
import re

import wandb

p = argparse.ArgumentParser()
p.add_argument("--path", default="zhiyuwang-university-of-cambridge/sudoku")
p.add_argument("--tag", default="hp_sweep")
p.add_argument("--size", help="only these dataset sizes (comma-separated), e.g. 1 / 10 / 100 / 10k / 100k / 1m / full")
p.add_argument("--arch", help="only this architecture, e.g. hrm / rt")
p.add_argument("--csv", help="also write the table to this CSV")
p.add_argument("--merge", action="store_true", help="update rows of the same run name in an existing --csv, keep the others "
               "(so a slow W&B scan can be limited to --size / --arch without dropping older rows)")
args = p.parse_args()

api = wandb.Api(timeout=60)
runs = api.runs(args.path, filters={"tags": {"$in": [args.tag]}}, per_page=500)

rows = []
for r in runs:
    m = re.match(r"^(?P<arch>[a-z]+)_(?P<size>[0-9]+[km]?|full)_lr(?P<lr>[0-9.e-]+)_wd(?P<wd>[0-9.]+)(?P<rest>.*)$", r.name)
    if not m:
        continue
    if args.size and m["size"] not in args.size.split(",") or args.arch and m["arch"] != args.arch:
        continue
    hist = [h for h in r.scan_history(keys=["eval/test_hard_exact_match", "_step"]) if h.get("eval/test_hard_exact_match") is not None]
    if not hist:
        continue
    best = max(hist, key=lambda h: h["eval/test_hard_exact_match"])
    rows.append({
        "arch": m["arch"], "size": m["size"], "lr": float(m["lr"]), "wd": float(m["wd"]), "variant": m["rest"].lstrip("_"),
        "run": r.name, "seed": int(r.config.get("seed", 0) or 0), "state": r.state, "evals": len(hist),
        "best": 100 * best["eval/test_hard_exact_match"], "best_step": best["_step"],
        "final": 100 * hist[-1]["eval/test_hard_exact_match"], "final_step": hist[-1]["_step"],
    })

size_order = {"1": -4, "10": -3, "100": -2, "1k": -1, "10k": 0, "100k": 1, "1m": 2, "full": 3}
rows.sort(key=lambda x: (x["arch"], size_order.get(x["size"], 9), -x["lr"], -x["wd"], x["variant"], x["seed"]))

print(f"{'run':36s} {'state':9s} {'lr':>8s} {'wd':>5s} {'best %':>7s} {'@step':>7s} {'final %':>8s} {'evals':>5s}")
last = None
for x in rows:
    key = (x["arch"], x["size"])
    if key != last:
        print(f"--- {x['arch']} {x['size']} ---")
        last = key
    flag = "" if x["state"] == "finished" else f" ({x['state']})"
    name = x["run"] if x["seed"] in (0, 1) else f"{x['run']} (seed {x['seed']})"
    print(f"{name:36s} {x['state']:9s} {x['lr']:8.1e} {x['wd']:5g} {x['best']:7.2f} {x['best_step']:7d} {x['final']:8.2f} {x['evals']:5d}{flag}")

for key in sorted({(x["arch"], x["size"]) for x in rows}, key=lambda k: (k[0], size_order.get(k[1], 9))):
    done = [x for x in rows if (x["arch"], x["size"]) == key and x["state"] == "finished" and x["variant"] not in ("noaug", "drawb")]
    if done:
        w = max(done, key=lambda x: x["best"])
        print(f"winner {key[0]} {key[1]}: {w['run']}  best {w['best']:.2f} % at step {w['best_step']}")

if args.csv:
    out = rows
    if args.merge and os.path.exists(args.csv):
        new = {x["run"] for x in rows}
        old = [x for x in csv.DictReader(open(args.csv)) if x["run"] not in new]
        for x in old:
            x.setdefault("seed", 1)   # CSVs written before the seed column existed hold seed-1 runs
            for k in ("lr", "wd", "best", "final"): x[k] = float(x[k])
            for k in ("evals", "best_step", "final_step", "seed"): x[k] = int(x[k])
        out = sorted(old + rows, key=lambda x: (x["arch"], size_order.get(x["size"], 9), -x["lr"], -x["wd"], x["variant"], x["seed"]))
    with open(args.csv, "w", newline="") as f:
        fields = ["arch", "size", "lr", "wd", "variant", "run", "seed", "state", "evals", "best", "best_step", "final", "final_step"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out)
    print("wrote", args.csv, f"({len(out)} rows, {len(rows)} from this scan)")

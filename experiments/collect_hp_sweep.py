"""Tabulate the lr / weight-decay sweep (run_hp_sweep.sh) from W&B: best and final test_hard exact
match per run, grouped by architecture and dataset size.

    .venv/bin/python experiments/collect_hp_sweep.py             # every run tagged hp_sweep
    .venv/bin/python experiments/collect_hp_sweep.py --size 10k --arch hrm
    .venv/bin/python experiments/collect_hp_sweep.py --csv outputs/hp_sweep/results.csv

"best" is the maximum of `eval/test_hard_exact_match` over the run's eval points (every 4160 steps),
i.e. the checkpoint saved as `best.pt`; "final" is the last eval. Both are on the 20k `test_hard`
split the 1k runs are scored on. Runs still training are included and marked.
"""
import argparse
import csv
import re

import wandb

p = argparse.ArgumentParser()
p.add_argument("--path", default="zhiyuwang-university-of-cambridge/sudoku")
p.add_argument("--tag", default="hp_sweep")
p.add_argument("--size", help="only this dataset size, e.g. 1 / 10 / 100 / 10k / 100k / 1m / full")
p.add_argument("--arch", help="only this architecture, e.g. hrm / rt")
p.add_argument("--csv", help="also write the table to this CSV")
args = p.parse_args()

api = wandb.Api(timeout=60)
runs = api.runs(args.path, filters={"tags": {"$in": [args.tag]}}, per_page=500)

rows = []
for r in runs:
    m = re.match(r"^(?P<arch>[a-z]+)_(?P<size>[0-9]+[km]?|full)_lr(?P<lr>[0-9.e-]+)_wd(?P<wd>[0-9.]+)(?P<rest>.*)$", r.name)
    if not m:
        continue
    if args.size and m["size"] != args.size or args.arch and m["arch"] != args.arch:
        continue
    hist = [h for h in r.scan_history(keys=["eval/test_hard_exact_match", "_step"]) if h.get("eval/test_hard_exact_match") is not None]
    if not hist:
        continue
    best = max(hist, key=lambda h: h["eval/test_hard_exact_match"])
    rows.append({
        "arch": m["arch"], "size": m["size"], "lr": float(m["lr"]), "wd": float(m["wd"]), "variant": m["rest"].lstrip("_"),
        "run": r.name, "state": r.state, "evals": len(hist),
        "best": 100 * best["eval/test_hard_exact_match"], "best_step": best["_step"],
        "final": 100 * hist[-1]["eval/test_hard_exact_match"], "final_step": hist[-1]["_step"],
    })

size_order = {"1": -4, "10": -3, "100": -2, "1k": -1, "10k": 0, "100k": 1, "1m": 2, "full": 3}
rows.sort(key=lambda x: (x["arch"], size_order.get(x["size"], 9), -x["lr"], -x["wd"], x["variant"]))

print(f"{'run':36s} {'state':9s} {'lr':>8s} {'wd':>5s} {'best %':>7s} {'@step':>7s} {'final %':>8s} {'evals':>5s}")
last = None
for x in rows:
    key = (x["arch"], x["size"])
    if key != last:
        print(f"--- {x['arch']} {x['size']} ---")
        last = key
    flag = "" if x["state"] == "finished" else f" ({x['state']})"
    print(f"{x['run']:36s} {x['state']:9s} {x['lr']:8.1e} {x['wd']:5g} {x['best']:7.2f} {x['best_step']:7d} {x['final']:8.2f} {x['evals']:5d}{flag}")

for key in sorted({(x["arch"], x["size"]) for x in rows}, key=lambda k: (k[0], size_order.get(k[1], 9))):
    done = [x for x in rows if (x["arch"], x["size"]) == key and x["state"] == "finished"]
    if done:
        w = max(done, key=lambda x: x["best"])
        print(f"winner {key[0]} {key[1]}: {w['run']}  best {w['best']:.2f} % at step {w['best_step']}")

if args.csv:
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", args.csv)

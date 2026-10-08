"""Tabulate the train-to-convergence arms from the run logs (no W&B): per architecture and dataset
size, the best `test_hard` exact match over every stage of the convergence recipe (constant lr ->
anneal -> slow decay; run names `<arch>_<size>_lr*_wd*_conv*` / `_long*` in logs/hpsweep (HRM, other
host) and logs/rtconv (RT, this host)), the step it was reached at, and whether the stage holding
any stage of the chain is still running (the point is then provisional). Writes
outputs/scaling_tuned/converged.csv for make_figure.py.

    .venv/bin/python experiments/collect_converged.py
"""
import csv
import glob
import os
import re
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "outputs", "scaling_tuned", "converged.csv")
size_order = {"10k": 0, "100k": 1, "1m": 2, "full": 3}

points = {}
for log in sorted(set(glob.glob(os.path.join(ROOT, "logs", "*", "*_conv*.log"))) | set(glob.glob(os.path.join(ROOT, "logs", "*", "*_long*.log")))):
    run = os.path.basename(log)[:-4]
    m = re.match(r"^(hrm|rt)_(10k|100k|1m|full)_lr", run)
    if not m:
        continue
    text = open(log, errors="replace").read().replace("\r", "\n")
    evals = [(int(s), float(x)) for s, x in re.findall(r"\[eval\] step (\d+) test_hard exact_match ([0-9.]+)", text)]
    if not evals:
        continue
    # a log still being written to (< 5 min old) whose run has no stop line is in progress
    running = time.time() - os.path.getmtime(log) < 300 and "stopping" not in text
    key = (m.group(1), m.group(2))
    p = points.setdefault(key, {"arch": key[0], "size": key[1], "best": -1, "best_step": 0, "best_run": "", "last": 0, "last_step": 0, "stages": [], "running": False})
    p["stages"].append(run)
    # a point is provisional while any stage of its chain is still running: the best may still move
    p["running"] = p["running"] or running
    s, x = max(evals, key=lambda e: e[1])
    if x > p["best"]:
        p.update(best=x, best_step=s, best_run=run)
    if evals[-1][0] > p["last_step"]:
        p.update(last=evals[-1][1], last_step=evals[-1][0])

rows = sorted(points.values(), key=lambda p: (p["arch"], size_order[p["size"]]))
print(f"{'arch':4s} {'size':5s} {'best %':>7s} {'@step':>8s} {'last %':>7s} {'@step':>8s}  status / stages")
for p in rows:
    print(f"{p['arch']:4s} {p['size']:5s} {100*p['best']:7.2f} {p['best_step']:8d} {100*p['last']:7.2f} {p['last_step']:8d}  "
          f"{'PROVISIONAL (a stage is running)' if p['running'] else 'done'}: {', '.join(p['stages'])}")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arch", "size", "best", "best_step", "best_run", "last", "last_step", "running", "stages"])
    w.writeheader()
    for p in rows:
        w.writerow({**{k: p[k] for k in ["arch", "size", "best_step", "best_run", "last_step"]}, "best": 100 * p["best"], "last": 100 * p["last"],
                    "running": int(p["running"]), "stages": " ".join(p["stages"])})
print("wrote", OUT)

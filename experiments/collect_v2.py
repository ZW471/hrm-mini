"""Status / results table for the v2 data-scaling runs (one single-GPU run per architecture and size,
`run_scaling_v2.sh`; extra seeds from `run_v2_seeds.sh`), read from logs/v2/<arch>_<size>_v2[_s<seed>].log
on the shared filesystem -- no W&B needed. Per run: best test_hard exact match (%), the step it was reached
at, the latest eval, the step the run has reached, its state (running / early-stopped / finished /
crashed), and the horizon; then per (arch, size) the mean best over the seeds that have stopped. Writes
outputs/scaling_v2/results_v2.csv (one row per run, `seed` column).

    .venv/bin/python experiments/collect_v2.py
"""
import csv
import glob
import os
import re
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "scaling_v2")
SIZES = ["1", "10", "100", "1k", "10k", "100k", "1m", "full"]
N = {"1": 1, "10": 10, "100": 100, "1k": 1000, "10k": 10000, "100k": 100000, "1m": 1000000, "full": 3831994}

rows = []
for log in sorted(glob.glob(os.path.join(ROOT, "logs", "v2", "*_v2*.log"))):
    m = re.match(r"^(hrm|rt)_(1|10|100|1k|10k|100k|1m|full)_v2(?:_s(\d+))?$", os.path.basename(log)[:-4])
    if not m:
        continue
    seed = int(m.group(3) or 1)
    text = open(log, errors="replace").read().replace("\r", "\n")
    evals = [(int(s), float(x)) for s, x in re.findall(r"\[eval\] step (\d+) test_hard exact_match ([0-9.]+)", text)]
    bars = re.findall(r"seed=\d+:.*?\| (\d+)/(\d+) \[", text)
    step, horizon = (int(bars[-1][0]), int(bars[-1][1])) if bars else (0, 0)
    if "[early-stop]" in text:
        state = "early-stopped"
    elif "Traceback" in text or "CUDA error" in text:
        state = "crashed"
    elif horizon and step >= horizon:
        state = "finished"
    elif time.time() - os.path.getmtime(log) < 600:
        state = "running"
    else:
        state = "stalled?"
    rate = re.findall(r"([0-9.]+)it/s\]", text)
    best_step, best = max(evals, key=lambda e: e[1]) if evals else (0, float("nan"))
    last_step, last = evals[-1] if evals else (0, float("nan"))
    rows.append({"arch": m.group(1), "size": m.group(2), "seed": seed, "n_puzzles": N[m.group(2)], "best": round(100 * best, 3),
                 "best_step": best_step, "last": round(100 * last, 3), "last_step": last_step, "step": step,
                 "horizon": horizon, "state": state, "it_per_s": rate[-1] if rate else "", "n_evals": len(evals),
                 "wandb": (re.findall(r"runs/([a-z0-9]+)", text) or [""])[0]})

rows.sort(key=lambda r: (r["arch"], SIZES.index(r["size"]), r["seed"]))
print(f"{'run':16s} {'best %':>7s} {'@step':>7s} {'last %':>7s} {'@step':>7s} {'progress':>17s}  state          it/s  wandb")
for r in rows:
    print(f"{r['arch'] + '_' + r['size'] + ' s' + str(r['seed']):16s} {r['best']:7.2f} {r['best_step']:7d} {r['last']:7.2f} {r['last_step']:7d} "
          f"{r['step']:8d}/{r['horizon']:<8d}  {r['state']:14s} {r['it_per_s']:>5s} {r['wandb']}")
print()
print(f"{'point':10s} {'mean best %':>11s}  seeds done (best per seed)      running")
for arch in ("hrm", "rt"):
    for size in SIZES:
        pts = [r for r in rows if r["arch"] == arch and r["size"] == size]
        if not pts:
            continue
        done = [r for r in pts if r["state"] in ("early-stopped", "finished")]
        run = [r for r in pts if r["state"] == "running"]
        mean = sum(r["best"] for r in done) / len(done) if done else float("nan")
        print(f"{arch + '_' + size:10s} {mean:11.2f}  {len(done)} ({', '.join(f'{r['best']:.2f}' for r in done) or '-'}){'':>{max(0, 22 - 6 * len(done))}}"
              f"{len(run)} ({', '.join(f'{r['best']:.2f}@{r['step'] // 1000}k' for r in run) or '-'})")
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "results_v2.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["arch"])
    w.writeheader()
    w.writerows(rows)
print("wrote", out)

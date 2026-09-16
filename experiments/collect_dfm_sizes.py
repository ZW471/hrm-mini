"""Summarise the run_dfm_sizes.sh sweep: params, best in-training exact match (tuned sampler, eta 10
and eta 0), step of the peak, and the last evaluated value.

    .venv/bin/python experiments/collect_dfm_sizes.py [logs/dfmsizes ...]
"""
import glob
import os
import re
import sys

EVAL = re.compile(r"\[step (\d+)\] test_hard_exact_match=([\d.]+).*?test_hard_exact_match_eta0=([\d.]+)")

def summarise(log: str):
    text = open(log, errors="replace").read()
    m = re.search(r"([\d.]+)M params", text)
    params = float(m.group(1)) if m else float("nan")
    evals = [(int(s), float(a), float(b)) for s, a, b in EVAL.findall(text)]
    return params, evals

logdirs = sys.argv[1:] or ["logs/dfmsizes"]
print(f"{'run':44s} {'params':>8s} {'best@eta10':>10s} {'step':>6s} {'best@eta0':>9s} {'step':>6s} "
      f"{'last':>6s} {'@step':>6s}  curve at eta10 (per 2.5k steps)")
for logdir in logdirs:
    for log in sorted(glob.glob(os.path.join(logdir, "dfm_*_seed*.log"))):
        name = re.sub(r"\.log$", "", os.path.basename(log))
        params, evals = summarise(log)
        if not evals:
            print(f"{name:44s} {params:7.2f}M  (no eval yet)")
            continue
        best = max(evals, key=lambda e: e[1])
        best0 = max(evals, key=lambda e: e[2])
        last = evals[-1]
        print(f"{name:44s} {params:7.2f}M {100 * best[1]:9.1f}% {best[0]:6d} {100 * best0[2]:8.1f}% {best0[0]:6d} "
              f"{100 * last[1]:5.1f}% {last[0]:6d}  " + " ".join(f"{100 * e[1]:.0f}" for e in evals))

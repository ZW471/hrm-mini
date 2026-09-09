"""Pull the eval curves of the graft arms and their baselines off W&B into one table.

Every arm here is scored on the same 20k `test_hard` puzzles, so `best` and `last` are directly
comparable across rows. `init` is the accuracy of the frozen checkpoint an arm starts from: the
graft is zero-gated, so grafted arms provably begin exactly there and `best - init` is the honest
measure of what training bought.
"""
import argparse
import wandb

PROJECT = "zhiyuwang-university-of-cambridge/sudoku"
# Accuracy of the pretrained checkpoint each grafted arm starts from (measured with eval.py).
INIT = {
    "graft_rt_frozen": 0.7087, "graft_rt_sft": 0.7087, "graft_rt_inline": 0.7087,
    "rt_sft": 0.7087, "graft_rt_frozen_zoom": 0.7087, "graft_rt_full_frozen": 0.8759,
}
ORDER = ["tuned_hrm", "tuned_rt", "graft_rt_frozen", "graft_rt_inline", "graft_rt_sft", "rt_sft",
         "graft_rt_frozen_zoom", "tuned_rt_full", "graft_rt_full_frozen"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-evals", type=int, default=5, help="skip smoke tests")
    args = ap.parse_args()

    rows = {}
    for r in wandb.Api().runs(PROJECT, order="-created_at"):
        name = r.name.rsplit(" ", 1)[0]
        if name not in ORDER:
            continue
        hist = r.history(keys=["eval/test_hard_exact_match"], pandas=False)
        vals = [x["eval/test_hard_exact_match"] for x in hist]
        if len(vals) < args.min_evals:
            continue
        rows.setdefault(name, []).append(vals)

    print(f"{'arm':22s} {'seed':>4s} {'init':>7s} {'best':>7s} {'@':>5s} {'last':>7s} {'best-init':>10s}")
    for name in ORDER:
        for i, vals in enumerate(rows.get(name, [])):
            init = INIT.get(name)
            best, at = max(vals), max(range(len(vals)), key=lambda j: vals[j]) + 1
            delta = f"{best - init:+.4f}" if init else "-"
            print(f"{name:22s} {i+1:>4d} {init if init else float('nan'):>7.4f} {best:>7.4f} "
                  f"{at:>5d} {vals[-1]:>7.4f} {delta:>10s}")
    return rows

if __name__ == "__main__":
    main()

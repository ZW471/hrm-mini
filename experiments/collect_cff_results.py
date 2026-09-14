"""Tabulate the cycle-FF experiment arms from W&B, newest run of each name last.

Every arm is scored on the same 20,000 `test_hard` puzzles, so `best` and `last` compare directly
across rows. `best` is the number that matters: on Sudoku-Extreme 1k every arm overfits after its
peak, so `last` mostly measures how hard it fell, not how well it learned.

    uv run python experiments/collect_cff_results.py            # all arms
    uv run python experiments/collect_cff_results.py --arms cff_sft rt_sft
"""
import argparse
import statistics

PROJECT = "zhiyuwang-university-of-cambridge/sudoku"
# Accuracy of the pretrained checkpoint the fine-tuned arms resume from, measured with eval.py.
# They are zero-gated at init, so they provably start exactly here. NOTE: this is a single constant
# over runs that do not all share an init. It tracks `malachite-saluki` (0.6977), used by every run
# from 2026-09-10 on; the four earlier resume runs (rt_sft n=1, cff_sft n=1, and the two frozen-core
# grafts) started from `nonchalant-malamute` at 0.7087, so their `vs init` reads 1.1 too high.
# Pass --arms to score one init's runs at a time if that matters.
SFT_INIT = 0.6977
ORDER = ["tuned_hrm", "tuned_rt", "cff_mlp_tied", "cff_mlp_untied", "cff_block_tied",
         "cff_block_untied", "rt_sft", "cff_sft", "cff_sft_lr10", "cff_sft_untied"]
RESUMES = {"rt_sft", "cff_sft", "cff_sft_lr10", "cff_sft_untied"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ORDER)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--min-evals", type=int, default=3, help="skip smoke tests")
    args = ap.parse_args()

    import wandb
    rows = {}
    for r in wandb.Api().runs(args.project, order="+created_at"):
        name = r.name.rsplit(" ", 1)[0]
        if name not in args.arms:
            continue
        vals = [x["eval/test_hard_exact_match"]
                for x in r.history(keys=["eval/test_hard_exact_match"], pandas=False)]
        if len(vals) >= args.min_evals:
            rows.setdefault(name, []).append(vals)

    print(f"{'arm':18s} {'n':>2s} {'best %':>16s} {'@ep':>5s} {'last %':>7s} {'vs init':>8s}")
    for name in args.arms:
        runs = rows.get(name, [])
        if not runs:
            continue
        bests = [max(v) for v in runs]
        mean = statistics.mean(bests) * 100
        spread = f"{mean:6.2f}" + (f" +-{statistics.stdev(bests) * 100:4.2f}" if len(bests) > 1 else "        ")
        at = statistics.mean(v.index(max(v)) + 1 for v in runs)
        last = statistics.mean(v[-1] for v in runs) * 100
        delta = f"{mean - SFT_INIT * 100:+7.2f}" if name in RESUMES else "      -"
        print(f"{name:18s} {len(runs):>2d} {spread:>16s} {at:>5.1f} {last:>7.2f} {delta:>8s}")
        for v in runs:
            print(f"   {[round(x * 100, 2) for x in v]}")

if __name__ == "__main__":
    main()

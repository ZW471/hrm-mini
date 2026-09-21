"""Build 10k / 100k / 1M training subsets of Sudoku-Extreme, nested and stratified by source.

The existing 1k subsample (`downloaded-datasets/sudoku-extreme-1k`) mirrors the full train set's
per-source mix (48.9 % puzzles4_forum_hardest_1905, 23 % 01_file1, 22 % puzzles1_unbiased, ...),
with matching rating and givens distributions. These subsets do the same, with two extra
guarantees that matter for the memorisation probe:

  * every subset CONTAINS the 1k train split -- the probe edits those 1,000 puzzles, so they must
    be training data at every size for the train split to stay a memorisation probe; and
  * the subsets are NESTED, 1k < 10k < 100k < 1M < full, so a bigger training set is strictly the
    smaller one plus more unique puzzles.

Construction: remove the 1k puzzles from the full train split, shuffle each source's remaining
rows once (seeded), and for each size take the first round((N - 1000) * share_source) rows of
every source. Per-source prefixes are nested, so the subsets are too. `test_hard.csv` is copied
from the 1k repo so every size evaluates on the identical 20k held-out puzzles.

Run:  uv run python dataset/build_sudoku_subsets.py
"""

import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "downloaded-datasets")
FULL, ONE_K = os.path.join(ROOT, "sudoku-extreme"), os.path.join(ROOT, "sudoku-extreme-1k")
SIZES = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000}
SEED = 0

README = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: test_hard
    path: test_hard.csv
---
# Sudoku-Extreme, {n:,}-puzzle training subset

Built by `dataset/build_sudoku_subsets.py` (seed {seed}): a stratified-by-source sample of the
full Sudoku-Extreme train split that contains the 1k subsample and is nested inside the next
size up (1k < 10k < 100k < 1M < full). `test_hard.csv` is the 1k repo's 20k held-out split.

| source | rows | share |
|---|---|---|
{table}
"""


def main():
    full = pd.read_csv(os.path.join(FULL, "train.csv"))
    one = pd.read_csv(os.path.join(ONE_K, "train.csv"))
    assert one.question.isin(full.question).all(), "the 1k split is not inside the full train split"
    rest = full[~full.question.isin(set(one.question))]
    share = rest.source.value_counts(normalize=True)
    print(f"full {len(full):,}  minus 1k -> {len(rest):,} rows over {len(share)} sources")

    rng = np.random.default_rng(SEED)
    shuffled = {s: g.iloc[rng.permutation(len(g))] for s, g in rest.groupby("source", sort=True)}

    for tag, n in SIZES.items():
        extra = n - len(one)
        # absorb rounding drift (a row or two either way) in the largest source, so the total is
        # exactly n; per-source prefixes stay nested because the largest source's cut only grows
        big = share.idxmax()
        taken = {s: int(round(extra * share[s])) for s in shuffled}
        taken[big] += extra - sum(taken.values())
        parts = [one] + [g.iloc[:taken[s]] for s, g in shuffled.items()]
        sub = pd.concat(parts, ignore_index=True)
        sub = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
        assert len(sub) == n, (tag, len(sub))
        assert sub.question.is_unique
        assert one.question.isin(sub.question).all()

        out = os.path.join(ROOT, f"sudoku-extreme-{tag}")
        os.makedirs(out, exist_ok=True)
        sub.to_csv(os.path.join(out, "train.csv"), index=False)
        shutil.copy(os.path.join(ONE_K, "test_hard.csv"), os.path.join(out, "test_hard.csv"))
        counts = sub.source.value_counts()
        table = "\n".join(f"| {s} | {c:,} | {c / n:.2%} |" for s, c in counts.items())
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(README.format(n=n, seed=SEED, table=table))
        print(f"  {tag:4s}: {len(sub):>9,} rows -> {out}")

    # nesting check
    prev = set(one.question)
    for tag in SIZES:
        q = set(pd.read_csv(os.path.join(ROOT, f"sudoku-extreme-{tag}", "train.csv")).question)
        assert prev <= q, f"{tag} does not contain the previous size"
        prev = q
    print("nesting verified: 1k < 10k < 100k < 1m < full")


if __name__ == "__main__":
    main()

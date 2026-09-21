"""Build the N = 1 / 10 / 100 training subsets of Sudoku-Extreme, nested inside the 1k split.

Sibling of `build_sudoku_subsets.py` for the bottom of the data-scaling curve. Same two properties:
the subsets are NESTED (1 < 10 < 100 < 1k, so every size is the smaller one plus more unique
puzzles) and stratified by source as far as N allows. Construction: shuffle the 1k train split once
(seeded), take round(100 * share_source) rows per source for N = 100 (rounding drift absorbed by the
largest source), shuffle those 100 once more into a fixed order; N = 10 is the first 10 rows of that
order and N = 1 the first row, with the order rotated so a `puzzles4_forum_hardest_1905` puzzle comes
first (49 % of the data and the hard tail `test_hard` is drawn from). Stratification is meaningless
at N <= 10, so those are just a fixed prefix.

Second draws `1b` / `10b` (rows 11-20 of the same order, `1b` = the first hardest_1905 puzzle among
them) exist because at this size *which* puzzle is a bigger effect than the training seed. They are
disjoint from `1` / `10` but still inside `100`.

`test_hard.csv` is copied from the 1k repo so every size evaluates on the identical 20k puzzles, and
each README lists the chosen puzzles so a subset can be checked by eye.

Run:  .venv/bin/python dataset/build_sudoku_tiny_subsets.py
"""

import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "downloaded-datasets")
ONE_K = os.path.join(ROOT, "sudoku-extreme-1k")
HARD = "puzzles4_forum_hardest_1905"
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
# Sudoku-Extreme, {n}-puzzle training subset{draw}

Built by `dataset/build_sudoku_tiny_subsets.py` (seed {seed}) from the 1k train split, nested
(1 < 10 < 100 < 1k) and stratified by source where N allows. `test_hard.csv` is the 1k repo's 20k
held-out split. Training applies the full board-symmetry x digit-relabel augmentation per batch, so
each row is one equivalence class of ~1e12 boards, not one sample.

| source | rows | share |
|---|---|---|
{table}

## Puzzles

| # | source | rating | question |
|---|---|---|---|
{rows}
"""


def write(tag, sub, draw=""):
    n = len(sub)
    out = os.path.join(ROOT, f"sudoku-extreme-{tag}")
    os.makedirs(out, exist_ok=True)
    sub.to_csv(os.path.join(out, "train.csv"), index=False)
    shutil.copy(os.path.join(ONE_K, "test_hard.csv"), os.path.join(out, "test_hard.csv"))
    counts = sub.source.value_counts()
    table = "\n".join(f"| {s} | {c:,} | {c / n:.0%} |" for s, c in counts.items())
    rows = "\n".join(f"| {i + 1} | {r.source} | {r.rating} | `{r.question}` |" for i, r in enumerate(sub.itertuples()))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(n=n, draw=draw, seed=SEED, table=table, rows=rows))
    print(f"  {tag:4s}: {n:>4} rows, {dict(counts)} -> {out}")


def main():
    one = pd.read_csv(os.path.join(ONE_K, "train.csv"))
    rng = np.random.default_rng(SEED)
    one = one.iloc[rng.permutation(len(one))].reset_index(drop=True)
    share = one.source.value_counts(normalize=True)

    # N = 100: per-source prefixes of the shuffled 1k, rounding drift absorbed by the largest source
    taken = {s: int(round(100 * share[s])) for s in share.index}
    taken[share.idxmax()] += 100 - sum(taken.values())
    hundred = pd.concat([g.iloc[:taken[s]] for s, g in one.groupby("source", sort=True)])
    hundred = hundred.iloc[rng.permutation(len(hundred))].reset_index(drop=True)
    # fixed order with a hardest_1905 puzzle first, so N = 1 is a hard-tail puzzle
    first = int(np.flatnonzero(hundred.source == HARD)[0])
    hundred = pd.concat([hundred.iloc[[first]], hundred.drop(index=first)]).reset_index(drop=True)
    assert len(hundred) == 100 and hundred.question.is_unique

    ten = hundred.iloc[:10]
    single = hundred.iloc[:1]
    ten_b = hundred.iloc[10:20].reset_index(drop=True)
    single_b = ten_b[ten_b.source == HARD].iloc[:1]
    assert len(single_b) == 1, "no hardest_1905 puzzle in rows 11-20; change the seed"

    write("100", hundred)
    write("10", ten)
    write("1", single)
    write("10b", ten_b, " (second draw)")
    write("1b", single_b, " (second draw)")

    # nesting check against the on-disk files
    q1k = set(one.question)
    for tag, inside in [("1", "10"), ("10", "100"), ("1b", "10b"), ("10b", "100")]:
        a = set(pd.read_csv(os.path.join(ROOT, f"sudoku-extreme-{tag}", "train.csv")).question)
        b = set(pd.read_csv(os.path.join(ROOT, f"sudoku-extreme-{inside}", "train.csv")).question)
        assert a <= b <= q1k, f"{tag} not inside {inside} / 1k"
    assert not set(ten.question) & set(ten_b.question)
    print("nesting verified: 1 < 10 < 100 < 1k (and 1b < 10b < 100, 10b disjoint from 10)")


if __name__ == "__main__":
    main()

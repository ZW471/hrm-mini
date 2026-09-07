"""Does the model solve Sudoku, or has it memorised the 1k training solutions?

The probe
---------
Take a puzzle the model was *trained* on and change a single given digit. The resulting puzzle is
a one-character edit of something the model has seen thousands of times (under augmentation), but
its solution is a different board -- typically dozens of cells change. So:

  * a model that actually solves should follow the new constraints: its prediction should match
    the NEW solution, and should differ from the old one on exactly the cells the solution moved.
  * a model that has memorised the training answers should ignore the edit and keep emitting the
    OLD solution, i.e. `pred(new) ~ pred(old)` even though `sol(new)` is far from `sol(old)`.

The decisive statistic is measured on the *diagnostic cells* -- the cells where the old and the
new solution disagree (the perturbed clue itself is excluded, since it is given in the input):

    copy rate   = P[pred(new)[i] == sol_old[i]]   over diagnostic cells i   -> memorisation
    solve rate  = P[pred(new)[i] == sol_new[i]]   over diagnostic cells i   -> real solving

Both are reported on all pairs and on the clean subset where the model reproduces the original
training solution exactly (only there is "copying" well defined).

Constructing the perturbed puzzle
---------------------------------
For each source puzzle we try (clue cell, new digit) candidates in random order and keep the first
that passes every test below, so the edit is a uniformly random qualifying one. Most edits fail --
~1.5% of the ~200 candidates per puzzle leave the puzzle solvable at all -- but a qualifying edit
exists for the large majority of puzzles.

  1. The edited puzzle has *exactly one* solution; an ambiguous puzzle would make "the" target
     undefined, and a contradictory one has no target at all.
  2. Its digit string does not appear in any split of the dataset.
  3. The orbit invariant of `eval_flow_sudoku.fingerprint` -- constant over the band/stack/row/
     column/transpose/digit-relabeling augmentation the training loader applies -- does not match
     any training solution. A non-match *proves* the puzzle is outside the training orbit; a match
     is inconclusive, so that edit is skipped and the next candidate is tried.

Prefer `last.pt` over `best.pt` here: `best.pt` is selected on test-set exact match, which for a
memorising run is an early, under-fit snapshot -- not the model whose memorisation is in question.
`--split test_hard` is the control: those puzzles were never trained on, so nothing can be copied,
and it also shows that the edited puzzles are no harder than the originals.

Usage
-----
    uv run python experiments/examine_failure_mode.py \
        --ckpt "hrm=checkpoints/tuned_hrm free-sturgeon/seed_1/last.pt" \
        --ckpt "mae=checkpoints/mae_flops_matched golden-raccoon/seed_1/last.pt"

Building the pairs takes a few minutes; `--pairs outputs/failure_mode/pairs_train.npz` reuses the
ones a previous run saved, so further checkpoints are scored on the identical puzzles.

Each run writes `pairs_<split>.npz` (every board and prediction), `summary_<split>.json` and
`metrics_<split>.csv` (one row per checkpoint, plus an oracle row) into `--out`.
"""

from typing import Any, Callable

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import tqdm
import yaml
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch.layers import Carry
from experiments.eval_flow_sudoku import fingerprint
from train import TrainConfig, generate, load_module, run_inference

# Below this many exactly-solved originals the copy/solve split is too noisy to read as a verdict
MIN_CLEAN_PAIRS = 10

# How far the edit moved the solution: pairs are bucketed by their number of diagnostic cells, so
# copy/solve can be read against the size of the change the model was asked to follow.
SHIFT_BUCKETS = [("1_10", 1, 10), ("11_20", 11, 20), ("21_30", 21, 30), ("31plus", 31, 81)]

DEFAULT_CKPTS = [
    "hrm=checkpoints/tuned_hrm free-sturgeon/seed_1/best.pt",
    "mae=checkpoints/mae_flops_matched golden-raccoon/seed_1/best.pt",
]

# --- Sudoku solving ---
def count_solutions(grid: np.ndarray, limit: int = 2) -> tuple[int, list[int] | None]:
    """Number of solutions of an [81] grid (0 = blank), capped at `limit`, plus the first one.

    Bitmask backtracking with most-constrained-cell ordering. Returns (0, None) if the givens
    already conflict.
    """
    board = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(board):
        if v:
            b, m = (i // 27) * 3 + (i % 9) // 3, 1 << (v - 1)
            if rows[i // 9] & m or cols[i % 9] & m or boxes[b] & m:
                return 0, None
            rows[i // 9] |= m
            cols[i % 9] |= m
            boxes[b] |= m

    found: list[list[int]] = []

    def rec():
        # Pick the blank cell with the fewest candidates
        best, best_mask, best_count = -1, 0, 10
        for i in range(81):
            if board[i] == 0:
                b = (i // 27) * 3 + (i % 9) // 3
                mask = 0x1FF & ~(rows[i // 9] | cols[i % 9] | boxes[b])
                count = bin(mask).count("1")
                if count < best_count:
                    best, best_mask, best_count = i, mask, count
                    if count <= 1:
                        break
        if best == -1:
            found.append(board.copy())
            return

        r, c, b = best // 9, best % 9, (best // 27) * 3 + (best % 9) // 3
        mask = best_mask
        while mask:
            m = mask & -mask
            mask ^= m
            board[best] = m.bit_length()
            rows[r] |= m
            cols[c] |= m
            boxes[b] |= m
            rec()
            board[best] = 0
            rows[r] ^= m
            cols[c] ^= m
            boxes[b] ^= m
            if len(found) >= limit:
                break

    rec()
    return len(found), found[0] if found else None

# --- Data ---
def load_split(dataset_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """[N, 81] questions (0 = blank) and answers for one CSV split."""
    rows = list(csv.DictReader(open(os.path.join(dataset_dir, f"{split}.csv"))))
    questions = np.stack([np.frombuffer(r["question"].replace(".", "0").encode(), np.uint8) for r in rows])
    answers = np.stack([np.frombuffer(r["answer"].encode(), np.uint8) for r in rows])
    return (questions - ord("0")).astype(np.int64), (answers - ord("0")).astype(np.int64)

def to_string(grid: np.ndarray) -> str:
    return "".join(str(int(v)) for v in grid)

def perturb(question: np.ndarray, rng: np.random.Generator,
            accept: Callable[[np.ndarray, np.ndarray], bool]) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    """Change one given digit so the puzzle stays uniquely solvable and `accept`s.

    Candidate edits are tried in random order, so the result is a uniformly random edit among the
    qualifying ones; an edit rejected by `accept` (the novelty filters) falls through to the next
    candidate instead of discarding the puzzle. Returns (cell, new_digit, new_question,
    new_solution), or None if the puzzle admits no qualifying edit.
    """
    clues = np.flatnonzero(question)
    candidates = [(int(c), d) for c in clues for d in range(1, 10) if d != question[c]]
    rng.shuffle(candidates)

    for cell, digit in candidates:
        new_question = question.copy()
        new_question[cell] = digit
        num, solution = count_solutions(new_question, limit=2)
        if num != 1:
            continue
        new_solution = np.array(solution, dtype=np.int64)
        if accept(new_question, new_solution):
            return cell, digit, new_question, new_solution
    return None

# --- Models ---
def is_flow_ckpt(ckpt: str) -> bool:
    """Flow runs save `args.json`; the train.py models save `model_config.json`."""
    return os.path.exists(os.path.join(os.path.dirname(ckpt), "args.json"))

@torch.inference_mode()
def predict_flow(ckpt: str, questions: np.ndarray, batch_size: int, args) -> np.ndarray:
    """Solve each puzzle with the flow model, using the sampler the eval protocol uses.

    Single-shot on purpose: the probe asks what the model does with one draw, not what a
    verified-restart search can find.
    """
    from experiments.eval_flow_sudoku import load_model as load_flow_model
    from experiments.flow_sudoku import sde_sample

    model, codec = load_flow_model(ckpt)
    conditional = getattr(model, "cond_embed", None) is not None
    if not conditional:
        raise SystemExit(f"{ckpt} is an unconditional model - it has no puzzle input to probe")

    device = torch.device("cuda")
    out = []
    for start in tqdm.trange(0, len(questions), batch_size, desc="  sampling", leave=False):
        q = torch.from_numpy(questions[start:start + batch_size]).to(device)
        mask = q > 0
        generator = torch.Generator(device=device).manual_seed(args.seed + start)
        x = sde_sample(model, q.shape[0], 81, codec.in_channels, args.flow_steps, device,
                       generator, args.flow_noise, cond=q, clamp_x1=codec.encode(q.clamp_min(1)),
                       clamp_mask=mask, guidance=args.flow_guidance, sampler=args.flow_sampler)
        out.append(torch.where(mask, q, codec.decode(x)).cpu().numpy())   # givens are given
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0).astype(np.int64)

def load_model(ckpt: str) -> tuple[nn.Module, TrainConfig, bool]:
    with open(os.path.join(os.path.dirname(ckpt), "model_config.json"), "r") as f:
        config = TrainConfig(**yaml.safe_load(f))

    model_cls = load_module(f"arch.{config.arch.name}")
    # Same metadata `dataset.sudoku.create_dataloader` hands the model (a decoder-only model
    # doubles the sequence and turns on causal masking itself).
    metadata = {"vocab_size": 10, "seq_len": 82, "is_causal": False}
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.arch.__pydantic_extra__ | metadata)
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True), assign=True)
        model.eval()
    return model, config, getattr(model_cls, "is_autoregressive", False)

@torch.inference_mode()
def predict(model: nn.Module, config: TrainConfig, is_autoregressive: bool,
            questions: np.ndarray, batch_size: int) -> np.ndarray:
    """[N, 81] puzzles -> [N, 81] predicted solutions, with the same decoding `eval.py` uses."""
    out = []
    for start in tqdm.trange(0, len(questions), batch_size, desc="  inference", leave=False):
        batch = np.pad(questions[start:start + batch_size], ((0, 0), (1, 0))).astype(np.int32)  # BOS
        x = torch.from_numpy(batch).cuda()
        if is_autoregressive:
            y_hat = generate(model, x)
        else:
            carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
            y_hat = None
            for _ in range(config.cycles_per_data):
                carry, y_hat = run_inference(model, carry, x)
        out.append(y_hat[:, 1:].cpu().numpy())  # drop the BOS slot
    return np.concatenate(out, axis=0).astype(np.int64)

# --- Metrics ---
def summarize(pred_old: np.ndarray, pred_new: np.ndarray,
              sol_old: np.ndarray, sol_new: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    """All statistics for one model. `cells` holds the perturbed cell of each pair."""
    solved_old = np.all(pred_old == sol_old, axis=-1)

    # Diagnostic cells: where the two solutions disagree, minus the perturbed clue (it is given).
    diagnostic = sol_old != sol_new
    diagnostic[np.arange(len(cells)), cells] = False

    num_diagnostic = diagnostic.sum(-1)
    board = np.ones_like(diagnostic)

    def rates(pairs: np.ndarray, scope: np.ndarray, exclusive: bool) -> dict[str, Any]:
        """copy / solve rate over the cells `scope` selects, on the pairs `pairs` selects."""
        sel = scope[pairs]
        if not sel.any():
            return {"copy_rate": float("nan"), "solve_rate": float("nan")} | ({"other_rate": float("nan")} if exclusive else {})
        copy = float((pred_new[pairs] == sol_old[pairs])[sel].mean())
        solve = float((pred_new[pairs] == sol_new[pairs])[sel].mean())
        # Only on the diagnostic cells are the two exclusive; over the whole board they overlap on
        # every cell the edit left alone, so they there sum to well over 1 and have no remainder.
        return {"copy_rate": copy, "solve_rate": solve} | ({"other_rate": 1.0 - copy - solve} if exclusive else {})

    everything = np.ones(len(cells), dtype=bool)
    return {
        "num_pairs": int(len(cells)),
        "num_solved_original": int(solved_old.sum()),
        "exact_match_original": float(solved_old.mean()),
        "exact_match_perturbed": float(np.all(pred_new == sol_new, axis=-1).mean()),
        "cell_acc_original": float((pred_old == sol_old).mean()),
        "cell_acc_perturbed": float((pred_new == sol_new).mean()),
        # How much the prediction moved, against how much the true solution moved
        "pred_agreement": float((pred_new == pred_old).mean()),
        "solution_agreement": float((sol_new == sol_old).mean()),
        "num_cells_solution_changed": float((sol_new != sol_old).sum(-1).mean()),
        "num_cells_pred_changed": float((pred_new != pred_old).sum(-1).mean()),
        "num_diagnostic_cells": float(diagnostic.sum(-1).mean()),
        "clue_echoed": float((pred_new[np.arange(len(cells)), cells] == sol_new[np.arange(len(cells)), cells]).mean()),
        # copy / solve at both scopes: the whole 81-cell board, and the diagnostic cells only
        "board": {"all": rates(everything, board, False),
                  "solved_original": rates(solved_old, board, False)},
        "diagnostic": {"all": rates(everything, diagnostic, True),
                       "solved_original": rates(solved_old, diagnostic, True)},
        # The same diagnostic-cell read, split by how far the edit moved the solution
        "by_shift": {name: {"num_pairs": int(bucket.sum()),
                            "em_edited": float(np.all(pred_new[bucket] == sol_new[bucket], axis=-1).mean())
                                         if bucket.any() else float("nan"),
                            **rates(bucket, diagnostic, True)}
                     for name, lo, hi, bucket in
                     [(n, lo, hi, (num_diagnostic >= lo) & (num_diagnostic <= hi)) for n, lo, hi in SHIFT_BUCKETS]},
    }

def csv_row(label: str, r: dict[str, Any]) -> dict[str, Any]:
    """One flat CSV record per model, over all pairs.

    `pred_agreement` is deliberately not a column: whenever a model reproduces the original solution
    exactly -- which on the train split it does for every pair -- it equals `board_copy`. The full
    set of statistics, that one included, is in `summary_<split>.json`.

    copy / solve are reported at both scopes: `board_*` over all 81 cells (where the two overlap on
    every cell the edit left alone) and `diag_*` over the diagnostic cells only (where they are
    mutually exclusive), and the diagnostic read is repeated per `SHIFT_BUCKETS` bucket as
    `diag_{copy,solve}_<bucket>`. See outputs/failure_mode/README.md for what each column means.
    """
    def num(x: float) -> float:
        return round(float(x), 4)

    row = {
        "model": label,
        "em_original": num(r["exact_match_original"]),
        "em_edited": num(r["exact_match_perturbed"]),
        "board_copy": num(r["board"]["all"]["copy_rate"]),
        "board_solve": num(r["board"]["all"]["solve_rate"]),
        "diag_copy": num(r["diagnostic"]["all"]["copy_rate"]),
        "diag_solve": num(r["diagnostic"]["all"]["solve_rate"]),
    }
    # The same copy / solve, split by how many cells the edit moved (see SHIFT_BUCKETS)
    for name, _lo, _hi in SHIFT_BUCKETS:
        b = r["by_shift"][name]
        row[f"diag_copy_{name}"] = num(b["copy_rate"])
        row[f"diag_solve_{name}"] = num(b["solve_rate"])
    return row

def render(name: str, grid: np.ndarray, reference: np.ndarray | None = None, cell: int | None = None) -> str:
    """9x9 grid; cells differing from `reference` are bracketed, the perturbed cell is starred."""
    lines = [name]
    for r in range(9):
        row = []
        for c in range(9):
            i = r * 9 + c
            v = "." if grid[i] == 0 else str(int(grid[i]))
            if i == cell:
                v = f"*{v}*"
            elif reference is not None and grid[i] != reference[i]:
                v = f"[{v}]"
            else:
                v = f" {v} "
            row.append(v)
        lines.append("".join(row))
    return "\n".join(lines)

def build_pairs(args) -> tuple[np.ndarray, ...]:
    """One uniquely-solvable, provably novel single-digit edit per source puzzle.

    Returns (source_index, cell, new_digit, question_old, question_new, solution_old, solution_new).
    """
    rng = np.random.default_rng(args.seed)
    questions, answers = load_split(args.dataset_dir, args.split)
    questions, answers = questions[:args.num_samples], answers[:args.num_samples]

    # Everything the model could have been shown: every puzzle string in the dataset, and every
    # augmentation orbit its training solutions live in.
    seen_questions = set()
    for split in ["train", "test_hard", "test_sudoku_bench"]:
        try:
            q, _ = load_split(args.dataset_dir, split)
        except FileNotFoundError:
            continue
        seen_questions.update(to_string(g) for g in q)
    train_fingerprints = {fingerprint(b) for b in load_split(args.dataset_dir, "train")[1]}

    rejected = {"duplicate_string": 0, "fingerprint_collision": 0}

    def is_novel(new_question: np.ndarray, new_solution: np.ndarray) -> bool:
        """The edited puzzle must be new to the model, both literally and up to augmentation."""
        if to_string(new_question) in seen_questions:
            rejected["duplicate_string"] += 1
            return False
        if fingerprint(new_solution) in train_fingerprints:
            rejected["fingerprint_collision"] += 1
            return False
        return True

    idx, cells, digits, q_new, s_new = [], [], [], [], []
    num_dropped = 0
    for i in tqdm.trange(len(questions), desc="perturbing"):
        edit = perturb(questions[i], rng, is_novel)
        if edit is None:
            num_dropped += 1
            continue
        cell, digit, new_question, new_solution = edit
        idx.append(i)
        cells.append(cell)
        digits.append(digit)
        q_new.append(new_question)
        s_new.append(new_solution)

    print(f"\n{len(idx)}/{len(questions)} usable pairs from split '{args.split}' "
          f"({num_dropped} puzzles had no uniquely-solvable, novel one-digit edit; "
          f"edits skipped by the novelty filters: {rejected})")
    if len(idx) == 0:
        return (np.array([]),) * 7

    idx = np.array(idx)
    return (idx, np.array(cells), np.array(digits),
            questions[idx], np.stack(q_new), answers[idx], np.stack(s_new))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, action="append", metavar="LABEL=PATH",
                        help="Checkpoint to probe, repeatable (default: the HRM and FLOP-matched MAE runs)")
    parser.add_argument("--dataset-dir", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    parser.add_argument("--split", type=str, default="train",
                        help="Split the source puzzles come from; 'train' is the memorisation probe, "
                             "a test split is the control (nothing to memorise there)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Source puzzles to attempt")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Reuse the (original, edited) pairs from a previous run's .npz instead of "
                             "rebuilding them, e.g. to probe more checkpoints on the identical puzzles")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--flow-sampler", type=str, default="heun", choices=["euler", "heun", "rk4"])
    parser.add_argument("--flow-steps", type=int, default=250)
    parser.add_argument("--flow-noise", type=float, default=20.0)
    parser.add_argument("--flow-guidance", type=float, default=2.0)
    parser.add_argument("--append", action="store_true",
                        help="Merge into the existing metrics CSV / summary / npz instead of replacing "
                             "them, so new checkpoints join the earlier ones on the same pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=2, help="Worked examples to print")
    parser.add_argument("--out", type=str, default="outputs/failure_mode",
                        help="Directory for the .npz dump and the JSON summary")
    args = parser.parse_args()

    ckpts = args.ckpt or DEFAULT_CKPTS
    labelled = []
    for spec in ckpts:
        label, _, path = spec.partition("=")
        labelled.append((label, path or label))

    # ----- Build (or reuse) the perturbed puzzles
    if args.pairs is not None:
        pairs = np.load(args.pairs)
        idx, cells, digits = pairs["source_index"], pairs["cell"], pairs["new_digit"]
        q_old, q_new = pairs["question_old"], pairs["question_new"]
        s_old, s_new = pairs["solution_old"], pairs["solution_new"]
        pairs.close()  # so --pairs and --out may point at the same file
        print(f"\nreusing {len(idx)} pairs from {args.pairs}")
    else:
        idx, cells, digits, q_old, q_new, s_old, s_new = build_pairs(args)
        if len(idx) == 0:
            return
    print(f"one changed given moves {(s_old != s_new).sum(-1).mean():.1f} of 81 solution cells on average "
          f"(min {(s_old != s_new).sum(-1).min()}, max {(s_old != s_new).sum(-1).max()})")

    # ----- Run the models
    results, predictions = {}, {}
    for label, path in labelled:
        print(f"\n[{label}] {path}")
        if is_flow_ckpt(path):
            print(f"  flow model: {args.flow_sampler}, {args.flow_steps} steps, "
                  f"noise {args.flow_noise}, guidance {args.flow_guidance}")
            pred_old = predict_flow(path, q_old, args.batch_size, args)
            pred_new = predict_flow(path, q_new, args.batch_size, args)
        else:
            model, config, is_autoregressive = load_model(path)
            pred_old = predict(model, config, is_autoregressive, q_old, args.batch_size)
            pred_new = predict(model, config, is_autoregressive, q_new, args.batch_size)
            del model
            torch.cuda.empty_cache()

        predictions[label] = (pred_old, pred_new)
        results[label] = summarize(pred_old, pred_new, s_old, s_new, cells)

    # ----- Report
    solution_agreement = results[labelled[0][0]]["solution_agreement"]
    oracle = {"exact_match_original": 1.0, "exact_match_perturbed": 1.0,
              "cell_acc_original": 1.0, "cell_acc_perturbed": 1.0,
              "pred_agreement": solution_agreement, "solution_agreement": solution_agreement,
              "num_pairs": len(idx), "num_solved_original": len(idx),
              "num_cells_solution_changed": float((s_old != s_new).sum(-1).mean()),
              "num_cells_pred_changed": float((s_old != s_new).sum(-1).mean()),
              "num_diagnostic_cells": results[labelled[0][0]]["num_diagnostic_cells"],
              "clue_echoed": 1.0,
              # A perfect solver emits sol_new everywhere: over the board it still matches sol_old
              # on every cell the edit left alone, on the diagnostic cells never.
              "board": {"all": {"copy_rate": solution_agreement, "solve_rate": 1.0},
                        "solved_original": {"copy_rate": solution_agreement, "solve_rate": 1.0}},
              "diagnostic": {"all": {"copy_rate": 0.0, "solve_rate": 1.0, "other_rate": 0.0},
                             "solved_original": {"copy_rate": 0.0, "solve_rate": 1.0, "other_rate": 0.0}},
              "by_shift": {name: {"num_pairs": results[labelled[0][0]]["by_shift"][name]["num_pairs"],
                                  "em_edited": 1.0, "copy_rate": 0.0, "solve_rate": 1.0, "other_rate": 0.0}
                           for name, _lo, _hi in SHIFT_BUCKETS}}
    rows = [(label, results[label]) for label, _ in labelled] + [("oracle", oracle)]

    header = (f"{'model':<13}{'EM orig':>10}{'EM edited':>11}{'cell orig':>11}{'cell edited':>13}"
              f"{'pred agree':>12}{'n pairs':>9}{'n solved':>10}")
    print("\n" + "=" * len(header))
    print("accuracy (each puzzle scored against its own solution)")
    print(header)
    print("-" * len(header))
    for label, r in rows:
        print(f"{label:<13}{r['exact_match_original']:>10.3f}{r['exact_match_perturbed']:>11.3f}"
              f"{r['cell_acc_original']:>11.3f}{r['cell_acc_perturbed']:>13.3f}"
              f"{r['pred_agreement']:>12.3f}{r['num_pairs']:>9d}{r['num_solved_original']:>10d}")
    print("=" * len(header))
    print("'pred agree' = how much the prediction stayed put across the edit; the oracle row is how")
    print("much the true solution stayed put, so 'pred agree' >> oracle means the model ignored the")
    print("edit. 'n solved' = pairs whose original solution the model reproduced exactly -- the only")
    print("pairs where there is a memorised answer to copy.")

    header = (f"{'model':<13}{'copy':>9}{'solve':>9}{'copy':>9}{'solve':>9}"
              f"{'copy':>9}{'solve':>9}{'copy':>9}{'solve':>9}")
    print("\n" + "=" * len(header))
    print("copy  = prediction on the edited puzzle equals the ORIGINAL solution (memorised)")
    print("solve = prediction on the edited puzzle equals its OWN solution (re-solved)")
    print(f"{'':<13}{'all 81 cells':^36}{'diagnostic cells only':^36}")
    print(f"{'':<13}{'all pairs':^18}{'solved orig':^18}{'all pairs':^18}{'solved orig':^18}")
    print(header)
    print("-" * len(header))
    for label, r in rows:
        b, d = r["board"], r["diagnostic"]
        print(f"{label:<13}"
              f"{b['all']['copy_rate']:>9.3f}{b['all']['solve_rate']:>9.3f}"
              f"{b['solved_original']['copy_rate']:>9.3f}{b['solved_original']['solve_rate']:>9.3f}"
              f"{d['all']['copy_rate']:>9.3f}{d['all']['solve_rate']:>9.3f}"
              f"{d['solved_original']['copy_rate']:>9.3f}{d['solved_original']['solve_rate']:>9.3f}")
    print("=" * len(header))
    n_same = 81 * solution_agreement
    n_diag = rows[0][1]["num_diagnostic_cells"]
    print(f"Over all 81 cells the two rates OVERLAP: the edit leaves ~{n_same:.0f} of 81 cells unchanged and a")
    print(f"correct prediction there counts as both, which is why even the oracle scores {solution_agreement:.3f} copy.")
    print(f"The diagnostic cells -- the ~{n_diag:.0f} cells the edit actually moved, perturbed clue excluded --")
    print("are the unambiguous read: there copy and solve are mutually exclusive and the oracle")
    print("scores 0.000 copy / 1.000 solve. 'solved orig' restricts to the pairs whose original")
    print("solution the model reproduced exactly.")

    # ----- Memorisation vs. how far the edit moved the solution
    counts = rows[0][1]["by_shift"]
    header = f"{'model':<17}" + "".join(f"{f'{lo}-{hi if hi < 81 else str(lo)+chr(43)}':>19}"
                                        for _n, lo, hi in SHIFT_BUCKETS)
    print("\n" + "=" * len(header))
    print("copy / solve on the diagnostic cells, by how many cells the edit moved the solution")
    print(f"{'diag cells moved':<17}" + "".join(f"{f'{lo}-{hi}' if hi < 81 else f'{lo}+':>19}"
                                                for _n, lo, hi in SHIFT_BUCKETS))
    print(f"{'n pairs':<17}" + "".join(f"{counts[n]['num_pairs']:>19d}" for n, _lo, _hi in SHIFT_BUCKETS))
    print(f"{'':<17}" + "".join(f"{'copy':>9}{'solve':>10}" for _ in SHIFT_BUCKETS))
    print("-" * len(header))
    for label, r in rows:
        print(f"{label:<17}" + "".join(f"{r['by_shift'][n]['copy_rate']:>9.3f}{r['by_shift'][n]['solve_rate']:>10.3f}"
                                       for n, _lo, _hi in SHIFT_BUCKETS))
    print("=" * len(header))
    print("A memoriser's copy rate should hold up (or rise) as the edit moves more cells: the further")
    print("the true answer travels, the more of the old board it is still emitting.")

    for label, _ in labelled:
        a, c = results[label]["diagnostic"]["all"], results[label]["diagnostic"]["solved_original"]
        n_solved = results[label]["num_solved_original"]
        if n_solved < MIN_CLEAN_PAIRS:
            print(f"\n[{label}] reproduced only {n_solved}/{len(idx)} of the original solutions exactly, "
                  f"so it has little memorised answer to copy in the first place; over all pairs it puts the "
                  f"old value on {a['copy_rate']:.1%} of moved cells and the correct new one on {a['solve_rate']:.1%}.")
        elif c["copy_rate"] > c["solve_rate"]:
            print(f"\n[{label}] MEMORISATION: on cells the edit moved, it keeps the old training answer "
                  f"{c['copy_rate']:.1%} of the time vs {c['solve_rate']:.1%} for the correct new value.")
        else:
            print(f"\n[{label}] re-solves: {c['solve_rate']:.1%} of moved cells take the correct new value "
                  f"vs {c['copy_rate']:.1%} that stay at the memorised one.")

    # ----- Worked examples
    for k in range(min(args.num_examples, len(idx))):
        print("\n" + "=" * 96)
        print(f"example {k} (source #{idx[k]}, cell {cells[k]} r{cells[k] // 9}c{cells[k] % 9}: "
              f"{q_old[k][cells[k]]} -> {digits[k]})")
        print(render("puzzle (edited given starred)", q_new[k], q_old[k], int(cells[k])))
        print(render("true solution of the edited puzzle ([] = moved by the edit)", s_new[k], s_old[k]))
        for label, _ in labelled:
            pred_old, pred_new = predictions[label]
            print(render(f"{label} prediction ([] = wrong)", pred_new[k], s_new[k]))

    # ----- Persist
    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, f"pairs_{args.split}.npz")
    arrays = {f"pred_old_{l}": p[0] for l, p in predictions.items()}
    arrays |= {f"pred_new_{l}": p[1] for l, p in predictions.items()}
    if args.append and os.path.exists(npz_path):
        with np.load(npz_path) as old:      # keep the earlier models' predictions
            arrays = {k: old[k] for k in old.files if k.startswith("pred_")} | arrays
    np.savez(npz_path, source_index=idx, cell=cells, new_digit=digits,
             question_old=q_old, question_new=q_new, solution_old=s_old, solution_new=s_new, **arrays)

    summary_path = os.path.join(args.out, f"summary_{args.split}.json")
    summary = {"split": args.split, "seed": args.seed, "checkpoints": dict(labelled), "results": results}
    if args.append and os.path.exists(summary_path):
        previous = json.load(open(summary_path))
        summary["checkpoints"] = previous.get("checkpoints", {}) | summary["checkpoints"]
        summary["results"] = previous.get("results", {}) | summary["results"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.out, f"metrics_{args.split}.csv")
    new_rows = {label: csv_row(label, r) for label, r in rows}
    merged = {}
    if args.append and os.path.exists(csv_path):
        for row in csv.DictReader(open(csv_path)):
            merged[row["model"]] = row
    merged.update(new_rows)
    merged = {k: v for k, v in merged.items() if k != "oracle"} | {"oracle": merged["oracle"]}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row("", rows[0][1]).keys()))
        writer.writeheader()
        writer.writerows(merged.values())

    print(f"\nwrote {args.out}/{{pairs,summary,metrics}}_{args.split}.{{npz,json,csv}}")

if __name__ == "__main__":
    main()

"""Sample from a trained flow-matching checkpoint and score the boards.

Reports, for a sweep of Euler step counts, how often a sample is a fully valid Sudoku solution,
and then runs a novelty test on the valid boards.

Novelty test
------------
Every training board comes with a huge augmentation orbit (band/stack/row/column permutations,
transpose, digit relabeling), so "did the model just memorise?" cannot be answered by string
equality. Instead we use a group invariant: for a solved board let pi_d map row -> column of
digit d. For digits d != e the permutation pi_d o pi_e^-1 acts on the rows, and its cycle type is
unchanged by column permutation (it cancels), by row permutation (conjugation), by digit
relabeling (it only renames the pair) and by transpose. The multiset of these cycle types over all
36 digit pairs is therefore constant on an orbit.

A generated board whose fingerprint matches no training board is PROVABLY not an augmented copy of
any training solution. A match is inconclusive -- the invariant is not a complete one.

Usage
-----
    uv run python experiments/eval_flow_sudoku.py --ckpt checkpoints/<run>/best.pt
"""

from collections import Counter
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.flow_sudoku import (BoardCodec, SudokuFlowTransformer, board_metrics,
                                     euler_sample, group_view, render_boards, sde_sample)

def is_valid(boards: np.ndarray) -> np.ndarray:
    """[N, 81] -> [N] bool: all 9 rows, 9 columns and 9 boxes are a permutation of 1..9."""
    return (np.sort(group_view(boards), axis=-1) == np.arange(1, 10)).all(axis=-1).all(axis=-1)

def cycle_type(p: np.ndarray) -> tuple[int, ...]:
    seen, out = np.zeros(9, bool), []
    for i in range(9):
        if seen[i]:
            continue
        n, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = p[j]
            n += 1
        out.append(n)
    return tuple(sorted(out))

def fingerprint(board: np.ndarray):
    """Orbit invariant of a solved board (see the module docstring)."""
    g = board.reshape(9, 9)
    pi = np.zeros((10, 9), int)   # pi[d][row] = column holding digit d in that row
    inv = np.zeros((10, 9), int)
    for d in range(1, 10):
        r, c = np.where(g == d)
        pi[d][r] = c
        inv[d][pi[d]] = np.arange(9)

    types = Counter()
    for d in range(1, 10):
        for e in range(d + 1, 10):
            types[cycle_type(pi[d][inv[e]])] += 1
    return tuple(sorted(types.items()))

def load_model(ckpt: str) -> tuple[torch.nn.Module, BoardCodec]:
    args = json.load(open(os.path.join(os.path.dirname(ckpt), "args.json")))
    codec = BoardCodec(args["repr_name"])
    with torch.device("cuda"):
        model = SudokuFlowTransformer(dict(
            seq_len=args["seq_len"], num_layers=args["num_layers"], hidden_size=args["hidden_size"],
            intermediate_size=args["intermediate_size"], head_dim=args["head_dim"], is_causal=False,
            norm_eps=args["norm_eps"], rope_theta=args["rope_theta"],
            in_channels=codec.in_channels, forward_dtype=args["forward_dtype"],
            qk_norm=args.get("qk_norm", False), conditional=args.get("conditional", False),
            adaln=args.get("adaln", False),
            # checkpoints from before --pos-embed existed recorded a rope_2d bool
            pos_embed=args.get("pos_embed", "rope2d" if args.get("rope_2d") else "rope1d")))
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True), assign=True)
        model.eval()
    return model, codec

def solve_puzzles(model, codec, args, device) -> None:
    """Use a model as a Sudoku solver by inpainting: pin the given cells, sample the rest.

    For an *unconditional* model this is zero-shot -- it was only ever trained to generate whole
    boards, and the puzzle enters solely as "these cells are already known", which is generic
    conditional sampling, not a Sudoku-specific mechanism.
    """
    from experiments.flow_sudoku import load_pairs
    q, a = load_pairs(args.dataset_dir, args.solve_split)
    pick = np.random.default_rng(0).permutation(len(q))[:args.num_samples]
    puzzles = torch.from_numpy(q[pick].astype(np.int64)).to(device)
    truth = torch.from_numpy(a[pick].astype(np.int64)).to(device)
    mask = puzzles > 0
    clamp_x1 = codec.encode(puzzles.clamp_min(1))

    print(f"solving {len(pick)} puzzles from '{args.solve_split}' "
          f"({float((~mask).sum(1).float().mean()):.1f} blanks each)", flush=True)
    cond = puzzles if getattr(model, "cond_embed", None) is not None else None
    for guide in args.guidance:
      for noise in args.noise_scale:
        for steps in args.sample_steps:
            gen = torch.Generator(device=device).manual_seed(args.seed)
            solved = torch.zeros(len(pick), dtype=torch.bool, device=device)   # verified solutions
            first, curve = None, []
            for attempt in range(args.restarts):
                x = sde_sample(model, len(pick), 81, codec.in_channels, steps, device, gen,
                               noise, cond=cond, clamp_x1=clamp_x1, clamp_mask=mask,
                               guidance=guide, sampler=args.sampler, codec=codec,
                               reveal_threshold=args.reveal_threshold, reveal_every=args.reveal_every)
                pred = torch.where(mask, puzzles, codec.decode(x))
                boards = pred.cpu().numpy()
                # A candidate is ACCEPTED on checkable evidence only: it is a valid board and it
                # agrees with the givens. The ground truth is never consulted to select it.
                ok = torch.from_numpy(is_valid(boards)).to(device)
                solved |= ok & (pred == truth).all(dim=-1)   # scored after acceptance
                if attempt == 0:
                    first = (float((pred == truth).all(dim=-1).float().mean()),
                             float((pred == truth).float().mean()),
                             float(board_metrics(boards)["valid_board_rate"]))
                curve.append((attempt + 1, float(solved.float().mean())))
            print(f"guide={guide:<4g} noise={noise:<4g} steps={steps:4d}  exact_match={first[0]:.4f}  "
                  f"cell_accuracy={first[1]:.4f}  valid_board={first[2]:.4f}"
                  + (f"  solved@{args.restarts}={solved.float().mean().item():.4f}" if args.restarts > 1 else ""),
                  flush=True)
            if args.restarts > 1:
                print("  solve_curve " + " ".join(f"{k}:{v:.4f}" for k, v in curve), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=[20, 50, 100, 250])
    parser.add_argument("--novelty-steps", type=int, default=100, help="Which step count to run the novelty test at")
    parser.add_argument("--train-csv", type=str, default="./downloaded-datasets/sudoku-extreme-1k/train.csv")
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0],
                        help="Stochastic sampler noise levels; 0 is the deterministic Euler ODE")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun", "rk4"])
    parser.add_argument("--reveal-threshold", type=float, default=None,
                        help="Commit cells whose endpoint estimate is this decided (0-1) and treat them "
                             "as givens for the rest of sampling; None disables")
    parser.add_argument("--reveal-every", type=int, default=10, help="Steps between reveal passes")
    parser.add_argument("--solve", action="store_true", help="Solve puzzles by inpainting instead of generating")
    parser.add_argument("--solve-split", type=str, default="test_hard")
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0], help="Classifier-free guidance scales")
    parser.add_argument("--restarts", type=int, default=1,
                        help="Draw this many candidates per puzzle and accept a verified-valid one")
    parser.add_argument("--dataset-dir", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    args = parser.parse_args()

    model, codec = load_model(args.ckpt)
    device = torch.device("cuda")
    if args.solve:
        solve_puzzles(model, codec, args, device)
        return

    # A model trained with condition dropout is BOTH models: hand it an all-blank puzzle and it
    # generates unconditionally, hand it a puzzle and it solves. Same weights either way.
    uncond_cond = (torch.zeros(args.num_samples, 81, dtype=torch.long, device=device)
                   if getattr(model, "cond_embed", None) is not None else None)
    valid_boards, best = None, -1.0
    for noise in args.noise_scale:
        for steps in args.sample_steps:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            x = (euler_sample(model, args.num_samples, 81, codec.in_channels, steps, device, generator)
                 if noise == 0 else
                 sde_sample(model, args.num_samples, 81, codec.in_channels, steps, device, generator, noise,
                            cond=uncond_cond, sampler=args.sampler))
            boards = codec.decode(x).cpu().numpy()
            m = board_metrics(boards)
            print(f"noise={noise:<4g} steps={steps:4d}  valid={m['valid_board_rate']:.4f}  "
                  f"group={m['group_satisfaction']:.4f} (row {m['row_satisfaction']:.3f} / "
                  f"col {m['col_satisfaction']:.3f} / box {m['box_satisfaction']:.3f})"
                  f"  unique={m['unique_sample_rate']:.4f}", flush=True)
            if steps == args.novelty_steps and m["valid_board_rate"] > best:
                best, valid_boards = m["valid_board_rate"], boards[is_valid(boards)]

    if valid_boards is None or len(valid_boards) == 0:
        print("\nNo valid boards at --novelty-steps; skipping the novelty test.")
        return

    rows = list(csv.DictReader(open(args.train_csv)))
    train = np.stack([np.frombuffer(r["answer"].encode(), np.uint8) - ord("0") for r in rows]).astype(np.int64)
    train_fps = {fingerprint(b) for b in train}

    novel = sum(fingerprint(b) not in train_fps for b in valid_boards)
    print(f"\n--- novelty @ {args.novelty_steps} steps ---")
    print(f"valid boards: {len(valid_boards)}/{args.num_samples}, all distinct: "
          f"{len(np.unique(valid_boards, axis=0)) == len(valid_boards)}")
    print(f"provably outside the training augmentation orbit: {novel}/{len(valid_boards)} "
          f"({novel / len(valid_boards):.1%})")
    print(f"fingerprint collides with a training board (inconclusive): {len(valid_boards) - novel}")
    print("\nexample generated valid board:\n" + render_boards(valid_boards[:1]))

if __name__ == "__main__":
    main()

"""Solve test puzzles with a trained discrete-flow checkpoint under a sweep of sampler settings.

    uv run python experiments/eval_dfm_sudoku.py --ckpt checkpoints/dfm_113m_mask_cfg_full/seed_1/best.pt \
        --sample-steps 16 64 256 --noise-scale 0 1 3 10 --guidance 1 2 3

Reports single-shot exact match per (steps, eta, guidance, temperature) cell, then optionally
`--restarts N` verified restarts (a valid board certifies itself, so the answer is never consulted).
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dfm_sudoku import SudokuDiscreteFlowTransformer, ctmc_sample
from flow_sudoku import load_pairs, board_metrics, group_view

def is_valid(boards: np.ndarray) -> np.ndarray:
    groups = group_view(boards)
    return (np.sort(groups, axis=-1) == np.arange(1, 10)).all(axis=-1).all(axis=-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset-dir", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", type=str, default="test_hard")
    p.add_argument("--num-puzzles", type=int, default=512)
    p.add_argument("--sample-steps", type=int, nargs="+", default=[64])
    p.add_argument("--noise-scale", type=float, nargs="+", default=[0.0])
    p.add_argument("--guidance", type=float, nargs="+", default=[2.0])
    p.add_argument("--temperature", type=float, nargs="+", default=[1.0])
    p.add_argument("--final", type=str, default="argmax", choices=["argmax", "sample"])
    p.add_argument("--restarts", type=int, default=1, help="Verified restarts at the best single-shot setting")
    p.add_argument("--clamp-givens", action=argparse.BooleanOptionalAction, default=None,
                   help="Write the puzzle back into the state at every step (default: the run's --givens mode)")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (as train.py / eval.py do)")
    args = p.parse_args()

    device = torch.device("cuda")
    run_dir = os.path.dirname(args.ckpt)
    cfg = json.load(open(os.path.join(run_dir, "args.json")))
    model = SudokuDiscreteFlowTransformer(dict(
        seq_len=81, num_layers=cfg["num_layers"], hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"], head_dim=cfg["head_dim"], is_causal=False,
        norm_eps=cfg["norm_eps"], rope_theta=cfg["rope_theta"], forward_dtype=cfg["forward_dtype"],
        pos_embed=cfg["pos_embed"], conditional=cfg["conditional"], qk_norm=cfg["qk_norm"],
        self_cond=cfg.get("self_cond", False))).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    if args.compile:
        model = torch.compile(model, dynamic=False, fullgraph=True)
    prior = cfg["prior"]

    q, a = load_pairs(args.dataset_dir, args.split)
    pick = np.random.default_rng(0).permutation(len(q))[:args.num_puzzles]
    cond = torch.from_numpy(q[pick].astype(np.int64)).to(device)
    truth = torch.from_numpy(a[pick].astype(np.int64)).to(device)
    clamp = (cfg.get("givens", "hard") == "hard") if args.clamp_givens is None else args.clamp_givens
    given = (cond > 0) if clamp else None
    print(f"{args.ckpt}: prior={prior}, givens={cfg.get('givens', 'hard')}, clamp={clamp}, "
          f"{cfg['num_params'] / 1e6:.2f}M params, {len(pick)} puzzles from {args.split}", flush=True)

    def solve(steps, eta, guidance, temperature, seed):
        outs = []
        for lo in range(0, cond.shape[0], args.batch):
            c = cond[lo:lo + args.batch]
            g = torch.Generator(device=device).manual_seed(seed + lo)
            outs.append(ctmc_sample(model, c.shape[0], prior, steps, device, g, cond=c,
                                    given=None if given is None else given[lo:lo + args.batch],
                                    eta=eta, guidance=guidance, temperature=temperature, final=args.final,
                                    self_cond=cfg.get("self_cond", False)))
        return torch.cat(outs)

    if args.compile:   # warm up so the wall clock below is steady-state
        solve(args.sample_steps[0], args.noise_scale[0], args.guidance[0], args.temperature[0], args.seed)
    best = (-1.0, None)
    for steps, eta, guidance, temp in itertools.product(args.sample_steps, args.noise_scale, args.guidance, args.temperature):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pred = solve(steps, eta, guidance, temp, args.seed)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
        em = (pred == truth).all(dim=-1).float().mean().item()
        cell = (pred == truth).float().mean().item()
        valid = board_metrics(pred.cpu().numpy())["valid_board_rate"]
        print(f"steps={steps:4d}  eta={eta:5g}  guidance={guidance:4g}  temp={temp:4g}  "
              f"exact_match={em:.4f}  cell_accuracy={cell:.4f}  valid_board={valid:.4f}  "
              f"wall={elapsed:.1f}s ({1000 * elapsed / cond.shape[0]:.2f} ms/puzzle)", flush=True)
        if em > best[0]:
            best = (em, (steps, eta, guidance, temp))

    if args.restarts > 1:
        steps, eta, guidance, temp = best[1]
        print(f"\nverified restarts at steps={steps} eta={eta} guidance={guidance} temp={temp}", flush=True)
        solved = torch.zeros(cond.shape[0], dtype=torch.bool, device=device)
        correct = torch.zeros_like(solved)
        for r in range(args.restarts):
            pred = solve(steps, eta, guidance, temp, args.seed + 1000 * (r + 1))
            valid = torch.from_numpy(is_valid(pred.cpu().numpy())).to(device)
            newly = valid & ~solved
            correct |= newly & (pred == truth).all(dim=-1)
            solved |= valid
            if (r + 1) in (1, 2, 4, 8, 16, 32, 64) or r + 1 == args.restarts:
                print(f"  restarts={r + 1:3d}  solved={solved.float().mean().item():.4f}  "
                      f"(correct among solved: {correct.sum().item()}/{solved.sum().item()})", flush=True)

if __name__ == "__main__":
    main()

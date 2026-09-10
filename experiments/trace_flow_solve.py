"""Trace a flow-matching solve of a Sudoku puzzle, step by step, into a Markdown file.

The solver is the one `eval_flow_sudoku.py --solve` uses: the conditional flow model is a velocity
field v(x_t, t) over the 81 cells of a 9x9 "image", integrated from t=0 (Gaussian noise) to t=1
(a board) by the marginal-preserving SDE

    dx = [v + (g^2/2) * score] dt + g dW,    score = (t*v - x) / (1 - t),   g(t) = noise * (1 - t),

with the given cells pinned to their own forward-noised value at every step and classifier-free
guidance on the puzzle conditioning. Nothing in the loop knows a Sudoku rule.

What gets printed at each step is the model's *endpoint estimate*

    x1_hat = x + (1 - t_next) * v(x_t, t),

decoded by argmax -- "given where the flow is now and which way it is pointing, what board does it
think it is heading for". That is the same quantity `sde_sample`'s adaptive-reveal branch uses to
decide a cell is settled. At t=1 it coincides with the sampled board.

Candidate acceptance is the verified-restart protocol from the README: several trajectories are
integrated in parallel and one is accepted iff it is a fully valid board that agrees with the
givens. The answer key is never consulted to choose; only afterwards, to report.

Usage
-----
    uv run python experiments/trace_flow_solve.py --ckpt checkpoints/flow_113m_cfg_full/best.pt
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.flow_sudoku import BoardCodec, board_metrics, load_pairs
from experiments.eval_flow_sudoku import is_valid, load_model


@torch.inference_mode()
def _drift(model, x, t, cond, guidance, noise_scale):
    """Drift of the marginal-preserving SDE and the raw velocity, exactly as in `sde_sample`."""
    t_batch = torch.full((x.shape[0],), t, device=x.device)
    v = model(x, t_batch, cond).float()
    if guidance != 1.0 and cond is not None:
        v_uncond = model(x, t_batch, torch.zeros_like(cond)).float()
        v = v_uncond + guidance * (v - v_uncond)
    if noise_scale <= 0:
        return v, v
    return v + 0.5 * (noise_scale ** 2) * (1.0 - t) * (t * v - x), v


@torch.inference_mode()
def traced_sample(model, codec, num, steps, device, generator, noise_scale, cond,
                  clamp_x1, clamp_mask, guidance, sampler):
    """`sde_sample`, but recording the decoded endpoint estimate after every step.

    Returns (x_final [num, 81, C] on device; est, conf, raw as [steps, num, 81] CPU arrays) --
    the endpoint-estimate board, its per-cell decision margin, and the argmax of the raw state.
    """
    x = torch.randn(num, 81, codec.in_channels, device=device, generator=generator)
    dt = 1.0 / steps
    est, confs, raws = [], [], []
    for i in range(steps):
        t, t_next = i * dt, min((i + 1) * dt, 1.0)
        if clamp_mask is not None:
            # Replacement-style conditioning: known cells are pinned to their own forward-noised
            # value, so the model only has to fill in the rest.
            known = (1.0 - t) * torch.randn(x.shape, device=device, generator=generator) + t * clamp_x1
            x = torch.where(clamp_mask[..., None], known, x)

        dw = (noise_scale * (1.0 - t) * math.sqrt(dt)
              * torch.randn(x.shape, device=device, generator=generator)) if noise_scale > 0 else 0.0

        k1, v1 = _drift(model, x, t, cond, guidance, noise_scale)
        if sampler == "euler":
            x = x + k1 * dt + dw
        elif sampler == "heun":                       # 2nd order: predictor, then trapezoid
            x_pred = x + k1 * dt + dw
            k2, _ = _drift(model, x_pred, t_next, cond, guidance, noise_scale)
            x = x + 0.5 * (k1 + k2) * dt + dw
        else:
            raise ValueError(f"unknown sampler: {sampler}")

        x1_hat = x + (1.0 - t_next) * v1
        top2 = x1_hat.topk(2, dim=-1).values
        est.append((x1_hat.argmax(dim=-1) + 1).to("cpu", torch.uint8))
        # margin in units of a clean one-hot gap: 1.0 means fully decided
        confs.append(((top2[..., 0] - top2[..., 1]) * codec.ONEHOT_STD).cpu().to(torch.float16))
        raws.append(codec.decode(x).to("cpu", torch.uint8))
    stack = lambda s: torch.stack(s).numpy()
    return x, stack(est), stack(confs), stack(raws)


def render(board, given, changed) -> str:
    """One board as 11 lines of HTML: givens bold, cells that just changed underlined."""
    lines = []
    for r in range(9):
        cells = []
        for c in range(9):
            d = str(int(board[r * 9 + c]))
            if given[r * 9 + c]:
                d = f"<b>{d}</b>"
            elif changed[r * 9 + c]:
                d = f"<u>{d}</u>"
            cells.append(d)
        lines.append("  ".join(" ".join(cells[k:k + 3]) for k in (0, 3, 6)))
        if r in (2, 5):
            lines.append("")
    return "\n".join(lines)


def render_puzzle(puzzle) -> str:
    lines = []
    for r in range(9):
        cells = [f"<b>{d}</b>" if d else "." for d in puzzle[r * 9:r * 9 + 9]]
        lines.append("  ".join(" ".join(cells[k:k + 3]) for k in (0, 3, 6)))
        if r in (2, 5):
            lines.append("")
    return "\n".join(lines)


def write_trace_markdown(path, meta, puzzle, truth, trace, conf, raw):
    """`trace`/`conf`/`raw`: [steps + 1, 81]; the last row is the accepted board at t=1."""
    given = puzzle > 0
    final = trace[-1]
    solved = bool((final == truth).all())
    steps = meta["steps"]

    md = []
    para = lambda s: md.append(s.rstrip() + "\n\n")

    para("# How a flow-matching model solves a Sudoku")
    para(f"**Checkpoint** `{meta['ckpt']}` — a 118M-parameter encoder transformer trained on the "
         "full 3.8M-puzzle `sudoku-extreme` split as a continuous flow-matching velocity field. "
         "It is not autoregressive and contains no solver: the whole 9×9 board is a point in a "
         "continuous space, and it moves from Gaussian noise to a solution along one trajectory.")
    para(f"**Sampler** `{meta['sampler']}` SDE, {steps} steps, noise scale {meta['noise']:g}, "
         f"classifier-free guidance {meta['guidance']:g}, givens re-pinned at every step.")
    para(f"**Puzzle** `{meta['split']}[{meta['index']}]` — {int(given.sum())} givens, "
         f"{81 - int(given.sum())} blanks. {meta['accepted']}/{meta['restarts']} trajectories "
         f"drawn in parallel produced a verified-valid board; trajectory #{meta['which']}, traced "
         f"below, {'passed' if meta['ok'] else '**failed**'} verification. Its final board "
         f"{'matches' if solved else '**does not match**'} the answer key.")
    para("**Reading the boards.** Each block is the model's *endpoint estimate* at that step — "
         "`x1_hat = x_t + (1−t)·v(x_t, t)`, decoded by argmax. That is the board the flow is "
         "currently heading for, not the noisy state it is passing through. At t = 1 the two "
         "coincide.")
    para("- **Bold** — a given: pinned by the puzzle, never moves.\n"
         "- <u>Underlined</u> — this digit changed from the previous step.\n"
         "- Plain — a cell the model filled in and has since left alone.")
    para("## The puzzle")
    para("<pre>\n" + render_puzzle(puzzle) + "\n</pre>")
    para("## The trajectory")

    prev, curve = None, []
    for i in range(len(trace)):
        board = trace[i]
        changed = np.zeros(81, bool) if prev is None else (board != prev)
        changed &= ~given
        n_changed, n_right = int(changed.sum()), int((board == truth).sum())
        decided = float((conf[i] > 0.5).mean())
        n_raw = int((raw[i] == board).sum())   # how far the noisy state still is from the target
        if i < steps:
            para(f"### Step {i + 1}/{steps} — t {i / steps:.3f} → {(i + 1) / steps:.3f}")
        else:
            para(f"### Final board — t = 1, givens re-imposed, "
                 f"{'verified valid' if meta['ok'] else '**REJECTED: not a legal board**'}")
        para(f"{f'{n_changed} cells changed' if prev is not None else 'first estimate'} · "
             f"{n_right}/81 match the solution · {decided:.0%} of cells decided · "
             f"raw state x_t agrees with the estimate in {n_raw}/81 cells")
        para("<pre>\n" + render(board, given, changed) + "\n</pre>")
        curve.append((i + 1, n_changed, n_right, n_raw))
        prev = board

    first_correct = next((k for k, _, r, _ in curve if r == 81), None)
    settle = next((k for k, c, _, _ in curve[1:] if c == 0), None)
    para("## Convergence")
    para(f"- The endpoint estimate becomes the correct solution at step "
         f"{first_correct if first_correct else 'never before the final board'} of {steps}, and "
         f"stops changing at step {settle if settle else 'never'}.\n"
         f"- The remaining steps are not idle: the *raw* state x_t is still travelling, agreeing "
         f"with the endpoint estimate in {curve[len(curve) // 2][3]}/81 cells at the halfway "
         f"point and {curve[-2][3]}/81 at the last step. The argmax settles long before the "
         f"continuous state arrives.\n"
         f"- Final board: valid = {bool(is_valid(final[None])[0])}, exact match against the "
         f"answer key = {solved}, cells differing from the solution = "
         f"{int((final != truth).sum())}/81.")
    para("| step | cells changed | correct /81 | x_t agrees /81 |\n|---:|---:|---:|---:|\n"
         + "\n".join(f"| {k} | {c} | {r} | {q} |" for k, c, r, q in curve))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("".join(md))
    return dict(solved=solved, first_correct=first_correct, settle=settle)


def solve_one(model, codec, puzzle, restarts, steps, device, seed, noise, guidance, sampler):
    """Draw `restarts` trajectories for one puzzle; return (traces, accepted index, n_accepted)."""
    puzzles = torch.from_numpy(puzzle.astype(np.int64)).to(device)[None].expand(restarts, 81).contiguous()
    mask = puzzles > 0
    gen = torch.Generator(device=device).manual_seed(seed)
    x, est, conf, raw = traced_sample(model, codec, restarts, steps, device, gen, noise, puzzles,
                                      codec.encode(puzzles.clamp_min(1)), mask, guidance, sampler)
    final = torch.where(mask, puzzles, codec.decode(x)).cpu().numpy().astype(np.uint8)
    # Accepted on checkable evidence only: a valid board that agrees with the givens.
    ok = is_valid(final) & ((final * (puzzle > 0)) == puzzle * (puzzle > 0)).all(axis=-1)
    accepted = np.flatnonzero(ok)
    return est, conf, raw, final, accepted


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default="checkpoints/flow_113m_cfg_full/best.pt")
    p.add_argument("--dataset-dir", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", type=str, default="test_hard")
    p.add_argument("--puzzle-index", type=int, default=0, help="Index into the split")
    p.add_argument("--steps", type=int, default=64, help="Integrator steps = rows in the trace")
    p.add_argument("--sampler", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--noise-scale", type=float, default=10.0)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--restarts", type=int, default=8,
                   help="Trajectories drawn in parallel; the first verified-valid one is traced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/flow_trace/solve_trace.md")
    p.add_argument("--trace-rejected", action="store_true",
                   help="Trace a trajectory that failed verification instead of an accepted one")
    args = p.parse_args()

    device = torch.device("cuda")
    model, codec = load_model(args.ckpt)
    assert getattr(model, "cond_embed", None) is not None, "checkpoint is not a conditional model"

    q, a = load_pairs(args.dataset_dir, args.split)
    puzzle, truth = q[args.puzzle_index].astype(np.int64), a[args.puzzle_index].astype(np.int64)

    est, conf, raw, final, accepted = solve_one(model, codec, puzzle, args.restarts, args.steps,
                                                device, args.seed, args.noise_scale,
                                                args.guidance, args.sampler)
    if args.trace_rejected:
        rejected = np.setdiff1d(np.arange(args.restarts), accepted)
        assert len(rejected), "every trajectory verified; nothing to trace"
        which = int(rejected[0])
    else:
        which = int(accepted[0]) if len(accepted) else 0
    trace = np.concatenate([est[:, which], final[which][None]], axis=0).astype(np.int64)
    conf1 = np.concatenate([conf[:, which], conf[-1, which][None]], axis=0)
    raw1 = np.concatenate([raw[:, which], final[which][None]], axis=0).astype(np.int64)

    meta = dict(ckpt=args.ckpt, split=args.split, index=args.puzzle_index, steps=args.steps,
                sampler=args.sampler, noise=args.noise_scale, guidance=args.guidance,
                accepted=len(accepted), restarts=args.restarts, which=which,
                ok=bool(which in accepted))
    r = write_trace_markdown(args.out, meta, puzzle, truth, trace, conf1, raw1)
    print(f"accepted {len(accepted)}/{args.restarts}; traced #{which}; solved={r['solved']}; "
          f"81/81 first reached at step {r['first_correct']}; settled at {r['settle']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

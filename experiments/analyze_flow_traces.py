"""How does the flow model actually solve a Sudoku? Trace many solves and measure.

`trace_flow_solve.py` renders one trajectory for a human to read. This script draws trajectories
for a whole slice of `test_hard`, records the model's endpoint estimate at every integrator step,
and tests specific claims about the mechanism against measurable quantities:

  1. how much of the final answer is already present in the very first estimate,
  2. whether the estimate improves monotonically or is revised,
  3. whether cells are settled progressively (like constraint propagation) or all at once,
  4. whether the cells the model settles first are the ones a logic solver deduces first,
  5. whether revisions are locally coupled (conflicting cells move together),
  6. how close to a legal board the intermediate estimates are,
  7. what the sampler's noise term contributes (noise 0 is the deterministic ODE),
  8. whether harder puzzles take the model more steps.

Everything is measured on the *endpoint estimate* x1_hat = x_t + (1-t) v(x_t, t) decoded by
argmax, which is the model's running answer; see `trace_flow_solve.py` for why.

Usage
-----
    uv run python experiments/analyze_flow_traces.py --num-puzzles 128 --restarts 8
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.flow_sudoku import BoardCodec, group_view, load_pairs
from experiments.eval_flow_sudoku import is_valid, load_model
from experiments.trace_flow_solve import traced_sample, write_trace_markdown

# [Sudoku reference machinery]
ROW = np.arange(81) // 9
COL = np.arange(81) % 9
BOX = (ROW // 3) * 3 + (COL // 3)
PEERS = np.zeros((81, 81), bool)
for _i in range(81):
    PEERS[_i] = (ROW == ROW[_i]) | (COL == COL[_i]) | (BOX == BOX[_i])
    PEERS[_i, _i] = False
UNITS = np.stack([np.eye(9, dtype=bool)[ROW], np.eye(9, dtype=bool)[COL],
                  np.eye(9, dtype=bool)[BOX]], 0).transpose(0, 2, 1).reshape(27, 81)


def propagation_depth(puzzle: np.ndarray) -> np.ndarray:
    """Round in which a naked/hidden-singles solver deduces each cell.

    Givens are round 0; each round assigns *every* cell that is a naked single (one candidate
    left) or a hidden single (the only cell in some unit that can hold a digit), simultaneously.
    Cells that the singles rules never reach -- the puzzle needs case analysis to finish -- get
    -1. This is the standard "how deep is the propagation" measure of difficulty, and it gives a
    per-cell logical ordering to compare the model's settling order against.
    """
    grid = puzzle.copy().astype(np.int64)
    depth = np.where(grid > 0, 0, -1)
    for rnd in range(1, 82):
        blank = grid == 0
        cand = np.ones((81, 9), bool)
        for c in np.flatnonzero(~blank):
            cand[PEERS[c], grid[c] - 1] = False
        cand[~blank] = False

        new = np.zeros(81, np.int64)
        naked = blank & (cand.sum(1) == 1)                           # only one digit fits here
        new[naked] = cand[naked].argmax(1) + 1
        for u in UNITS:                                              # only one cell fits this digit
            for d in range(9):
                spots = np.flatnonzero(u & blank & cand[:, d])
                if len(spots) == 1:
                    new[spots[0]] = d + 1
        if not new.any():
            break
        depth[new > 0] = rnd
        grid[new > 0] = new[new > 0]
    return depth


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (average ranks for ties)."""
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def bad_groups(boards: np.ndarray) -> np.ndarray:
    """[..., 81] digits -> [..., 27] bool: which rows/cols/boxes are not a permutation of 1..9."""
    flat = boards.reshape(-1, 81)
    g = np.sort(group_view(flat), axis=-1)
    bad = (g != np.arange(1, 10)).any(axis=-1)
    return bad.reshape(boards.shape[:-1] + (27,))


def violated_groups(boards: np.ndarray) -> np.ndarray:
    """[..., 81] digits -> [...] count of the 27 groups that are not a permutation."""
    return bad_groups(boards).sum(-1)


def in_violated_group(boards: np.ndarray) -> np.ndarray:
    """[..., 81] digits -> [..., 81] bool: is this cell inside at least one broken group?"""
    bad = bad_groups(boards).reshape(-1, 27)
    return (bad.astype(np.float32) @ UNITS.astype(np.float32) > 0).reshape(boards.shape)


# [Sampling]
@torch.inference_mode()
def run_traces(model, codec, puzzles_np, restarts, steps, device, seed, noise, guidance,
               sampler, chunk_rows=512):
    """Draw `restarts` trajectories per puzzle. Returns est/conf [S,P,R,81] and final [P,R,81]."""
    P = len(puzzles_np)
    rows = np.repeat(np.arange(P), restarts)
    N = P * restarts
    est_out = np.zeros((steps, N, 81), np.uint8)
    conf_out = np.zeros((steps, N, 81), np.float16)
    fin_out = np.zeros((N, 81), np.uint8)
    for lo in range(0, N, chunk_rows):
        hi = min(lo + chunk_rows, N)
        pz = torch.from_numpy(puzzles_np[rows[lo:hi]].astype(np.int64)).to(device)
        mask = pz > 0
        gen = torch.Generator(device=device).manual_seed(seed + lo)
        x, est, conf, _ = traced_sample(model, codec, hi - lo, steps, device, gen, noise, pz,
                                        codec.encode(pz.clamp_min(1)), mask, guidance, sampler)
        est_out[:, lo:hi], conf_out[:, lo:hi] = est, conf
        fin_out[lo:hi] = torch.where(mask, pz, codec.decode(x)).cpu().numpy().astype(np.uint8)
        print(f"  rows {hi}/{N}", flush=True)
    r = lambda v, tail: v.reshape((steps, P, restarts, 81) if tail else (P, restarts, 81))
    return r(est_out, True), r(conf_out, True), r(fin_out, False)


def cell_covariates(puzzle):
    """Per-cell difficulty proxies available before any solving: candidates left after eliminating
    the givens from each blank cell's 20 peers, and how many of those peers are givens."""
    grid = puzzle.astype(np.int64)
    cand = np.ones((81, 9), bool)
    for c in np.flatnonzero(grid > 0):
        cand[PEERS[c], grid[c] - 1] = False
    return cand.sum(1), (PEERS & (grid > 0)[None]).sum(1)


# [Analysis]
def analyse(est, conf, final, puzzles, truths, steps):
    """est/conf [S,P,R,81], final [P,R,81] -> measurements over the accepted trajectories."""
    P, R = final.shape[:2]
    given = puzzles > 0
    gv = np.broadcast_to(given[:, None], (P, R, 81))

    ok = is_valid(final.reshape(-1, 81)).reshape(P, R)
    agrees = (final * gv == puzzles[:, None] * gv).all(-1)
    accepted = ok & agrees                      # decidable without the answer key
    correct = (final == truths[:, None]).all(-1)

    pi, ri = np.nonzero(accepted)
    A = len(pi)
    traj = est[:, pi, ri].astype(np.int64)      # [S, A, 81]
    cnf = conf[:, pi, ri].astype(np.float32)
    fin = final[pi, ri].astype(np.int64)
    tru, gvn = truths[pi], given[pi]
    blank = ~gvn

    # --- when does each cell take its final value for good ---
    mism = traj != fin[None]
    lock = np.where(mism.any(0), steps - 1 - mism[::-1].argmax(0), -1) + 2   # 1-based
    board_lock = lock.max(1)
    n_correct = (traj == tru[None]).sum(-1)                                  # [S, A]
    reached = (n_correct == 81).any(0)
    first_correct = np.where(reached, (n_correct == 81).argmax(0) + 1, steps + 1)

    changed = np.zeros_like(mism)
    changed[1:] = (traj[1:] != traj[:-1]) & blank[None]
    n_changed = changed.sum(-1)
    switches = changed.sum(0)                                                # [A, 81]
    non_monotone = (np.diff(n_correct, axis=0) < 0).any(0)

    # --- locality: when a cell is revised, are its 20 peers revised too? ---
    peer_changed = np.einsum("sac,cd->sad", changed.astype(np.float32),
                             PEERS.astype(np.float32)) > 0
    p_base = float(changed.mean())
    p_cond = float((changed & peer_changed).sum() / max(changed.sum(), 1))
    p_indep = 1 - (1 - p_base) ** 20

    # --- how legal is the running estimate ---
    bad = bad_groups(traj)                                                   # [S, A, 27]
    viol = bad.sum(-1)                                                       # [S, A] of 27

    # --- is the repair aimed at the conflicts? P(revised next | currently in a broken group) ---
    conflicted = in_violated_group(traj)[:-1] & blank[None]                  # [S-1, A, 81]
    nxt = changed[1:]                                                        # revised at step k+1
    clean = (~in_violated_group(traj)[:-1]) & blank[None]
    p_fix = float(nxt[conflicted].mean()) if conflicted.any() else float("nan")
    p_idle = float(nxt[clean].mean()) if clean.any() else float("nan")
    # and does a revision actually reduce the number of broken groups?
    dviol = np.diff(viol, axis=0)
    moved = n_changed[1:] > 0
    viol_drop = float(dviol[moved].mean()) if moved.any() else float("nan")

    # --- settling order vs logical difficulty ---
    depth = np.stack([propagation_depth(p) for p in puzzles])
    cov = [cell_covariates(p) for p in puzzles]
    ncand = np.stack([c[0] for c in cov])
    npeer = np.stack([c[1] for c in cov])
    dep, nc, npg = depth[pi], ncand[pi], npeer[pi]
    rho_cand, rho_peer, rho_depth = [], [], []
    for k in range(A):
        b = blank[k]
        if b.sum() < 5 or len(np.unique(lock[k][b])) < 2:
            continue
        rho_cand.append(spearman(lock[k][b].astype(float), nc[k][b].astype(float)))
        rho_peer.append(spearman(lock[k][b].astype(float), npg[k][b].astype(float)))
        m = b & (dep[k] > 0)
        if m.sum() >= 5 and len(np.unique(dep[k][m])) > 1:
            rho_depth.append(spearman(lock[k][m].astype(float), dep[k][m].astype(float)))
    singles_frac = float((depth[~given] > 0).mean())     # cells elementary rules ever reach

    # --- is the model's own confidence at step 1 informative? ---
    settled_at_1 = (traj[0] == fin) & blank
    moves_later = (traj[0] != fin) & blank
    conf1_settled = float(cnf[0][settled_at_1].mean()) if settled_at_1.any() else float("nan")
    conf1_moves = float(cnf[0][moves_later].mean()) if moves_later.any() else float("nan")

    # --- do independent restarts take different routes to the same answer? ---
    same_final, same_step1 = [], []
    for p in range(P):
        idx = np.flatnonzero(accepted[p])
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                same_final.append(float((final[p, idx[i]] == final[p, idx[j]]).mean()))
                same_step1.append(float((est[0, p, idx[i]] == est[0, p, idx[j]]).mean()))
    # the same two quantities without conditioning on acceptance (no selection effect)
    all_step1 = [float((est[0, p, i] == est[0, p, j]).mean())
                 for p in range(P) for i in range(R) for j in range(i + 1, R)]
    n_correct_all = (est == truths[None, :, None]).sum(-1)                    # [S, P, R]

    # --- failures ---
    fail = final[~accepted].astype(np.int64)
    fail_viol = violated_groups(fail) if len(fail) else np.zeros(0)
    fail_wrong = ((fail != truths[:, None].repeat(R, 1)[~accepted]).sum(-1)
                  if len(fail) else np.zeros(0))

    return dict(
        P=P, R=R, A=A, accept_rate=float(accepted.mean()),
        solve_rate_any=float(correct.any(1).mean()), single_shot=float(correct[:, 0].mean()),
        accepted_and_wrong=int((accepted & ~correct).sum()),
        first_est_correct=float(n_correct[0].mean()),
        solved_at_step1=float((first_correct == 1).mean()),
        solved_by_step4=float((first_correct <= 4).mean()),
        first_est_final=float((traj[0] == fin).mean()),
        n_correct=n_correct, n_changed=n_changed, viol=viol, lock=lock, board_lock=board_lock,
        first_correct=first_correct, blank=blank, switches=switches,
        non_monotone=float(non_monotone.mean()),
        max_drop=float(np.max(-np.min(np.diff(n_correct, axis=0), axis=0))) if A else 0.0,
        mean_drop=float(np.mean(-np.minimum(np.min(np.diff(n_correct, axis=0), axis=0), 0))),
        p_base=p_base, p_cond=p_cond, p_indep=p_indep,
        p_fix=p_fix, p_idle=p_idle, viol_drop=viol_drop,
        rho_cand=float(np.nanmean(rho_cand)) if rho_cand else float("nan"),
        rho_peer=float(np.nanmean(rho_peer)) if rho_peer else float("nan"),
        rho_depth=float(np.nanmean(rho_depth)) if rho_depth else float("nan"),
        n_rho_depth=len(rho_depth), singles_frac=singles_frac,
        conf1_settled=conf1_settled, conf1_moves=conf1_moves,
        same_final=float(np.mean(same_final)) if same_final else float("nan"),
        same_step1=float(np.mean(same_step1)) if same_step1 else float("nan"),
        same_step1_all=float(np.mean(all_step1)) if all_step1 else float("nan"),
        first_est_correct_all=float(n_correct_all[0].mean()),
        curve_all=n_correct_all.reshape(len(est), -1).mean(1),
        n_fail=int(len(fail)), fail_viol=fail_viol, fail_wrong=fail_wrong,
        depth=depth, ncand=ncand, pi=pi, ri=ri, accepted=accepted, correct=correct,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default="checkpoints/flow_113m_cfg_full/best.pt")
    p.add_argument("--dataset-dir", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    p.add_argument("--split", type=str, default="test_hard")
    p.add_argument("--num-puzzles", type=int, default=128)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--sampler", type=str, default="heun")
    p.add_argument("--noise-scale", type=float, default=10.0)
    p.add_argument("--ablate-noise", type=float, default=0.0,
                   help="Second run at this noise level (0 = the deterministic ODE)")
    p.add_argument("--ablate-restarts", type=int, default=4)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/flow_trace")
    p.add_argument("--reuse", action="store_true", help="Reuse traces.npz instead of resampling")
    args = p.parse_args()

    device = torch.device("cuda")
    model, codec = load_model(args.ckpt)
    q, a = load_pairs(args.dataset_dir, args.split)
    puzzles = q[:args.num_puzzles].astype(np.int64)
    truths = a[:args.num_puzzles].astype(np.int64)

    print(f"main run: {args.num_puzzles} puzzles x {args.restarts} restarts, noise {args.noise_scale}")
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "traces.npz")
    if args.reuse and os.path.exists(cache):
        z = np.load(cache)
        est, conf, final = z["est"], z["conf"], z["final"]
        est0, conf0, final0 = z["est_ablate"], z["conf_ablate"], z["final_ablate"]
        print(f"reusing {cache}")
    else:
        est, conf, final = run_traces(model, codec, puzzles, args.restarts, args.steps, device,
                                      args.seed, args.noise_scale, args.guidance, args.sampler)
        print(f"ablation: noise {args.ablate_noise}")
        est0, conf0, final0 = run_traces(model, codec, puzzles, args.ablate_restarts, args.steps,
                                         device, args.seed + 99991, args.ablate_noise,
                                         args.guidance, args.sampler)
        np.savez_compressed(cache, est=est, conf=conf, final=final, puzzles=puzzles,
                            truths=truths, est_ablate=est0, conf_ablate=conf0,
                            final_ablate=final0)
        print(f"wrote {cache}")

    main_stats = analyse(est, conf, final, puzzles, truths, args.steps)
    abl_stats = analyse(est0, conf0, final0, puzzles, truths, args.steps)

    scalars = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
    json.dump({"main": scalars(main_stats), "ablation": scalars(abl_stats), "args": vars(args)},
              open(os.path.join(args.out_dir, "analysis.json"), "w"), indent=2, default=str)
    arrays = ("n_correct", "n_changed", "viol", "lock", "board_lock", "first_correct", "blank",
              "switches", "depth", "ncand", "pi", "ri", "accepted", "correct", "fail_viol",
              "fail_wrong", "curve_all")
    np.savez_compressed(os.path.join(args.out_dir, "stats.npz"),
                        **{f"main_{k}": main_stats[k] for k in arrays},
                        **{f"abl_{k}": abl_stats[k] for k in arrays})

    # A compact console digest; the Markdown report is written by report_flow_traces.py
    s = main_stats
    print(f"\naccept rate {s['accept_rate']:.3f}  solved@{args.restarts} {s['solve_rate_any']:.3f}  "
          f"single-shot {s['single_shot']:.3f}  accepted-but-wrong {s['accepted_and_wrong']}")
    print(f"first estimate: {s['first_est_correct']:.1f}/81 correct, "
          f"{s['first_est_final']:.3f} of the final board already in place")
    print(f"board freezes at step: median {np.median(s['board_lock']):.0f}, "
          f"p90 {np.percentile(s['board_lock'], 90):.0f}, max {s['board_lock'].max()}")
    print(f"non-monotone trajectories: {s['non_monotone']:.3f}")
    print(f"changes coupled: P(peer changed | changed) {s['p_cond']:.3f} vs {s['p_indep']:.3f} independent")
    print(f"targeted repair: P(revised | in a broken group) {s['p_fix']:.4f} vs "
          f"{s['p_idle']:.4f} for cells in only-legal groups; a revising step changes the "
          f"broken-group count by {s['viol_drop']:+.2f}")
    print(f"spearman(lock, #candidates) {s['rho_cand']:.3f}, (lock, #peer givens) {s['rho_peer']:.3f}, "
          f"(lock, singles depth) {s['rho_depth']:.3f} on {s['n_rho_depth']} trajectories")
    print(f"elementary singles rules ever reach {s['singles_frac']:.3f} of the blanks")
    print(f"confidence at step 1: settled cells {s['conf1_settled']:.3f} vs later-revised {s['conf1_moves']:.3f}")
    print(f"restarts: agree {s['same_step1']:.3f} at step 1 (accepted only), "
          f"{s['same_step1_all']:.3f} over all restarts, {s['same_final']:.3f} at the end")
    print(f"first estimate over ALL trajectories: {s['first_est_correct_all']:.1f}/81 correct")
    print(f"ablation noise {args.ablate_noise}: accept {abl_stats['accept_rate']:.3f}, "
          f"switches/cell {abl_stats['switches'].mean():.2f} vs {s['switches'].mean():.2f}")
    print(f"\nwrote {args.out_dir}/traces.npz and analysis.json")


if __name__ == "__main__":
    main()

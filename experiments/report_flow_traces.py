"""Turn the measurements in `outputs/flow_trace/` into figures and a stats block.

Reads `analysis.json` + `stats.npz` written by `analyze_flow_traces.py` and emits:
  fig_convergence.png  -- correct cells and constraint violations against integrator step
  fig_lock.png         -- when cells stop changing (per cell, and per board)
  fig_noise.png        -- the same trajectory statistics with the sampler's noise term removed
  stats.md             -- every number quoted in the report, with its symbol

Usage
-----
    uv run python experiments/report_flow_traces.py --dir outputs/flow_trace
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACCENT, MUTED = "#1b1b1b", "#c2410c", "#94a3b8"

_R, _C = np.arange(81) // 9, np.arange(81) % 9
_B = (_R // 3) * 3 + (_C // 3)
PEERS_F = (((_R[:, None] == _R) | (_C[:, None] == _C) | (_B[:, None] == _B))
           & ~np.eye(81, dtype=bool)).astype(np.float32)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/flow_trace")
    a = ap.parse_args()
    J = json.load(open(os.path.join(a.dir, "analysis.json")))
    S = np.load(os.path.join(a.dir, "stats.npz"))
    m, b = J["main"], J["ablation"]
    steps = J["args"]["steps"]
    x = np.arange(1, steps + 1)

    # --- convergence -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=160)
    nc = S["main_n_correct"]                       # [S, A] accepted trajectories
    ax[0].plot(x, S["main_curve_all"], color=MUTED, lw=1.6, label="all trajectories")
    ax[0].plot(x, nc.mean(1), color=ACCENT, lw=2, label="verified-valid ones")
    lo, hi = np.percentile(nc, [10, 90], axis=1)
    ax[0].fill_between(x, lo, hi, color=ACCENT, alpha=0.15, lw=0)
    ax[0].axhline(81, color=INK, lw=0.8, ls=":")
    ax[0].set_xlabel("integrator step"); ax[0].set_ylabel("cells matching the solution / 81")
    ax[0].set_title("Most of the answer is there after one evaluation", fontsize=9, color=INK)
    ax[0].legend(fontsize=7, frameon=False)
    style(ax[0])

    v = S["main_viol"]
    ax[1].plot(x, v.mean(1), color=ACCENT, lw=2)
    lo, hi = np.percentile(v, [10, 90], axis=1)
    ax[1].fill_between(x, lo, hi, color=ACCENT, alpha=0.15, lw=0)
    ax[1].set_xlabel("integrator step"); ax[1].set_ylabel("violated groups / 27")
    ax[1].set_title("Constraint violations fall steadily to zero", fontsize=9, color=INK)
    style(ax[1])
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "fig_convergence.png")); plt.close(fig)

    # --- when things stop moving -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=160)
    lock, blank = S["main_lock"], S["main_blank"]
    ax[0].hist(lock[blank], bins=np.arange(1, steps + 2) - 0.5, color=ACCENT, alpha=0.85)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step at which a cell takes its final value")
    ax[0].set_ylabel("blank cells (log)")
    ax[0].set_title("Most cells never move at all", fontsize=9, color=INK)
    style(ax[0])

    ax[1].hist(S["main_board_lock"], bins=np.arange(1, steps + 2) - 0.5, color=ACCENT, alpha=0.85,
               label=f"noise {J['args']['noise_scale']:g}")
    ax[1].hist(S["abl_board_lock"], bins=np.arange(1, steps + 2) - 0.5, color=MUTED, alpha=0.7,
               label=f"noise {J['args']['ablate_noise']:g} (ODE)")
    ax[1].set_xlabel("step at which the whole board freezes")
    ax[1].set_ylabel("trajectories")
    ax[1].set_title("All the work happens in the first ~20 steps", fontsize=9, color=INK)
    ax[1].legend(fontsize=7, frameon=False)
    style(ax[1])
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "fig_lock.png")); plt.close(fig)

    # --- what the noise term buys ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=160)
    for key, col, lab in (("main", ACCENT, f"noise {J['args']['noise_scale']:g}"),
                          ("abl", MUTED, f"noise {J['args']['ablate_noise']:g} (ODE)")):
        sw = S[f"{key}_switches"][S[f"{key}_blank"]]
        ax[0].hist(sw, bins=np.arange(0, 9) - 0.5, alpha=0.7, color=col, label=lab, log=True)
        ax[1].plot(x, S[f"{key}_n_changed"].mean(1), color=col, lw=2, label=lab)
    ax[0].set_xlabel("times a blank cell is revised"); ax[0].set_ylabel("cells (log)")
    ax[0].set_title("Revisions per cell", fontsize=9, color=INK)
    ax[1].set_xlabel("integrator step"); ax[1].set_ylabel("cells revised at this step")
    ax[1].set_title("Revision rate over the trajectory", fontsize=9, color=INK)
    for k in (0, 1):
        ax[k].legend(fontsize=7, frameon=False); style(ax[k])
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, "fig_noise.png")); plt.close(fig)

    # --- what the model gets wrong at step 1 -------------------------------------------------
    T = np.load(os.path.join(a.dir, "traces.npz"))
    est, puzzles, truths = T["est"], T["puzzles"], T["truths"]
    blank_p = puzzles == 0                                          # [P, 81]
    ncand = S["main_ncand"]                                         # [P, 81]
    e1 = est[0].astype(np.int64)                                    # [P, R, 81]
    wrong = (e1 != truths[:, None]) & blank_p[:, None]
    right = (e1 == truths[:, None]) & blank_p[:, None]
    cand_b = np.broadcast_to(ncand[:, None], wrong.shape)
    cand_wrong, cand_right = float(cand_b[wrong].mean()), float(cand_b[right].mean())
    # clustering: how many of a wrong cell's 20 peers are also wrong, vs the base rate
    peer_wrong = np.einsum("prc,cd->prd", wrong.astype(np.float32), PEERS_F)
    clust = float(peer_wrong[wrong].mean())
    base = float(wrong.mean() * 20)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=160)
    bins = np.arange(1.5, 10.5)
    ax[0].hist(cand_b[right], bins=bins, density=True, alpha=0.75, color=MUTED,
               label=f"correct at step 1 (mean {cand_right:.2f})")
    ax[0].hist(cand_b[wrong], bins=bins, density=True, alpha=0.75, color=ACCENT,
               label=f"wrong at step 1 (mean {cand_wrong:.2f})")
    ax[0].set_xlabel("candidate digits left after eliminating the givens")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("Local under-constraint does NOT explain the errors", fontsize=9, color=INK)
    ax[1].bar(["observed", "if independent"], [clust, base], color=[ACCENT, MUTED], width=0.55)
    ax[1].set_ylabel("wrong peers per wrong cell")
    ax[1].set_title("The errors arrive in mutually conflicting groups", fontsize=9, color=INK)
    for k in (0, 1):
        style(ax[k])
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig_first_errors.png")); plt.close(fig)

    # --- the numbers -------------------------------------------------------------------------
    def q(name, arr, fmt="{:.2f}"):
        return (f"| {name} | " + " | ".join(fmt.format(z) for z in
                np.percentile(arr, [10, 50, 90])) + f" | {fmt.format(np.mean(arr))} |")

    out = ["# Measured quantities\n",
           f"\n{m['P']} puzzles x {m['R']} restarts = {m['P'] * m['R']} trajectories at noise "
           f"{J['args']['noise_scale']:g}; {m['A']} were verified-valid and are the ones the "
           f"trajectory statistics below are computed over.\n",
           "\n| quantity | p10 | median | p90 | mean |\n|---|---:|---:|---:|---:|\n"]
    out.append(q("cells matching the solution in the FIRST estimate", S["main_n_correct"][0], "{:.1f}") + "\n")
    out.append(q("step at which the whole board freezes", S["main_board_lock"], "{:.0f}") + "\n")
    out.append(q("step at which the estimate first equals the solution", S["main_first_correct"], "{:.0f}") + "\n")
    out.append(q("violated groups (of 27) in the first estimate", S["main_viol"][0], "{:.1f}") + "\n")
    out.append(q("revisions per blank cell", S["main_switches"][S["main_blank"]], "{:.2f}") + "\n")
    out.append(q("step at which a blank cell takes its final value", S["main_lock"][S["main_blank"]], "{:.0f}") + "\n")

    out.append("\n## Scalars\n\n| quantity | noise "
               f"{J['args']['noise_scale']:g} | noise {J['args']['ablate_noise']:g} (ODE) |\n"
               "|---|---:|---:|\n")
    rows = [("verified-valid trajectories", "accept_rate", "{:.4f}"),
            ("single-shot exact match", "single_shot", "{:.4f}"),
            ("puzzles solved by at least one restart", "solve_rate_any", "{:.4f}"),
            ("accepted but wrong (verification failures)", "accepted_and_wrong", "{:.0f}"),
            ("cells correct in the first estimate, all trajectories", "first_est_correct_all", "{:.1f}"),
            ("first estimate already equals the final board", "first_est_final", "{:.4f}"),
            ("estimate is the full solution at step 1", "solved_at_step1", "{:.4f}"),
            ("estimate is the full solution by step 4", "solved_by_step4", "{:.4f}"),
            ("trajectories whose accuracy ever drops", "non_monotone", "{:.4f}"),
            ("largest single-step drop in correct cells", "max_drop", "{:.0f}"),
            ("P(revised next step | cell sits in a broken group)", "p_fix", "{:.4f}"),
            ("P(revised next step | all of its groups are legal)", "p_idle", "{:.4f}"),
            ("change in broken-group count at a revising step", "viol_drop", "{:+.2f}"),
            ("P(a peer is revised | a cell is revised)", "p_cond", "{:.4f}"),
            ("  ... same probability if revisions were independent", "p_indep", "{:.4f}"),
            ("revision rate per cell-step", "p_base", "{:.4f}"),
            ("spearman(settling step, candidates left)", "rho_cand", "{:.3f}"),
            ("spearman(settling step, givens among peers)", "rho_peer", "{:.3f}"),
            ("blanks reachable by naked/hidden singles", "singles_frac", "{:.4f}"),
            ("step-1 confidence, cells already at their final value", "conf1_settled", "{:.3f}"),
            ("step-1 confidence, cells revised later", "conf1_moves", "{:.3f}"),
            ("agreement between two restarts at step 1", "same_step1_all", "{:.4f}"),
            ("agreement between two accepted restarts at t=1", "same_final", "{:.4f}")]
    for label, key, fmt in rows:
        out.append(f"| {label} | {fmt.format(m[key])} | {fmt.format(b[key])} |\n")

    out.append(f"\n## The first estimate's errors\n\n"
               f"- mean candidates left in a blank cell the first estimate gets **wrong**: "
               f"{cand_wrong:.2f}\n"
               f"- ... in one it gets **right**: {cand_right:.2f}\n"
               f"- wrong peers per wrong cell: {clust:.2f}, against {base:.2f} if the errors were "
               f"scattered independently\n")

    open(os.path.join(a.dir, "stats.md"), "w").write("".join(out))
    print("".join(out))
    print(f"wrote {a.dir}/stats.md and three figures")


if __name__ == "__main__":
    main()

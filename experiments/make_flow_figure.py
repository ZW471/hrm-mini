"""Generate flow_matching_architecture.svg (render to PNG with cairosvg at 2880 px wide)."""
import math
import os

W, H = 1440, 2100
SANS = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

INK, MUTED, RULE = "#1f2430", "#6b7280", "#c7ccd6"
BLUE, BLUE_F = "#3b6fd4", "#dfe8fa"
PURP, PURP_F = "#7a5cc7", "#ebe3f8"
ORNG, ORNG_F = "#c5583e", "#fbe3dc"
GRN, GRN_F = "#2f8f5b", "#dcf1e5"
GREY, GREY_F = "#9aa3b2", "#f1f3f6"
DARK, DARK_F = "#1f2430", "#f1f3f6"
YEL, YEL_F = "#c9a227", "#fbf1cc"

out = []
def add(s): out.append(s)

def text(x, y, s, size=11.5, fill=INK, anchor="start", weight="normal", mono=False):
    fam = MONO if mono else SANS
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')

def lines(x, y, rows, size=11.5, fill=INK, mono=False, lh=None):
    lh = lh or size * 1.4
    for i, r in enumerate(rows):
        text(x, y + i * lh, r, size, fill, mono=mono)
    return y + len(rows) * lh

def box(x, y, w, h, label, stroke=BLUE, fill=BLUE_F, size=12.5, weight="700", r=7, sub=None, mono=False):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    if sub:
        text(x + w / 2, y + h / 2 - 2, label, size, INK, "middle", weight, mono)
        text(x + w / 2, y + h / 2 + 12, sub, 9.5, MUTED, "middle")
    else:
        text(x + w / 2, y + h / 2 + size * 0.36, label, size, INK, "middle", weight, mono)

def arrow(x1, y1, x2, y2, color=INK, marker="arr", dashed=False, width=1.5):
    d = ' stroke-dasharray="5,4"' if dashed else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d} marker-end="url(#{marker})"/>')

def path(d, color=INK, marker=None, dashed=False, width=1.5):
    m = f' marker-end="url(#{marker})"' if marker else ""
    dd = ' stroke-dasharray="5,4"' if dashed else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dd}{m}/>')

def rule(y):
    add(f'<line x1="32" y1="{y}" x2="{W-32}" y2="{y}" stroke="{RULE}" stroke-width="1"/>')

def plus(cx, cy, color=INK, r=7):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>')
    add(f'<line x1="{cx-4}" y1="{cy}" x2="{cx+4}" y2="{cy}" stroke="{color}" stroke-width="1.5"/>')
    add(f'<line x1="{cx}" y1="{cy-4}" x2="{cx}" y2="{cy+4}" stroke="{color}" stroke-width="1.5"/>')

# --- board glyphs -----------------------------------------------------------------------------
import random
rng = random.Random(3)

def grid(x, y, size, kind, seed=0, label=None, label_dy=14):
    """9x9 mini board. kind: noise | mixed | board | puzzle | velocity"""
    c = size / 9
    r = random.Random(seed)
    add(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
    for i in range(9):
        for j in range(9):
            cx, cy = x + j * c, y + i * c
            if kind == "noise":
                g = int(120 + 120 * r.random())
                add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="rgb({g},{g},{g+8})"/>')
            elif kind == "mixed":
                if r.random() < 0.55:
                    g = int(140 + 100 * r.random())
                    add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="rgb({g},{g},{g+8})"/>')
                else:
                    add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="{GRN_F}"/>')
            elif kind == "board":
                add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="{GRN_F}"/>')
            elif kind == "puzzle":
                if r.random() < 0.31:
                    add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="{GRN}"/>')
            elif kind == "velocity":
                p = r.random()
                col = PURP_F if p < 0.7 else PURP
                add(f'<rect x="{cx}" y="{cy}" width="{c}" height="{c}" fill="{col}"/>')
    for k in (3, 6):
        add(f'<line x1="{x+k*c}" y1="{y}" x2="{x+k*c}" y2="{y+size}" stroke="{INK}" stroke-width="1"/>')
        add(f'<line x1="{x}" y1="{y+k*c}" x2="{x+size}" y2="{y+k*c}" stroke="{INK}" stroke-width="1"/>')
    if label:
        text(x + size / 2, y + size + label_dy, label, 10.5, MUTED, "middle")

# --- header -----------------------------------------------------------------------------------
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">')
add('<defs>')
for name, col in [("arr", INK), ("arrB", BLUE), ("arrP", PURP), ("arrO", ORNG), ("arrG", GREY), ("arrM", MUTED), ("arrGr", GRN)]:
    add(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text(32, 40, "Flow matching on Sudoku: the board is the state — continuous and discrete", 22, INK, weight="700")
text(32, 64, "flow_113m_cfg_1k · flow_113m_cfg_full — one encoder transformer, read three ways: the velocity field, the regression target it is trained on, and the drift", 12.5, MUTED)
text(32, 80, "of the stochastic sampler that solves puzzles. No recurrent state, no solver, no autoregression: the 9×9 board itself is the state. Plus its discrete counterpart, dfm_113m_* / dfm_L16d256_* (13M).", 12.5, MUTED)
rule(96)

# =============================================================================================
# Panel A: the network
# =============================================================================================
yA = 128
text(32, yA, "The network  v(x_t, t | puzzle)", 17, INK, weight="700")
text(32, yA + 20, "SudokuFlowTransformer · experiments/flow_sudoku.py", 11.5, MUTED, mono=True)
y = lines(32, yA + 46, ["h = x_proj(x_t) + cond_embed(puzzle)",
                        "h = h + t_mlp(sinusoid(t))",
                        "v = out_proj(block_16(... block_1(h)))"], mono=True)
y = lines(32, y + 6, ["16 post-norm blocks, 768 wide, 12 heads, qk-norm",
                      "axial 2D RoPE: row index and column index rotate",
                      "separate halves of every head (no 3x3-box prior)",
                      "117.99M params: 113.2M in the blocks (the name),",
                      "4.7M in the timestep MLP · bidirectional over 81 cells"], 11.5, MUTED)
text(32, y + 14, "16 block-forwards per evaluation", 14, INK, weight="700")
text(32, y + 32, "input and output are both 81 x 9 boards", 11.5, MUTED)

# diagram
gx, gy = 500, yA + 62
gs = 72
grid(gx, gy, gs, "mixed", 11, "x_t  [81 x 9]")
cy = gy + gs / 2
arrow(gx + gs, cy, gx + gs + 28, cy)
box(gx + gs + 30, cy - 17, 68, 34, "x_proj", DARK, DARK_F, 11.5, "700")
px = gx + gs + 130
arrow(gx + gs + 98, cy, px - 8, cy)
plus(px, cy)
# blocks
bx0 = px + 26
arrow(px + 8, cy, bx0 - 2, cy, BLUE, "arrB")
bw, bg = 52, 8
shown = ["block 1", "block 2", "block 3", "…", "block 15", "block 16"]
bx = bx0
for i, lab in enumerate(shown):
    if lab == "…":
        text(bx + 14, cy + 5, "…", 16, BLUE, "middle", "700")
        bx += 28 + bg
        continue
    box(bx, cy - 22, bw, 44, "", BLUE, BLUE_F)
    text(bx + bw / 2, cy - 5, "attn", 10, INK, "middle", "700")
    text(bx + bw / 2, cy + 9, "mlp", 10, INK, "middle", "700")
    text(bx + bw / 2, cy + 33, lab.replace("block ", "b"), 9.5, MUTED, "middle")
    if i < len(shown) - 1:
        arrow(bx + bw, cy, bx + bw + bg - 1, cy, BLUE, "arrB", width=1.3)
    bx += bw + bg
bend = bx - bg
add(f'<rect x="{bx0-8}" y="{cy-34}" width="{bend-bx0+16}" height="86" rx="9" fill="none" stroke="{BLUE}" stroke-width="1" stroke-dasharray="4,3"/>')
text((bx0 + bend) / 2, cy + 64, "16 × TransformerBlock, untied; every cell attends to all 81 (axial 2D RoPE inside attn)", 10.5, BLUE, "middle")
arrow(bend, cy, bend + 28, cy)
box(bend + 30, cy - 17, 76, 34, "out_proj", DARK, DARK_F, 11.5, "700")
arrow(bend + 106, cy, bend + 134, cy, PURP, "arrP")
grid(bend + 138, gy, gs, "velocity", 5, "v  [81 x 9]")
text(bend + 138 + gs / 2, gy + gs + 28, "= predicted x_1 − x_0", 10.5, MUTED, "middle")

# conditioning from above: puzzle
pgx, pgy = px - 36 - 60, gy - 78
# puzzle glyph above the plus
grid(px - 26, yA - 14, 52, "puzzle", 21)
text(px - 34, yA + 16, "puzzle (0 = blank)", 10, GRN, "end")
arrow(px, yA + 40, px, yA + 48, GRN, "arrGr")
box(px - 45, yA + 50, 90, 24, "cond_embed", GRN, GRN_F, 10.5, "700")
arrow(px, yA + 74, px, cy - 9, GRN, "arrGr")
text(px + 52, yA + 57, "one embedding per digit; an all-blank puzzle is the unconditional case (no null token)", 10, GRN)

# conditioning from below: t
tby = cy + 74
text(px - 212, tby + 20, "t ∈ [0,1]", 11.5, ORNG, "start", "700", mono=True)
arrow(px - 150, tby + 16, px - 138, tby + 16, ORNG, "arrO")
box(px - 136, tby + 2, 62, 28, "sinusoid", ORNG, ORNG_F, 10.5, "700")
arrow(px - 74, tby + 16, px - 66, tby + 16, ORNG, "arrO")
box(px - 64, tby + 2, 52, 28, "t_mlp", ORNG, ORNG_F, 10.5, "700")
path(f"M {px-12} {tby+16} L {px} {tby+16} L {px} {cy+9}", ORNG, "arrO")
text(px + 12, tby + 20, "added once, to every cell (a DiT-style adaLN variant exists in the code; these runs do not use it)", 10, ORNG)

text(gx, tby + 50, "No z, no z_H, nothing carried between evaluations: the board x_t is the only state, and every evaluation is a fresh read of it.", 10.5, INK, "start")

rule(yA + 246)

# =============================================================================================
# Panel B: training
# =============================================================================================
yB = yA + 280
text(32, yB, "Training: regress the straight path", 17, INK, weight="700")
text(32, yB + 20, "flow_matching_loss · rectified / conditional-OT flow", 11.5, MUTED, mono=True)
y = lines(32, yB + 46, ["x_0 ~ N(0, I)    x_1 = whiten(onehot(solution))",
                        "t ~ U(0,1)       x_t = (1−t)·x_0 + t·x_1",
                        "loss = ‖ v(x_t, t | puzzle) − (x_1 − x_0) ‖²"], mono=True)
y = lines(32, y + 6, ["puzzle dropped 10 % of the time, so the same weights",
                      "also learn the unconditional field (needed for CFG)",
                      "HRM's augmentation group (transpose, band/stack,",
                      "row/col, digit relabel) applied on the GPU per draw",
                      "batch 768 · 83,200 steps = tuned_hrm's 20 epochs"], 11.5, MUTED)
text(32, y + 14, "1 evaluation per training sample", 14, INK, weight="700")
text(32, y + 32, "≈ HRM's 28 block-forwards at 512 wide, per sample", 11.5, MUTED)

# diagram: path from noise to board
lx0, lx1, ly = 560, 1160, yB + 70
grid(lx0 - 80, ly - 32, 64, "noise", 31, "x_0  noise · t = 0")
grid(lx1 + 16, ly - 32, 64, "board", 41, "x_1  solution · t = 1")
add(f'<line x1="{lx0-12}" y1="{ly}" x2="{lx1+12}" y2="{ly}" stroke="{INK}" stroke-width="2"/>')
# ticks
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    tx = lx0 + f * (lx1 - lx0)
    add(f'<line x1="{tx}" y1="{ly-5}" x2="{tx}" y2="{ly+5}" stroke="{INK}" stroke-width="1.2"/>')
    text(tx, ly - 12, f"t = {f:g}", 9.5, MUTED, "middle")
tf = 0.38
tx = lx0 + tf * (lx1 - lx0)
grid(tx - 32, ly + 14, 64, "mixed", 12)
text(tx, ly + 92, "x_t = (1−t)·x_0 + t·x_1", 10.5, ORNG, "middle", mono=True)
add(f'<circle cx="{tx}" cy="{ly}" r="5" fill="{ORNG}"/>')
# target arrow above the line
text((lx0 + lx1) / 2 + 60, ly - 40, "target velocity  u = x_1 − x_0   (the same constant along the whole segment)", 11, INK, "middle", mono=True)
arrow(lx0 + 0.55 * (lx1 - lx0), ly - 26, lx0 + 0.72 * (lx1 - lx0), ly - 26, INK, "arr", width=1.6)

# model box under the path, to the right of x_t
mx, my = tx + 120, ly + 18
box(mx, my, 132, 56, "v(x_t, t | puzzle)", BLUE, BLUE_F, 12, "700", sub="16 blocks, one pass", mono=True)
arrow(tx + 34, ly + 46, mx - 2, ly + 46, INK)
grid(mx - 10, my + 72, 40, "puzzle", 22)
arrow(mx + 10, my + 70, mx + 10, my + 58, GRN, "arrGr")
text(mx + 10, my + 126, "puzzle (10 %: blanked)", 10, GRN, "middle")
# loss
path(f"M {mx+132} {my+28} L {mx+160} {my+28} L {mx+160} {my+80} L {mx+188} {my+80}", PURP, "arrP")
box(mx + 190, my + 60, 156, 40, "MSE(v, x_1 − x_0)", DARK, DARK_F, 11.5, "700", mono=True)
text(mx + 268, my + 116, "AdamATan2 · lr 1e-4 · EMA 0.999 · bf16 · grad-clip 0.5", 10, MUTED, "middle")

rule(yB + 232)

# =============================================================================================
# Panel C: sampling
# =============================================================================================
yC = yB + 264
text(32, yC, "Solving: 64 stochastic Heun steps, givens pinned", 17, INK, weight="700")
text(32, yC + 20, "sde_sample · heun · noise 10 · guidance 2 · clamp_givens", 11.5, MUTED, mono=True)
y = lines(32, yC + 46, ["dx = [ v_cfg + ½σ²(1−t)·(t·v − x) ] dt + σ(1−t) dW",
                        "v_cfg = v(∅) + 2·( v(puzzle) − v(∅) )",
                        "given cells: re-noised to t and re-pinned every step",
                        "board = argmax(x_1) → check all 27 groups"], mono=True)
y = lines(32, y + 6, ["σ(1−t) → 0 as t → 1, so the noise dies out at the end",
                      "σ = 0 is the deterministic ODE: it plateaus at 55 %",
                      "the noise term is the repair mechanism, not a detail",
                      "restarts: a valid board certifies itself, so redraw",
                      "until one verifies (never consults the answer)"], 11.5, MUTED)
text(32, y + 14, "256 evaluations per solve", 14, INK, weight="700")
text(32, y + 32, "64 steps × 2 (Heun) × 2 (CFG) · 4,096 block-forwards", 11.5, MUTED)
text(32, y + 48, "vs HRM: 16 recurrent steps × 28 = 448, at 512 wide (≈20× less)", 11.5, MUTED)

# chain diagram
cx0, cyC = 560, yC + 112
grid(cx0 - 80, cyC - 32, 64, "noise", 51, "x_0 ~ N(0, I)")
sw, sg = 96, 22
steps = ["step 1", "step 2", "step 3", "…", "step 20", "…", "step 64"]
sx = cx0
arrow(cx0 - 14, cyC, cx0 - 2, cyC, INK)
kinds = ["mixed", "mixed", "mixed", None, "board", None, "board"]
seeds = [61, 62, 63, 0, 64, 0, 65]
firstx = None
for i, lab in enumerate(steps):
    if lab == "…":
        text(sx + 10, cyC + 5, "…", 16, INK, "middle", "700")
        sx += 20 + sg
        arrow(sx - sg + 6, cyC, sx - 2, cyC, INK, width=1.3)
        continue
    if firstx is None:
        firstx = sx
    box(sx, cyC - 24, sw, 48, "", BLUE, BLUE_F)
    text(sx + sw / 2, cyC - 7, lab, 11.5, INK, "middle", "700")
    text(sx + sw / 2, cyC + 6, "v(puzzle)  v(∅)", 8.5, MUTED, "middle", mono=True)
    text(sx + sw / 2, cyC + 17, "× 2 (Heun)", 8.5, MUTED, "middle", mono=True)
    # puzzle injected from below
    arrow(sx + sw / 2, cyC + 82, sx + sw / 2, cyC + 26, GREY, "arrG", width=1.3)
    # endpoint estimate glyph above
    k = kinds[i]
    grid(sx + sw / 2 - 22, cyC - 84, 44, k, seeds[i])
    if i < len(steps) - 1:
        arrow(sx + sw, cyC, sx + sw + sg - 1, cyC, INK, width=1.3)
    lastx = sx
    sx += sw + sg
# puzzle rail
add(f'<line x1="{cx0-16}" y1="{cyC+82}" x2="{lastx+sw/2}" y2="{cyC+82}" stroke="{GREY}" stroke-width="1.3"/>')
grid(cx0 - 80, cyC + 56, 52, "puzzle", 23)
text(cx0 - 54, cyC + 122, "puzzle", 10, GRN, "middle")
text(cx0 - 12, cyC + 98, "givens re-noised to t and pinned at every step (replacement inpainting) · CFG costs a second, blank-puzzle evaluation", 10, MUTED)
# endpoint estimate annotation
text(firstx - 8, cyC - 96, "endpoint estimate x + (1−t)·v,", 10, MUTED, "end")
text(firstx - 8, cyC - 84, "decoded: the running answer", 10, MUTED, "end")
text(firstx + sw / 2, cyC - 102, "≈83 % of cells right", 9.5, ORNG, "middle")
text(firstx + sw / 2, cyC - 91, "after 1 evaluation", 9.5, ORNG, "middle")
text(lastx - (sw + sg) - 20 + sw / 2, cyC - 102, "0 broken groups", 9.5, GRN, "middle")
text(lastx - (sw + sg) - 20 + sw / 2, cyC - 91, "by step ~20", 9.5, GRN, "middle")
# output
arrow(lastx + sw, cyC, lastx + sw + 26, cyC, INK)
box(lastx + sw + 28, cyC - 17, 66, 34, "argmax", DARK, DARK_F, 11.5, "700")
arrow(lastx + sw + 94, cyC, lastx + sw + 118, cyC, GRN, "arrGr")
grid(lastx + sw + 122, cyC - 32, 64, "board", 42, "board → verify")
text(lastx + sw + 154, cyC + 60, "fails? redraw x_0", 9.5, MUTED, "middle")
text(lastx + sw + 154, cyC + 72, "and repair again", 9.5, MUTED, "middle")

text(cx0 - 12, cyC + 118, "Guess the whole board, then repair it: a cell inside a broken row / column / box is 130× more likely to be revised on the next step; σ = 0 removes the repair.",
     10.5, INK, "start")

rule(yC + 262)


# =============================================================================================
# Panel D: the discrete variant
# =============================================================================================
yE = yC + 296
text(32, yE, "Discrete variant: tokens instead of a relaxation", 17, INK, weight="700")
text(32, yE + 20, "SudokuDiscreteFlowTransformer · experiments/dfm_sudoku.py · dfm_113m_unif_sc*_cfg_1k", 11.5, MUTED, mono=True)
y = lines(32, yE + 46, ["x_t ∈ {1..9}^81:  each cell = x_1 w.p. t, else a uniform digit",
                        "h = tok_embed(x_t) + cond_embed(puzzle) + t_mlp(sinusoid(t))",
                        "h = h + W_sc · p_prev              # self-conditioning",
                        "p(x_1 | x_t) = softmax(lm_head(blocks(h)))",
                        "loss = cross-entropy on the non-given cells"], mono=True)
y = lines(32, y + 6, ["same 16 blocks, 2D RoPE, CFG, augmentation, budget",
                      "p_prev = the model's own posterior from a no-grad",
                      "pass on the same x_t (zeros half the time; zero-init W_sc)",
                      "sampling: CTMC Euler — a cell jumps to x̂_1 ~ p(x_1|x_t)",
                      "at rate 1/(1−t), plus η random re-noising (detailed balance)"], 11.5, MUTED)
text(32, y + 14, "same 16 block-forwards per evaluation", 14, INK, weight="700")
text(32, y + 32, "+ a no-grad pass for self-conditioning (train) / none extra (sample)", 11.5, MUTED)

# diagram: one sampling step of the discrete chain
dx0, dy = 560, yE + 84
grid(dx0 - 80, dy - 32, 64, "puzzle", 71, "x_t  tokens, givens fixed")
# tokens shown as digit-ish: reuse puzzle glyph but label differently
arrow(dx0 - 14, dy, dx0 + 2, dy, INK)
box(dx0 + 4, dy - 24, 118, 48, "", BLUE, BLUE_F)
text(dx0 + 63, dy - 6, "16 blocks", 11.5, INK, "middle", "700")
text(dx0 + 63, dy + 9, "p(x_1 | x_t, t, puzzle)", 8.5, MUTED, "middle", mono=True)
# self-cond loop from the output posterior back to the input
arrow(dx0 + 122, dy, dx0 + 150, dy, PURP, "arrP")
grid(dx0 + 154, dy - 32, 64, "velocity", 72, "posterior p  [81 × 9]")
path(f"M {dx0+186} {dy+52} L {dx0+186} {dy+70} L {dx0+63} {dy+70} L {dx0+63} {dy+26}", PURP, "arrP", dashed=True)
text(dx0 + 63, dy + 84, "self-conditioning: the previous step's posterior is fed back in through a zero-init projection", 10, PURP, "start")
# sample x1_hat and jump
arrow(dx0 + 218, dy, dx0 + 246, dy, INK)
box(dx0 + 248, dy - 17, 96, 34, "sample x̂_1", DARK, DARK_F, 11, "700")
arrow(dx0 + 344, dy, dx0 + 372, dy, INK)
box(dx0 + 374, dy - 24, 150, 48, "", ORNG, ORNG_F)
text(dx0 + 449, dy - 6, "CTMC jump", 11.5, INK, "middle", "700")
text(dx0 + 449, dy + 9, "x ≠ x̂_1 → x̂_1 w.p. dt/(1−t)", 8.5, MUTED, "middle", mono=True)
text(dx0 + 449, dy + 44, "+ re-noise w.p. η·dt", 9.5, ORNG, "middle")
arrow(dx0 + 524, dy, dx0 + 552, dy, GRN, "arrGr")
grid(dx0 + 556, dy - 32, 64, "puzzle", 73, "x_{t+dt}")
text(dx0 + 556 + 32, dy + 58, "→ next step;", 9.5, MUTED, "middle")
text(dx0 + 556 + 32, dy + 70, "argmax at t = 1", 9.5, MUTED, "middle")
# annotations
text(dx0 - 80, dy + 112, "What changed: the state is hard tokens, so the soft information a continuous x_t carries between steps is gone — unless the model is handed", 10.5, INK, "start")
text(dx0 - 80, dy + 126, "its own posterior. Self-conditioning restores it and is worth 2.7× the accuracy on 1k (24 → 53 → 64 in-training with two refinement passes). Mask prior: worse at every setting.", 10.5, INK, "start")

rule(yE + 258)

# =============================================================================================
# Panel E: results
# =============================================================================================
yD = yE + 292
text(32, yD, "Continuous vs discrete, 1k vs full — same backbone, same budget", 17, INK, weight="700")
text(32, yD + 20, "test_hard exact match, 512 puzzles, EMA weights · n = 3 seeds except the full-data discrete run · 1k arms early-stopped at their peak", 11.5, MUTED, mono=True)

# result blocks (left column)
bx_ = 32
text(bx_, yD + 50, "flow_113m_cfg_1k  (continuous)", 12, ORNG, weight="700", mono=True)
text(bx_, yD + 74, "44.21 ± 2.18", 20, INK, weight="700")
text(bx_, yD + 91, "in-training, 64 steps · best at epoch 4–5, then overfits", 10.5, MUTED)
text(bx_, yD + 106, "50.4 ± 1.2 tuned sampler · 91.3 ± 1.0 with 32 verified restarts", 10.5, MUTED)

text(bx_, yD + 130, "flow_113m_cfg_full  (continuous)", 12, BLUE, weight="700", mono=True)
text(bx_, yD + 154, "74.54 ± 0.49", 20, INK, weight="700")
text(bx_, yD + 171, "in-training · best = final, still climbing · 3.8M puzzles", 10.5, MUTED)

text(bx_, yD + 198, "dfm_113m_unif_sc2_cfg_1k  (discrete)", 12, GRN, weight="700", mono=True)
text(bx_, yD + 222, "84.5 ± 1.2", 20, INK, weight="700")
text(bx_, yD + 239, "full test_hard (20k), tuned sampler (128 steps, η 10, guidance 5) · 86.5 ± 0.8 on the 512 selection subset", 10.5, MUTED)
text(bx_, yD + 254, "best at epoch 3, every seed · 97.7 ± 0.2 @8, 99.3 ± 0.3 @32 verified restarts (512 subset)", 10.5, MUTED)

text(bx_, yD + 281, "dfm_113m_unif_sc2_cfg_full  (discrete)", 12, GRN, weight="700", mono=True)
text(bx_, yD + 305, "86.7 / 91.6 (η 10)", 20, INK, weight="700")
text(bx_, yD + 322, "in-training at step 83,200, full schedule · n = 1 · sampler not yet tuned", 10.5, MUTED)
text(bx_, yD + 337, "reference: tuned_hrm on 1k, 80.65 ± 2.40 (n=6, full test_hard) · the 13M discrete model is in the panel below", 10.5, MUTED)

# mini chart of the curves
chx, chy, chw, chh = 470, yD + 44, 420, 220
steps_k = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 83.2]
c1k = [0, 6.1, 20.4, 30.9, 41.5, 31.8, 30.3, 28.3, 26.2, 25.7, 24.8, 24.9, 24.9]
cfu = [0, 6.4, 20.1, 28.1, 43.9, 56.1, 59.8, 66.1, 69.1, 71.3, 72.9, 73.6, 74.2]
d_unif = [(0,0),(2.5,5.1),(5,13.9),(7.5,20.9),(10,24.0),(12.5,19.0),(15,16.4),(17.5,16.0),(20,15.6),(22.5,14.7),(25,15.8)]
d_sc   = [(0,0),(2.5,4.9),(5,25.2),(7.5,43.4),(10,52.7),(12.5,52.5),(15,37.3),(17.5,33.8),(20,30.9),(22.5,29.9),(25,25.2),(27.5,26.2),(30,27.0)]
d_sc2  = [(0,0),(2.5,4.1),(5,27.5),(7.5,47.1),(10,60.9),(12.5,64.1),(15,50.4),(17.5,36.1),(20,30.9)]
d_full = [(0,0),(5,26.4),(10,52.9),(15,69.1),(20,73.4),(25,75.6),(30,79.1),(35,80.7),(40,81.3)]
XMAX = 40.0
def X(s): return chx + min(s, XMAX) / XMAX * chw
def Y(v): return chy + chh - v / 100 * chh
add(f'<line x1="{chx}" y1="{chy}" x2="{chx}" y2="{chy+chh}" stroke="{RULE}" stroke-width="1"/>')
add(f'<line x1="{chx}" y1="{chy+chh}" x2="{chx+chw}" y2="{chy+chh}" stroke="{RULE}" stroke-width="1"/>')
for v in (25, 50, 75, 100):
    add(f'<line x1="{chx}" y1="{Y(v)}" x2="{chx+chw}" y2="{Y(v)}" stroke="{RULE}" stroke-width="0.6" stroke-dasharray="2,3"/>')
    text(chx - 6, Y(v) + 4, f"{v}", 9.5, MUTED, "end")
for e in (2, 4, 6, 8):
    s = e * 4.16
    add(f'<line x1="{X(s)}" y1="{chy+chh}" x2="{X(s)}" y2="{chy+chh+4}" stroke="{MUTED}" stroke-width="1"/>')
    text(X(s), chy + chh + 15, f"ep {e}", 9.5, MUTED, "middle")
text(chx + chw / 2, chy + chh + 30, "optimizer steps (4,160 per epoch); first 40k of the 83,200-step schedule", 9.5, MUTED, "middle")
text(chx - 6, chy - 8, "test_hard exact match (%), in-training eval at 64 steps", 9.5, MUTED, "start")
add(f'<line x1="{chx}" y1="{Y(80.65)}" x2="{chx+chw}" y2="{Y(80.65)}" stroke="{PURP}" stroke-width="1.2" stroke-dasharray="6,4"/>')
text(chx + chw - 4, Y(80.65) - 5, "tuned_hrm best on 1k, 80.65", 9.5, PURP, "end")
def poly(xs, ys, col, width=2.2, dashed=False):
    pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(xs, ys))
    d = ' stroke-dasharray="5,3"' if dashed else ""
    add(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{width}" stroke-linejoin="round"{d}/>')
poly(steps_k, cfu, BLUE)
poly(steps_k, c1k, ORNG)
poly([a for a,_ in d_unif], [b for _,b in d_unif], GRN, 1.6, dashed=True)
poly([a for a,_ in d_sc], [b for _,b in d_sc], GRN, 1.6)
poly([a for a,_ in d_sc2], [b for _,b in d_sc2], GRN, 2.6)
poly([a for a,_ in d_full], [b for _,b in d_full], GRN, 2.0, dashed=True)

text(X(XMAX) - 4, Y(cfu[7]) + 14, "cont. full", 9.5, BLUE, "end", "700")
text(X(XMAX) - 4, Y(c1k[7]) - 6, "cont. 1k", 9.5, ORNG, "end", "700")
pk = max(d_sc2, key=lambda p: p[1])
add(f'<circle cx="{X(pk[0])}" cy="{Y(pk[1])}" r="4" fill="#ffffff" stroke="{GRN}" stroke-width="1.8"/>')
text(X(pk[0]) + 8, Y(pk[1]) - 4, f"discrete 1k, unif + self-cond ×2 (seed 1): {pk[1]:.0f}", 9.5, GRN, "start", "700")
text(X(10) + 8, Y(52.7) + 12, "unif + self-cond", 9, GRN, "start")
text(X(10) + 8, Y(24.0) + 12, "unif, no self-cond", 9, GRN, "start")
text(X(26), Y(75.6) + 14, "disc. full (86.7 at 83k)", 9, GRN, "start")

# what differs table (right column)
tx0 = 940
text(tx0, yD + 50, "Continuous vs discrete flow", 13, INK, weight="700")
rows = [
    ("state",     "cont: x_t ∈ R^{81×9}, whitened one-hots",      "disc: x_t ∈ {1..9}^81, hard tokens"),
    ("prior",     "cont: N(0, I)",                                 "disc: uniform digits (mask prior: worse)"),
    ("path",      "cont: straight line, x_t = (1−t)x_0 + t x_1",   "disc: per cell, x_1 w.p. t else noise"),
    ("target",    "cont: velocity x_1 − x_0, MSE",                 "disc: p(x_1 | x_t), cross-entropy"),
    ("sampler",   "cont: Heun SDE, g = σ(1−t), 2 NFE/step",        "disc: CTMC Euler, rate 1/(1−t) + η, 1 NFE/step"),
    ("givens",    "cont: re-noised and pinned each step",          "disc: never noised, never in the loss"),
    ("memory",    "cont: soft state carried in x_t itself",        "disc: only via self-conditioning on p_prev"),
    ("on 1k",     "cont: 44 in-train · 50 tuned · 91 @32 restarts", "disc: 67 in-train · 84.5 tuned (20k) · 99.3 @32 restarts"),
]
ty = yD + 74
for k, a, b in rows:
    text(tx0, ty, k, 10.5, MUTED, "start", "700")
    text(tx0 + 60, ty, a, 10.5, BLUE)
    text(tx0 + 60, ty + 13, b, 10.5, GRN)
    ty += 30

rule(yD + 362)

# =============================================================================================
# Panel F: size sweep and the HRM-sized comparison
# =============================================================================================
yF = yD + 396
text(32, yF, "Same size as HRM: the 13M discrete model ties HRM on the full test set, at 3.5× the inference wall clock and 1/22 of the training compute", 17, INK, weight="700")
text(32, yF + 20, "run_dfm_sizes.sh · dfm_L<layers>d<hidden>_unif_sc2t[_lr3]_cfg_1k · outputs/dfm_sizes/README.md · 1k puzzles, batch 768, tuned sampler (128 steps, η 10, guidance 5)", 11.5, MUTED, mono=True)

# --- scatter: params vs best in-training exact match, coloured by depth --------------------------
sx0, sy0, sw_, sh_ = 60, yF + 52, 470, 250
deep = [  # (params M, best in-training %, label, lr)
    (2.50, 69.3, "L12d128", "3e-4"), (3.28, 77.5, "L16d128", "3e-4"), (5.61, 72.3, "L12d192", "3e-4"),
    (7.38, 81.3, "L16d192", "1e-3"), (9.97, 79.9, "L12d256", "3e-4"), (10.92, 85.4, "L24d192", "3e-4"),
    (13.12, 81.3, "L16d256", "3e-4"), (16.26, 82.4, "L20d256", "3e-4"), (22.43, 79.9, "L12d384", "1e-4"),
    (29.51, 79.7, "L16d384", "1e-4"), (39.87, 80.1, "L12d512", "1e-4"), (52.45, 81.8, "L16d512", "1e-4"),
    (117.99, 86.3, "L16d768", "1e-4"),
]
shallow = [
    (1.71, 43.5, "L8d128"), (3.68, 10.2, "L4d256"), (6.83, 63.5, "L8d256"), (11.81, 53.7, "L6d384"),
    (14.70, 35.2, "L4d512"), (15.35, 72.3, "L8d384"), (20.99, 53.3, "L6d512"), (27.28, 64.3, "L8d512"),
]
PMIN, PMAX = 1.5, 150.0
def PX(pm): return sx0 + (math.log10(pm) - math.log10(PMIN)) / (math.log10(PMAX) - math.log10(PMIN)) * sw_
def PY(v): return sy0 + sh_ - v / 100 * sh_
add(f'<line x1="{sx0}" y1="{sy0}" x2="{sx0}" y2="{sy0+sh_}" stroke="{RULE}" stroke-width="1"/>')
add(f'<line x1="{sx0}" y1="{sy0+sh_}" x2="{sx0+sw_}" y2="{sy0+sh_}" stroke="{RULE}" stroke-width="1"/>')
for v in (25, 50, 75, 100):
    add(f'<line x1="{sx0}" y1="{PY(v)}" x2="{sx0+sw_}" y2="{PY(v)}" stroke="{RULE}" stroke-width="0.6" stroke-dasharray="2,3"/>')
    text(sx0 - 6, PY(v) + 4, f"{v}", 9.5, MUTED, "end")
for pm in (2, 5, 10, 20, 50, 100):
    add(f'<line x1="{PX(pm)}" y1="{sy0+sh_}" x2="{PX(pm)}" y2="{sy0+sh_+4}" stroke="{MUTED}" stroke-width="1"/>')
    text(PX(pm), sy0 + sh_ + 15, f"{pm}M", 9.5, MUTED, "middle")
text(sx0 + sw_ / 2, sy0 + sh_ + 30, "parameters (log scale) · each point is one shape at its best LR of {1e-4, 3e-4, 1e-3}", 9.5, MUTED, "middle")
text(sx0 - 6, sy0 - 8, "best in-training exact match (%), 512-puzzle subset, tuned sampler, 1 seed (3-seed means: L16d192, L16d256)", 9.5, MUTED, "start")
# HRM reference: a point at 12.59M / 80.65 with its ± band
add(f'<rect x="{sx0}" y="{PY(83.05)}" width="{sw_}" height="{PY(78.25)-PY(83.05)}" fill="{PURP_F}" opacity="0.6"/>')
add(f'<line x1="{sx0}" y1="{PY(80.65)}" x2="{sx0+sw_}" y2="{PY(80.65)}" stroke="{PURP}" stroke-width="1.2" stroke-dasharray="6,4"/>')
add(f'<rect x="{PX(12.59)-5}" y="{PY(80.65)-5}" width="10" height="10" fill="{PURP}" transform="rotate(45 {PX(12.59)} {PY(80.65)})"/>')
text(PX(12.59) - 8, PY(80.65) + 15, "tuned_hrm", 9.5, PURP, "end", "700")
# shallow (grey) then deep (green)
for pm, v, lab in shallow:
    add(f'<circle cx="{PX(pm)}" cy="{PY(v)}" r="4" fill="{GREY}" stroke="#ffffff" stroke-width="1"/>')
for pm, v, lab in [(3.68, 10.2, "L4d256"), (14.70, 35.2, "L4d512"), (1.71, 43.5, "L8d128")]:
    text(PX(pm) + 6, PY(v) + 4, lab, 8.5, MUTED, "start")
text(PX(27.28) + 6, PY(64.3) - 6, "L8d512", 8.5, MUTED, "start")
text(PX(6.83) + 6, PY(63.5) - 6, "4–8 blocks: any width", 9, GREY, "start", "700")
pts = sorted(deep)
add('<polyline points="' + " ".join(f"{PX(a):.1f},{PY(b):.1f}" for a, b, _, _ in pts) + f'" fill="none" stroke="{GRN}" stroke-width="1.2" opacity="0.5"/>')
for pm, v, lab, lr in deep:
    r = 5.5 if lab in ("L16d256", "L16d192", "L16d768") else 4
    add(f'<circle cx="{PX(pm)}" cy="{PY(v)}" r="{r}" fill="{GRN}" stroke="#ffffff" stroke-width="1"/>')
text(PX(2.5) - 4, PY(69.3) + 14, "L12d128", 8.5, GRN, "start")
text(PX(3.28) - 7, PY(77.5) + 4, "L16d128 3.3M", 8.5, GRN, "end")
text(PX(10.92) + 7, PY(85.4) - 6, "L24d192", 8.5, GRN, "start")
text(PX(117.99) - 8, PY(86.3) - 8, "L16d768 113M", 9, GRN, "end", "700")
text(PX(7.38) - 7, PY(81.3) - 8, "L16d192 7.4M", 9, GRN, "end", "700")
text(PX(13.12) + 7, PY(81.3) + 14, "L16d256 13.1M", 9, GRN, "start", "700")
text(PX(30), PY(93), "12–24 blocks: ~80 % from 10M to 52M", 9, GRN, "start", "700")
# offline 3-seed numbers, in the empty lower-right of the plot
lx, ly_ = PX(36), PY(50)
text(lx, ly_, "full test_hard (20k), 1 sample", 9.5, INK, "start", "700")
text(lx, ly_ + 14, "113M   84.5 ± 1.2  n=3", 9.5, GRN, "start", mono=True)
text(lx, ly_ + 27, "13.1M  81.5 ± 0.5  n=3", 9.5, GRN, "start", "700", mono=True)
text(lx, ly_ + 40, "7.4M   80.8 ± 1.1  n=3", 9.5, GRN, "start", mono=True)
text(lx, ly_ + 53, "13.1M, 1D RoPE 77.5 ± 3.6", 9.5, GRN, "start", mono=True)
text(lx, ly_ + 66, "HRM    80.65 ± 2.40 n=6", 9.5, PURP, "start", "700", mono=True)
cap = sy0 + sh_ + 52
text(32, cap, "Depth is what matters. L4d512 has HRM's 12.58M of transformer weights arranged", 10.5, INK)
text(32, cap + 14, "as HRM arranges them (4 blocks, 512 wide) and gets 35 %; the same count as 16", 10.5, INK)
text(32, cap + 28, "distinct 256-wide blocks gets 81.5 % (full split). Width 256 → 768 is then worth 3 points.", 10.5, INK)
text(32, cap + 42, "Small models need lr 3e-4 (+5 to +20 over 1e-4); 32-block models do not train at 1e-4.", 10.5, INK)

# --- comparison table: HRM vs the 13M DFM -------------------------------------------------------
tx1 = 600
text(tx1, yF + 52, "HRM vs the HRM-sized discrete flow model", 13, INK, weight="700")
hdr_y = yF + 72
text(tx1 + 190, hdr_y, "tuned_hrm", 10.5, PURP, "start", "700", mono=True)
text(tx1 + 420, hdr_y, "dfm_L16d256_unif_sc2t_lr3 (n=3)", 10.5, GRN, "start", "700", mono=True)
rows = [
    ("params",                 "12.59M · 4 blocks × 512, reused",           "13.12M · 16 blocks × 256, untied"),
    ("test_hard 20k, 1 sample","80.65 ± 2.40  (n=6)",                       "81.5 ± 0.5  (81.8 / 81.7 / 81.0)"),
    ("512 selection subset",   "—",                                         "84.1 ± 1.2  (selection bias ≈ 2.6)"),
    ("+ 8 verified restarts",  "— (deterministic)",                         "98.0  (512 subset)"),
    ("block-forwards / solve", "16 segments × 28 = 448",                    "128 steps × 2 CFG × 16 = 4,096"),
    ("inference MACs / cell",  "1.41 G",                                    "3.22 G  (2.3×)"),
    ("wall clock, 20k puzzles","15 s on one H100  (0.75 ms/puzzle)",        "52 s  (2.6 ms/puzzle; 3.5×)"),
    ("train MACs / cell / step","264 M  (28 fwd + BPTT)",                   "50 M  (1 graded + ½·2 no-grad fwd)"),
    ("steps to best",          "46–79k  (epoch 11–19)",                     "15k  (epoch 3.6)"),
    ("train FLOPs to best",    "2.1 × 10¹⁸",                                "9.4 × 10¹⁶  (22× less)"),
    ("wall clock to best",     "≈ 2.9 GPU-h  (28 min × 8 H100 for 83k)",    "10 min on one H100  (0.17 GPU-h)"),
    ("what iterates",          "latents z_L, z_H inside the net",           "the board, in the sampler"),
    ("intermediate states",    "learned; no gradient across the carry",     "sampled from the forward process (teacher-forced)"),
    ("revision / stochastic",  "latent only / no",                          "any cell can jump; η re-noise; restarts / yes"),
    ("Sudoku prior",           "none",                                      "none (2D RoPE row/col; no box)"),
    ("test-time knobs",        "none",                                      "steps, η, guidance (selected on the same 512)"),
]
ty = hdr_y + 20
for k, a, b in rows:
    text(tx1, ty, k, 10, MUTED, "start", "700")
    text(tx1 + 190, ty, a, 10, INK)
    text(tx1 + 420, ty, b, 10, INK)
    ty += 17
text(tx1, ty + 12, "Why it works: it is trained on its own intermediate states (a random-t corrupted board is handed to it, with", 10.5, INK)
text(tx1, ty + 26, "per-cell targets) instead of having to invent a latent trajectory; the state is categorical, revised under", 10.5, INK)
text(tx1, ty + 40, "cross-entropy, with a soft carry-over (self-conditioning: 24 → 86 at 113M); the parameters go into 16 distinct", 10.5, INK)
text(tx1, ty + 54, "blocks rather than 4 reused ones; and failures are far-off legal-looking boards that verified restarts re-roll.", 10.5, INK)
text(tx1, ty + 68, "What it costs: 3.5× HRM's wall clock per puzzle, a sampler to tune, and early stopping at epoch 3–4 (it overfits by 5–7).", 10.5, INK)

add('</svg>')
open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow_matching_architecture.svg"), "w").write("\n".join(out))
print("wrote svg")

"""Generate the cycle-FF figures and render them to PNG.

    uv run --with cairosvg python experiments/make_cff_figure.py

Writes, at the repo root:
  cycle_ff_architectures.svg/.png  -- RT vs RT+cycle FF layer (cff_sft_p7) vs HRM, one pass unrolled
  cycle_ff_timescale.svg/.png      -- the timescale sweep: per-epoch curves and best-vs-period

The numbers are pasted from `experiments/collect_cff_results.py` (2026-09-15) so the script has no
W&B dependency; re-paste them if an arm is re-run.
"""
import os
import statistics

import cairosvg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SANS = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
INK, MUTED, RULE, GRID = "#1f2430", "#6b7280", "#c7ccd6", "#e5e8ee"
BLUE, BLUE_F = "#3b6fd4", "#dbe6fb"
PURP, PURP_F = "#7a5cc7", "#e9e1f8"
ORNG, ORNG_F = "#c5583e", "#fce4dc"
GREY, GREY_F = "#9aa3b2", "#eef0f4"
DARK = "#4b5563"

# ----------------------------------------------------------------------------- data
# test_hard exact match (%), one list per seed, one entry per epoch.
INIT = 69.77  # tuned_rt malachite-saluki best.pt, the checkpoint every *_sft arm resumes from
GRAFT = {
    "rt_sft": [  # no cycle FF layer: the matched control
        [69.3, 68.65, 68.04, 66.86, 65.65, 65.12, 64.15, 63.5],
        [69.36, 69.04, 68.11, 67.1, 66.19, 65.09, 64.41, 63.62],
        [69.49, 68.61, 68.0, 66.84, 66.27, 64.89, 63.92, 63.32]],
    "cff_sft": [  # period 1, post phase
        [70.86, 70.17, 72.92, 72.07, 70.87, 69.18, 68.18, 66.35],
        [70.32, 70.61, 72.28, 72.26, 71.28, 69.21, 67.25, 65.43],
        [70.09, 71.6, 73.27, 73.16, 71.75, 69.72, 67.81, 66.17]],
    "cff_sft_pre": [  # period 1, pre phase
        [70.26, 68.98, 71.26, 72.43, 71.55, 69.88, 67.11, 66.15],
        [70.16, 69.51, 71.32, 72.81, 70.94, 68.92, 67.84, 65.93],
        [70.44, 69.51, 69.04, 71.4, 71.14, 68.36, 67.28, 66.07]],
    "cff_sft_p2": [
        [69.87, 69.23, 68.29, 69.72, 71.42, 70.07, 69.7, 68.19],
        [70.42, 69.22, 68.78, 70.22, 71.86, 70.51, 68.67, 68.23],
        [70.08, 69.21, 68.81, 69.98, 70.82, 68.85, 68.98, 66.44]],
    "cff_sft_p3": [
        [70.45, 69.27, 69.07, 70.68, 71.24, 69.8, 68.23, 66.48],
        [69.9, 69.37, 70.64, 72.35, 71.36, 69.15, 67.89, 67.64],
        [69.75, 69.14, 68.63, 70.29, 71.18, 70.3, 68.96, 67.98]],
    "cff_sft_p7": [
        [69.85, 73.95, 76.92, 74.89, 73.31, 69.71, 67.05, 64.75],
        [69.73, 74.95, 76.77, 74.8, 72.87, 69.99, 66.78, 65.72],
        [70.29, 74.05, 76.19, 74.74, 73.7, 70.22, 67.88, 65.43]],
}
SCRATCH = {
    "tuned_rt": [
        [2.3, 23.89, 60.84, 68.25, 70.68, 70.73, 69.46, 68.74, 67.49, 66.65, 65.85, 64.58, 64.38, 63.72, 62.79, 62.28, 61.66, 61.14, 60.13, 59.18],
        [2.23, 21.74, 55.52, 66.57, 69.39, 70.71, 70.41, 69.18, 67.91, 67.32, 66.39, 64.98, 64.75, 63.89, 63.28, 62.84, 61.68, 61.54, 61.13, 60.08],
        [2.31, 27.9, 62.03, 68.45, 70.51, 70.75, 69.39, 68.28, 66.85, 64.95, 63.74, 62.66, 62.29, 61.83, 61.34, 60.7, 59.3, 58.58, 58.14, 57.81],
        [2.26, 23.3, 60.39, 66.94, 69.03, 69.81, 69.39, 68.26, 67.63, 66.71, 65.8, 64.9, 63.94, 63.28, 62.14, 60.64, 60.12, 58.92, 57.76, 56.84]],
    "cff_block_tied": [
        [2.31, 23.42, 55.93, 64.72, 68.8, 70.0, 69.36, 68.38, 67.01, 65.49, 64.81, 63.11, 62.67, 62.1, 61.2, 60.18, 59.51, 59.22, 58.5, 57.92]],
    "cff_block_tied_p3": [
        [2.31, 23.65, 53.48, 61.19, 65.12, 66.94, 67.74, 66.84, 66.54, 65.59, 63.81, 63.26, 62.7, 61.58, 61.15, 60.58, 60.7, 59.75, 58.64, 58.74],
        [2.32, 23.36, 56.01, 66.79, 70.6, 71.25, 70.34, 69.49, 67.57, 66.48, 65.15, 63.4, 62.96, 62.51, 61.13, 59.94, 59.31, 59.74, 58.46, 57.49],
        [2.31, 22.89, 54.59, 64.85, 69.62, 72.55, 72.7, 71.61, 70.49, 69.1, 66.95, 65.87, 64.62, 63.7, 62.99, 62.34, 61.37, 61.16, 60.45, 58.65]],
    "cff_block_tied_p7": [
        [2.32, 27.32, 57.89, 68.16, 71.75, 72.74, 72.18, 70.7, 69.03, 67.51, 66.05, 64.66, 63.1, 61.82, 60.9, 60.11, 58.97, 58.22, 57.72, 56.94],
        [2.27, 27.76, 60.95, 67.61, 69.5, 71.71, 72.76, 71.58, 70.31, 69.14, 67.79, 66.33, 65.22, 63.6, 62.86, 61.89, 61.71, 60.9, 60.34, 59.56],
        [2.31, 26.15, 57.86, 66.45, 69.72, 69.57, 69.96, 69.22, 67.91, 67.39, 65.45, 64.33, 63.09, 61.83, 61.09, 60.08, 59.59, 58.9, 57.81, 57.82]],
    "tuned_hrm": [
        [0.57, 16.74, 34.08, 56.46, 63.76, 69.18, 73.14, 76.34, 78.52, 78.95, 80.47, 80.98, 80.91, 81.19, 81.0, 80.95, 80.65, 81.27, 81.32, 80.55],
        [1.81, 18.35, 41.32, 64.06, 73.4, 77.42, 78.99, 79.52, 80.23, 80.89, 81.35, 81.56, 81.83, 81.47, 81.39, 81.31, 81.17, 81.22, 80.69, 80.57],
        [0.34, 4.89, 21.86, 54.49, 68.0, 75.04, 78.24, 79.19, 80.01, 80.89, 81.5, 80.68, 80.96, 80.48, 80.21, 79.61, 79.09, 78.23, 78.35, 77.43],
        [1.19, 16.56, 36.23, 61.22, 71.36, 75.57, 77.53, 78.87, 79.44, 79.99, 80.08, 80.3, 80.69, 80.59, 80.19, 80.58, 80.76, 81.15, 80.69, 80.66],
        [1.29, 19.27, 38.32, 59.06, 69.08, 74.3, 77.86, 78.97, 79.93, 80.15, 80.73, 81.22, 81.47, 82.3, 81.84, 81.9, 81.91, 81.74, 82.18, 81.75],
        [8.31, 23.48, 55.35, 65.95, 70.68, 73.1, 73.15, 72.96, 73.46, 73.81, 73.64, 73.79, 74.62, 74.54, 74.87, 74.46, 75.02, 75.23, 75.82, 75.13]],
}

def best(runs):
    b = [max(r) for r in runs]
    return statistics.mean(b), (statistics.stdev(b) if len(b) > 1 else 0.0)

def mean_curve(runs):
    n = min(len(r) for r in runs)
    return [statistics.mean(r[i] for r in runs) for i in range(n)]

def band(runs):
    n = min(len(r) for r in runs)
    return [min(r[i] for r in runs) for i in range(n)], [max(r[i] for r in runs) for i in range(n)]

# ----------------------------------------------------------------------------- svg helpers
class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.out = w, h, []
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{SANS}">')
        self.add("<defs>")
        for mid, col in [("arr", INK), ("arrB", BLUE), ("arrP", PURP), ("arrO", ORNG), ("arrG", GREY), ("arrM", MUTED)]:
            self.add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                     f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
        self.add("</defs>")
        self.add(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    def add(self, s):
        self.out.append(s)

    def text(self, x, y, s, size=11.5, fill=INK, anchor="start", weight="normal", mono=False, italic=False):
        fam = MONO if mono else SANS
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#x27;")
        extra = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.4, marker=None, dash=None):
        m = f' marker-end="url(#{marker})"' if marker else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{m}{d}/>')

    def path(self, d, stroke=INK, width=1.2, marker=None, dash=None, fill="none", opacity=None):
        m = f' marker-end="url(#{marker})"' if marker else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{m}{da}{op}/>')

    def box(self, x, y, w, h, label, stroke, fill, size=13, weight="600"):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size, INK, "middle", weight)

    def rule(self, y):
        self.line(32, y, self.w - 32, y, RULE, 1)

    def write(self, name):
        self.add("</svg>")
        svg = "\n".join(self.out)
        with open(os.path.join(ROOT, name + ".svg"), "w") as f:
            f.write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=os.path.join(ROOT, name + ".png"), output_width=self.w * 2)
        print("wrote", name + ".svg", name + ".png")

# ----------------------------------------------------------------------------- figure 1: architectures
def panel_header(s, y, title, sub, code, notes, score, score_sub):
    s.text(32, y, title, 17, INK, weight="700")
    s.text(32, y + 20, sub, 11.5, MUTED, mono=True)
    for i, c in enumerate(code):
        s.text(32, y + 46 + 16 * i, c, 11.5, INK, mono=True)
    ny = y + 46 + 16 * len(code) + 8
    for i, n in enumerate(notes):
        s.text(32, ny + 16 * i, n, 11.5, MUTED)
    sy = ny + 16 * len(notes) + 12
    s.text(32, sy, score, 20, INK, weight="700")
    s.text(32, sy + 18, score_sub, 11, MUTED)

def x_rail(s, xs, y_rail, y_box_bottom):
    s.line(353, y_rail, xs[-1], y_rail, GREY, 1.4)
    s.text(347, y_rail + 4, "x", 13, MUTED, "end", italic=True)
    for x in xs:
        s.line(x, y_rail, x, y_box_bottom + 2, GREY, 1.2, "arrG")

def readout(s, x, y, src_colour, marker):
    s.line(x, y, x + 38, y, src_colour, 1.4, marker)
    s.box(x + 40, y - 20, 68, 40, "lm_head", DARK, GREY_F, 12)
    s.line(x + 108, y, x + 136, y, INK, 1.4, "arr")
    s.text(x + 140, y + 4, "logits", 11, INK)

UNTIED = [("#fde2cf", "#d97a2b"), ("#fbe7c6", "#c98a1e"), ("#f9ecc1", "#b99512"), ("#fde0d9", "#d9634f"),
          ("#fbd9c9", "#cf6f3a"), ("#f7e3cf", "#b5773f"), ("#fce4dc", "#c5583e")]

def core_row(s, xs, bw, gap, y, label="core", z_labels=True):
    for i, x in enumerate(xs):
        s.box(x, y - 20, bw, 40, label, BLUE, BLUE_F, 13 if bw >= 46 else 11)
        if i < len(xs) - 1:
            s.line(x + bw, y, x + bw + gap - 2, y, BLUE, 1.4, "arrB")
            if z_labels:
                s.text(x + bw + gap / 2, y - 8, "z", 11, BLUE, "middle", italic=True)

def architectures():
    s = SVG(1440, 1280)
    s.text(32, 40, "Four ways to spend a forward pass on Sudoku-Extreme 1k", 22, INK, weight="700")
    s.text(32, 64, "Same fill = shared weights. Blue = fast state, orange/purple = slow state. Vertical grey arrows = the puzzle "
           "embedding x injected every cycle. Panels 1–2 unroll one recurrent step; panel 3 unrolls two, to show the slow clock.", 12.5, MUTED)
    s.rule(86)

    bw, gap, x0 = 60, 58, 405
    core_x = [x0 + i * (bw + gap) for i in range(7)]
    core_c = [x + bw / 2 for x in core_x]
    ctrl, _ = best(GRAFT["rt_sft"])

    # --- panel 1: RT
    y = 234
    panel_header(s, 118, "Recurrent transformer", "tuned_rt · rt@RecurrentTransformer",
                 ["z = core(z + x)     ×7 cycles", "logits = lm_head(z)"],
                 ["core = 4 layers, shared across all 7 cycles", "12.59M params · 28 block-forwards / pass", "one state, one timescale"],
                 "70.50 ± 0.46", "best test_hard, n=4 · peaks ep 6, then overfits to 58.5")
    s.line(353, y, core_x[0] - 2, y, BLUE, 1.4, "arrB")
    s.text(355, y - 8, "z₀", 11.5, BLUE, italic=True)
    core_row(s, core_x, bw, gap, y)
    x_rail(s, core_c, y + 76, y + 20)
    readout(s, core_x[-1] + bw, y, BLUE, "arrB")
    s.path(f"M{core_c[-1]},{y - 20} V{y - 74} H{core_c[0]} V{y - 24}", BLUE, 1.2, "arrB", "5 4")
    s.text(core_c[3], y - 82, "z carried to the next recurrent step (detached)", 10.5, MUTED, "middle")
    s.text(353, 128, "no slow state", 11.5, MUTED, italic=True)
    s.rule(362)

    # --- panel 2: RT + cycle FF layer, period 1 (the original arm: cff_sft_untied)
    y = 510
    u, usd = best([[72.41, 73.61, 73.28, 72.7, 71.32, 70.23, 68.41, 67.11],   # cff_sft_untied, seeds 1-3
                   [71.86, 73.19, 72.79, 72.71, 70.99, 69.94, 67.8, 64.96],
                   [72.32, 73.8, 73.35, 73.51, 71.99, 70.86, 68.74, 66.74]])
    panel_header(s, 394, "RT + cycle FF layer, period 1", "cff_sft_untied · rt_cff@RecurrentTransformerCFF",
                 ["z   = core(z + z_H + x)", "z_H = z_H + g · cff_i(z_H + z)   every cycle", "logits = lm_head(z)"],
                 ["same 4-layer core + 7 untied 1-layer blocks", "27.3M params · 28 + 7 block-forwards / pass",
                  "grafted onto a converged RT; gate g zero-init"],
                 f"{u:.2f} ± {usd:.2f}", f"n=3 · +{u - ctrl:.2f} vs rt_sft control ({ctrl:.2f}): same ckpt, same budget")
    yH = y - 94
    s.line(353, y, core_x[0] - 2, y, BLUE, 1.4, "arrB")
    s.text(355, y - 8, "z₀", 11.5, BLUE, italic=True)
    s.line(353, yH, core_c[0] + 29, yH, ORNG, 1.4, "arrO")
    s.text(355, yH - 8, "z_H = 0", 11.5, ORNG, italic=True)
    core_row(s, core_x, bw, gap, y)
    for i, x in enumerate(core_x):
        cx = x + bw + 1
        fill, stroke = UNTIED[i]
        s.add(f'<rect x="{cx}" y="{yH - 20}" width="56" height="40" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        s.text(cx + 28, yH + 4.5, f"cff{'₁₂₃₄₅₆₇'[i]}", 12.5, INK, "middle", "600")
        s.path(f"M{x + bw},{y} L{cx + 10},{yH + 22}", BLUE, 1.2, "arrB")
        if i < 6:
            s.path(f"M{cx + 46},{yH + 20} L{x + bw + gap - 2},{y - 6}", stroke, 1.2, "arrO")
            s.line(cx + 56, yH, cx + 56 + gap + 2, yH, stroke, 1.3, "arrO")
    x_rail(s, core_c, y + 76, y + 20)
    readout(s, core_x[-1] + bw, y, BLUE, "arrB")
    last = core_x[-1] + bw + 1 + 56
    s.line(last + 2, yH, last + 48, yH, ORNG, 1.3, "arrO", "5 4")
    s.text(last + 52, yH + 4, "z_H carried", 10.5, MUTED)
    s.text(last + 50, yH + 16, "(+ z, as in RT)", 10.5, MUTED)
    s.text(core_c[0] + 59, yH - 32, "7 distinct weight sets · z_H fed back into the core's input on the next cycle", 11, ORNG, italic=True)
    s.text(core_c[3] + 59, y + 96, "cff_period = 1 → the 'slow' state updates every cycle: its own weights and feedback path, not yet a slower clock",
           10.5, MUTED, "middle", italic=True)
    s.rule(638)

    # --- panel 3: RT + cycle FF layer, period 7, two recurrent steps unrolled
    y = 786
    p7, p7sd = best(GRAFT["cff_sft_p7"])
    p1, _ = best(GRAFT["cff_sft"])
    panel_header(s, 670, "RT + cycle FF layer, period 7", "cff_sft_p7 · rt_cff@RecurrentTransformerCFF",
                 ["z_H = z_H + g · cff(z_H + z)   once, before cycle 1", "z   = core(z + z_H + x)        ×7 cycles", "logits = lm_head(z)"],
                 ["same 4-layer core + one tied 1-layer block", "14.7M params · 28 + 1 block-forwards / pass",
                  "grafted onto a converged RT; gate g zero-init"],
                 f"{p7:.2f} ± {p7sd:.2f}", f"n=3 · +{p7 - ctrl:.2f} vs rt_sft control ({ctrl:.2f}) · +{p7 - p1:.2f} vs period 1 tied ({p1:.2f})")
    sw, sg = 40, 14
    sc_x = [405 + i * (sw + sg) for i in range(14)]
    sc_c = [x + sw / 2 for x in sc_x]
    yH = y - 94
    rail_y = yH + 34
    s.line(353, y, sc_x[0] - 2, y, BLUE, 1.4, "arrB")
    s.text(355, y - 8, "z₀", 11.5, BLUE, italic=True)
    core_row(s, sc_x, sw, sg, y, z_labels=False)
    x_rail(s, sc_c, y + 76, y + 20)
    # two applications of the same tied block: before cycle 1 and before cycle 8
    cffA, cffB = 372, sc_x[7] - 40
    s.line(353, yH, cffA - 2, yH, ORNG, 1.4, "arrO")
    s.text(340, yH - 8, "z_H", 11.5, ORNG, "end", italic=True)
    s.path(f"M{sc_x[0] - 14},{y} V{yH + 22}", BLUE, 1.2, "arrB")          # z_0 up into cff (pass 1)
    s.path(f"M{sc_x[6] + sw + 2},{y} L{cffB + 12},{yH + 22}", BLUE, 1.2, "arrB")   # z after cycle 7 into cff (pass 2)
    for k, (cx, cores) in enumerate([(cffA, sc_c[:7]), (cffB, sc_c[7:])]):
        s.box(cx, yH - 20, 56, 40, "cff", ORNG, ORNG_F, 12.5)
        s.path(f"M{cx + 28},{yH + 20} V{rail_y} H{cores[-1]}", ORNG, 1.2)
        for c in cores:
            s.line(c, rail_y, c, y - 24, ORNG, 1.1, "arrO")
    s.line(cffA + 56, yH, cffB - 2, yH, ORNG, 1.3, "arrO")
    s.text(cffA + 64, yH - 8, "z_H carried to the next step (detached)", 10.5, ORNG, italic=True)
    s.text(cffA + 64, rail_y - 5, "z_H steers all 7 core cycles of the step", 10.5, ORNG, italic=True)
    s.text(cffB + 64, rail_y - 5, "same block, same weights", 10.5, ORNG, italic=True)
    s.line(sc_x[6] + sw, y + 22, sc_x[6] + sw + 7, y + 22, "#ffffff", 0)  # no-op spacer
    # readout after each step
    readout(s, sc_x[-1] + sw, y, BLUE, "arrB")
    s.line(cffB + 56, yH, sc_x[-1] + sw + 40, yH, ORNG, 1.3, "arrO", "5 4")
    s.text(sc_x[-1] + sw + 44, yH + 4, "z_H carried", 10.5, MUTED)
    for k, (a, b) in enumerate([(0, 6), (7, 13)]):
        s.path(f"M{sc_x[a]},{y + 78} V{y + 84} H{sc_x[b] + sw} V{y + 78}", MUTED, 1.0)
        s.text((sc_x[a] + sc_x[b] + sw) / 2, y + 98, f"recurrent step {k + 1}: one slow update, then 7 fast cycles", 10.5, MUTED, "middle")
    s.rule(914)
    s.text(sc_c[6] + 27, y + 116, "logits = lm_head(z) are read after every step", 10.5, MUTED, "middle", italic=True)

    # --- panel 4: HRM
    y = 786 + 280
    hrm, hrmsd = best(SCRATCH["tuned_hrm"])
    panel_header(s, 670 + 280, "HRM", "tuned_hrm · hrm@HRM",
                 ["z_L = L(z_L + z_H + x)   ×6", "z_H = H(z_H + z_L)          ×1", "   … repeat for 2 H cycles", "logits = lm_head(z_H)"],
                 ["L = 2 layers, H = 2 layers, both tied", "12.59M params · 24 L + 4 H = 28 block-forwards", "output read from the slow state"],
                 f"{hrm:.2f} ± {hrmsd:.2f}", "best test_hard, n=6 · never fully fits train")
    lw, lg = 46, 18
    L_x = [405 + i * (lw + lg) for i in range(12)]
    L_c = [x + lw / 2 for x in L_x]
    yH = y - 94
    s.line(353, y, L_x[0] - 2, y, BLUE, 1.4, "arrB")
    s.text(355, y - 8, "z_L", 11.5, BLUE, italic=True)
    H1x = L_c[5] + 2
    s.line(353, yH, H1x - 2, yH, PURP, 1.4, "arrP")
    s.text(355, yH - 8, "z_H", 11.5, PURP, italic=True)
    core_row(s, L_x, lw, lg, y, "L", z_labels=False)
    x_rail(s, L_c, y + 76, y + 20)
    H2x = L_x[-1] + lw + 22
    s.box(H1x, yH - 20, 60, 40, "H", PURP, PURP_F)
    s.path(f"M{L_x[5] + lw},{y} L{H1x + 10},{yH + 22}", BLUE, 1.2, "arrB")
    s.box(H2x, yH - 20, 60, 40, "H", PURP, PURP_F)
    s.path(f"M{L_x[-1] + lw + 2},{y} L{H2x + 10},{yH + 22}", BLUE, 1.2, "arrB")
    s.line(H1x + 60, yH, H2x - 2, yH, PURP, 1.3, "arrP")
    rail_y = yH + 34
    s.path(f"M{H1x + 30},{yH + 20} V{rail_y} H{L_c[11] - 31}", PURP, 1.2)
    for cx in L_c[6:]:
        s.line(cx - 31, rail_y, cx - 31, y - 8, PURP, 1.1, "arrP")
    s.text(H1x + 38, rail_y - 6, "z_H steers every L cycle of the next group", 10.5, PURP, italic=True)
    s.path(f"M375,{yH} V{rail_y} H{L_c[5] - 31}", PURP, 1.0, dash="5 4")
    for cx in L_c[:6]:
        s.line(cx - 31, rail_y, cx - 31, y - 8, PURP, 1.0, "arrP", "5 4")
    readout(s, H2x + 60, yH, PURP, "arrP")
    s.line(L_x[-1] + lw + 2, y, L_x[-1] + lw + 48, y, BLUE, 1.3, "arrB", "5 4")
    s.text(L_x[-1] + lw + 52, y + 4, "z_L, z_H carried", 10.5, MUTED)
    for k, (a, b) in enumerate([(0, 5), (6, 11)]):
        s.path(f"M{L_x[a]},{y + 78} V{y + 84} H{L_x[b] + lw} V{y + 78}", MUTED, 1.0)
        s.text((L_x[a] + L_x[b] + lw) / 2, y + 98, f"H cycle {k + 1}: L_cycles = 6, then one H update", 10.5, MUTED, "middle")
    s.rule(1176)

    # --- legend strip
    s.text(32, 1198, "What differs", 12.5, INK, weight="700")
    cols = [
        (160, "slow state", ["none", "yes, updated every core cycle", "yes, updated once per 7 core cycles", "yes, updated once per 6 L cycles"]),
        (470, "weights", ["1 core, tied", "core tied + 7 untied cff blocks", "core tied + 1 tied cff block", "L tied, H tied"]),
        (775, "readout", ["fast state z", "fast state z", "fast state z", "slow state z_H"]),
        (1080, "training", ["from scratch, 20 ep", "graft onto converged RT, 8 ep", "graft onto converged RT, 8 ep", "from scratch, 20 ep"]),
    ]
    rows = [("RT: ", BLUE, 1216), ("cff p1: ", ORNG, 1231), ("cff p7: ", ORNG, 1246), ("HRM: ", PURP, 1261)]
    for x, head, vals in cols:
        s.text(x, 1198, head, 11.5, MUTED, weight="600")
        for (lab, col, yy), v in zip(rows, vals):
            s.text(x, yy, lab, 11, col, weight="600")
            s.text(x + 52, yy, v, 11, INK)
    s.write("cycle_ff_architectures")

# ----------------------------------------------------------------------------- figure 2: timescale sweep
class Axes:
    def __init__(self, s, x, y, w, h, xlim, ylim):
        self.s, self.x, self.y, self.w, self.h, self.xlim, self.ylim = s, x, y, w, h, xlim, ylim

    def px(self, v):
        return self.x + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def frame(self, xticks, yticks, xlabel, ylabel, xfmt=str):
        s = self.s
        self.clip = f"clip{int(self.x)}"
        s.add(f'<clipPath id="{self.clip}"><rect x="{self.x - 6}" y="{self.y - 6}" width="{self.w + 12}" height="{self.h + 12}"/></clipPath>')
        for v in yticks:
            s.line(self.x, self.py(v), self.x + self.w, self.py(v), GRID, 1)
            s.text(self.x - 8, self.py(v) + 4, f"{v:g}", 10.5, MUTED, "end")
        s.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, RULE, 1)
        for v in xticks:
            s.text(self.px(v), self.y + self.h + 16, xfmt(v), 10.5, MUTED, "middle")
        if xlabel:
            s.text(self.x + self.w / 2, self.y + self.h + 34, xlabel, 11, MUTED, "middle")
        s.text(self.x - 36, self.y - 10, ylabel, 10.5, MUTED, "start")

    def legend(self, items, x, y):
        """items: (label, colour, dash, marker) with marker in {None, 'filled', 'hollow'}."""
        for i, (label, colour, dash, marker) in enumerate(items):
            yy = y + i * 15
            if marker:
                fill = colour if marker == "filled" else "#ffffff"
                self.s.add(f'<circle cx="{x + 9}" cy="{yy - 4}" r="4" fill="{fill}" stroke="{colour}" stroke-width="1.6"/>')
            else:
                self.s.line(x, yy - 4, x + 18, yy - 4, colour, 2, dash=dash)
            self.s.text(x + 24, yy, label, 10.5, INK)

    def polyline(self, xs, ys, stroke, width=2, dash=None, opacity=None):
        d = "M" + " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys))
        self.s.add(f'<g clip-path="url(#{self.clip})">')
        self.s.path(d, stroke, width, dash=dash, opacity=opacity)
        self.s.add("</g>")

    def band(self, xs, lo, hi, fill):
        top = " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, hi))
        bot = " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(reversed(xs), reversed(lo)))
        self.s.add(f'<path d="M{top} L{bot} Z" fill="{fill}" stroke="none" opacity="0.5" clip-path="url(#{self.clip})"/>')

    def hline(self, v, stroke, label, dash="2 3", above=True, anchor="start"):
        self.s.line(self.x, self.py(v), self.x + self.w, self.py(v), stroke, 1.1, dash=dash)
        lx = self.x + 4 if anchor == "start" else self.x + self.w - 4
        self.s.text(lx, self.py(v) + (-4 if above else 11), label, 10, stroke, anchor, italic=True)

    def dot(self, x, y, fill, stroke, r=4.5):
        self.s.add(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')

    def errbar(self, x, y, e, stroke):
        if e > 0:
            self.s.line(self.px(x), self.py(y - e), self.px(x), self.py(y + e), stroke, 1.4)
            for v in (y - e, y + e):
                self.s.line(self.px(x) - 4, self.py(v), self.px(x) + 4, self.py(v), stroke, 1.4)

def curve_series(ax, runs, colour, band_fill, label, width=2, dash=None, label_dy=0, label_at=None):
    m = mean_curve(runs)
    xs = list(range(1, len(m) + 1))
    lo, hi = band(runs)
    if band_fill:
        ax.band(xs, lo, hi, band_fill)
    ax.polyline(xs, m, colour, width, dash)
    if label:
        i = (label_at - 1) if label_at else max(range(len(m)), key=lambda k: m[k])
        ax.dot(xs[i], m[i], "#ffffff", colour, 3.5)
        ax.s.text(ax.px(xs[i]) + 7, ax.py(m[i]) + 4 + label_dy, label, 10.5, colour, weight="600")
    return m

def timescale():
    s = SVG(1440, 580)
    s.text(32, 40, "A slower clock is what the cycle FF layer was missing", 22, INK, weight="700")
    s.text(32, 64, "Sudoku-Extreme 1k, test_hard exact match. Lines are the mean over seeds, bands the seed min–max. "
           "Every *_sft arm resumes from the same 69.77 % RT checkpoint behind a zero-init gate, so each starts exactly there.", 12.5, MUTED)
    s.rule(86)
    TOP, AH = 152, 320
    hrm, _ = best(SCRATCH["tuned_hrm"])
    tuned_rt, _ = best(SCRATCH["tuned_rt"])
    ctrl, _ = best(GRAFT["rt_sft"])
    p7, p7sd = best(GRAFT["cff_sft_p7"])
    CFF1 = "#b98a6d"

    # (a) graft curves
    ax = Axes(s, 92, TOP, 380, AH, (1, 8), (62, 82))
    s.text(52, 116, "a · Grafted onto a converged RT, by epoch", 13, INK, weight="700")
    ax.frame(range(1, 9), range(62, 83, 4), "fine-tuning epoch", "test_hard %")
    ax.hline(hrm, PURP, f"tuned_hrm best, from scratch ({hrm:.2f})")
    ax.hline(INIT, GREY, f"RT init ({INIT})", anchor="end")
    curve_series(ax, GRAFT["rt_sft"], MUTED, None, "rt_sft", 1.6, "6 4", label_dy=12, label_at=2)
    curve_series(ax, GRAFT["cff_sft"], CFF1, None, None, 1.6, "2 3")
    curve_series(ax, GRAFT["cff_sft_p7"], ORNG, ORNG_F, "cff_sft_p7", 2.4, label_dy=-6)
    s.text(ax.x + ax.w - 4, ax.py(78.3) + 4, f"{p7:.2f} ± {p7sd:.2f}  (+{p7 - ctrl:.2f} vs control)", 11.5, ORNG, "end", weight="700")
    ax.legend([("cff_sft_p7: slow update once per pass", ORNG, None, None),
               ("cff_sft: slow update every cycle", CFF1, "2 3", None),
               ("rt_sft: no slow state (control)", MUTED, "6 4", None)], ax.x + 8, ax.py(64.6))

    # (b) best vs slow updates per pass
    ax = Axes(s, 590, TOP, 340, AH, (-0.5, 4.5), (66, 80))
    s.text(550, 116, "b · Best test_hard vs how often the slow state updates", 13, INK, weight="700")
    cats = [("none", "no slow state"), ("7 / pass", "period 1"), ("4 / pass", "period 2"), ("3 / pass", "period 3"), ("1 / pass", "period 7")]
    ax.frame(range(5), range(66, 81, 2), "", "best test_hard %", xfmt=lambda i: cats[i][0])
    for i, (_, sub) in enumerate(cats):
        s.text(ax.px(i), ax.y + ax.h + 28, sub, 9.5, MUTED, "middle")
    s.text(ax.x + ax.w / 2, ax.y + ax.h + 46, "slow updates per 7-cycle pass", 11, MUTED, "middle")
    graft_arms = ["rt_sft", "cff_sft_pre", "cff_sft_p2", "cff_sft_p3", "cff_sft_p7"]
    scratch_arms = ["tuned_rt", "cff_block_tied", "cff_block_tied_p3", "cff_block_tied_p7"]
    scratch_x = [0, 1, 3, 4]
    g = [best(GRAFT[a]) for a in graft_arms]
    sc = [best(SCRATCH[a]) for a in scratch_arms]
    ax.polyline(scratch_x, [m for m, _ in sc], MUTED, 1.2, "3 3", opacity=0.6)
    ax.polyline(range(5), [m for m, _ in g], ORNG, 1.4, "3 3", opacity=0.6)
    for x, (m, e) in zip(scratch_x, sc):
        ax.errbar(x, m, e, GREY)
        ax.dot(x, m, "#ffffff", MUTED)
    for x, (m, e) in enumerate(g):
        ax.errbar(x, m, e, ORNG)
        ax.dot(x, m, ORNG if x else MUTED, ORNG if x else MUTED)
        s.text(ax.px(x), ax.py(m + e) - 8 if x else ax.py(m - e) + 16, f"{m:.1f}", 10.5, ORNG if x else MUTED, "middle", weight="600")
    ax.legend([("grafted onto a converged RT  (cff_sft_*, 8 ep)", ORNG, None, "filled"),
               ("trained from scratch  (cff_block_tied_*, 20 ep)", MUTED, None, "hollow")], ax.x + 8, ax.y + 16)

    # (c) from-scratch curves
    ax = Axes(s, 1030, TOP, 370, AH, (1, 20), (50, 85))
    s.text(990, 116, "c · Trained from scratch, by epoch", 13, INK, weight="700")
    ax.frame([1, 5, 10, 15, 20], range(50, 86, 5), "training epoch", "test_hard %")
    curve_series(ax, SCRATCH["tuned_hrm"], PURP, PURP_F, "tuned_hrm", 2, label_at=13)
    curve_series(ax, SCRATCH["tuned_rt"], MUTED, None, None, 1.6, "6 4")
    curve_series(ax, SCRATCH["cff_block_tied_p7"], ORNG, ORNG_F, "cff_block_tied_p7", 2.4, label_dy=-7)
    b7, b7sd = best(SCRATCH["cff_block_tied_p7"])
    ax.legend([("tuned_hrm", PURP, None, None),
               ("cff_block_tied_p7: once per pass", ORNG, None, None),
               ("tuned_rt: no slow state", MUTED, "6 4", None)], ax.px(7), ax.py(57.5))
    s.text(ax.x + ax.w - 4, ax.py(51.5) + 4, f"{b7:.2f} ± {b7sd:.2f}  (+{b7 - tuned_rt:.2f} vs tuned_rt, n=3)", 11.5, ORNG, "end", weight="700")

    s.rule(536)
    s.text(32, 560, "Period 7 (one slow update before the seven core cycles, cff_phase = pre) is the only schedule that separates the two clocks. "
           "Periods 2 and 3 sit at or below period 1: the gain is not monotone in slowness, it switches on at full separation.", 11.5, MUTED)
    s.write("cycle_ff_timescale")

if __name__ == "__main__":
    architectures()
    timescale()

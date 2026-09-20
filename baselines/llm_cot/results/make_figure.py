"""Draw results/llm_sudoku_results.svg from results/sft_scaling.json and the other arms' evals.

Two panels: (A) distilled-CoT SFT accuracy vs. number of teacher traces, one line per
thinking budget; (B) every LLM arm on Sudoku-Extreme test_hard as a bar chart. Plain SVG,
no dependencies, so it renders in a README.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "sft_scaling.json")))

# reference palette: first three categorical slots (validated all-pairs), chart chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

W, H = 1080, 576
out = []
add = out.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family=\'{FONT}\' font-size="13" fill="{INK}">')
add(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
add(f'<text x="24" y="34" font-size="18" font-weight="600">LLM baselines on Sudoku-Extreme (test_hard, exact match, 256 puzzles)</text>')
add(f'<text x="24" y="54" fill="{INK2}">Student: Qwen3.5-4B. Every trace is verified against the gold solution; puzzles come only from the 1k training split.</text>')

# ------------------------------------------------------------------ panel A: scaling lines
ax, ay, aw, ah = 70, 110, 470, 360  # plot box
xs = sorted(int(k) for k in S)
xmin, xmax = 200, 1050
ymax = 20.0
sx = lambda x: ax + (x - xmin) / (xmax - xmin) * aw
sy = lambda y: ay + ah - y / ymax * ah
add(f'<text x="{ax}" y="{ay - 26}" font-size="14" font-weight="600">A. Distilled-CoT SFT: accuracy grows with verified traces</text>')
add(f'<text x="{ax}" y="{ay - 10}" fill="{INK2}" font-size="12">test_hard pass@1 (%), by thinking budget at evaluation</text>')
for t in range(0, 21, 5):
    add(f'<line x1="{ax}" y1="{sy(t):.1f}" x2="{ax + aw}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
    add(f'<text x="{ax - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="12">{t}</text>')
add(f'<line x1="{ax}" y1="{ay + ah}" x2="{ax + aw}" y2="{ay + ah}" stroke="{AXIS}" stroke-width="1"/>')
for x in xs:
    add(f'<text x="{sx(x):.1f}" y="{ay + ah + 18}" text-anchor="middle" fill="{MUTED}" font-size="12">{x}</text>')
add(f'<text x="{ax + aw / 2:.0f}" y="{ay + ah + 40}" text-anchor="middle" fill="{INK2}" font-size="12">verified teacher traces used for SFT (one per puzzle)</text>')
add(f'<text x="{sx(993):.1f}" y="{ay + ah + 32}" text-anchor="middle" fill="{MUTED}" font-size="10">609 full + 384 hinted</text>')
series = [("64k", BLUE), ("32k", ORANGE), ("16k", AQUA)]
for key, col in series:
    pts = [(sx(x), sy(S[str(x)][key] * 100)) for x in xs]
    d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for px, py in pts:
        add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}" stroke="{SURFACE}" stroke-width="2"/>')
    lx, ly = pts[-1]
    v = S[str(xs[-1])][key] * 100
    add(f'<text x="{lx + 10:.1f}" y="{ly + 4:.1f}" font-size="12" fill="{INK}">{v:.1f}% <tspan fill="{INK2}">@{key}</tspan></text>')
# legend (top-left of the plot)
for i, (key, col) in enumerate(series):
    y = ay + 14 + i * 18
    add(f'<line x1="{ax + 12}" y1="{y}" x2="{ax + 34}" y2="{y}" stroke="{col}" stroke-width="2"/>')
    add(f'<text x="{ax + 40}" y="{y + 4}" font-size="12" fill="{INK2}">{key} thinking budget</text>')

# ------------------------------------------------------------------ panel B: all arms, bars
bx, by, bw = 800, 110, 240  # bar box: labels to the left of bx
bars = [
    ("Qwen3.5-4B base, 16k thinking", 0.1),
    ("RLVR from base (GRPO, curriculum), 4B, 32k", 0.0),
    ("Self-distilled CoT SFT, Qwen3.8-27B, 32k", 0.0),
    ("Direct-answer SFT, Qwen3.8-27B", 4.0),
    ("Prompting only, Qwen3.8-27B, 32k thinking", 5.3),
    ("Distilled-CoT SFT, 4B, 32k thinking", S["993"]["32k"] * 100),
    ("Distilled-CoT SFT, 4B, 64k thinking", S["993"]["64k"] * 100),
]
bars.sort(key=lambda t: t[1])
add(f'<text x="{bx - 230}" y="{by - 26}" font-size="14" font-weight="600">B. Every LLM arm, best reported budget</text>')
add(f'<text x="{bx - 230}" y="{by - 10}" fill="{INK2}" font-size="12">test_hard pass@1 (%)</text>')
bmax = 20.0
bsx = lambda v: bx + v / bmax * bw
slot, thick = 44, 22
for t in (0, 5, 10, 15, 20):
    add(f'<line x1="{bsx(t):.1f}" y1="{by}" x2="{bsx(t):.1f}" y2="{by + slot * len(bars)}" stroke="{GRID}" stroke-width="1"/>')
    add(f'<text x="{bsx(t):.1f}" y="{by + slot * len(bars) + 16}" text-anchor="middle" fill="{MUTED}" font-size="12">{t}</text>')
add(f'<line x1="{bx}" y1="{by}" x2="{bx}" y2="{by + slot * len(bars)}" stroke="{AXIS}" stroke-width="1"/>')
for i, (label, v) in enumerate(bars):
    y = by + i * slot + (slot - thick) / 2
    x1 = bsx(v)
    add(f'<text x="{bx - 10}" y="{y + thick / 2 + 4:.1f}" text-anchor="end" font-size="12" fill="{INK2}">{label}</text>')
    if v > 0:
        r = min(4, (x1 - bx) / 2)
        # square at the baseline, 4px rounded data-end
        add(f'<path d="M{bx},{y} H{x1 - r:.1f} a{r},{r} 0 0 1 {r},{r} V{y + thick - r:.1f} a{r},{r} 0 0 1 -{r},{r} H{bx} Z" fill="{BLUE}"/>')
    add(f'<text x="{x1 + 6:.1f}" y="{y + thick / 2 + 4:.1f}" font-size="12" fill="{INK}">{v:.1f}%</text>')
add(f'<text x="24" y="{H - 46}" fill="{INK2}" font-size="11">Teachers for the 4B student: DeepSeek V4 Pro (solves 35% of training puzzles per attempt; 522 traces after retries and repair) and '
    f'Claude Opus 5 on the unsolved tail (87 traces),</text>')
add(f'<text x="24" y="{H - 30}" fill="{INK2}" font-size="11">plus 384 DeepSeek traces of the remaining puzzles with cells revealed down to 40 blanks. '
    f'The 27B numbers are from earlier arms of this study (test_hard subsets of 1024 / 512).</text>')
add(f'<text x="24" y="{H - 14}" fill="{INK2}" font-size="11">RLVR from the distilled checkpoint is in progress and not shown.</text>')
add('</svg>')
open(os.path.join(HERE, "llm_sudoku_results.svg"), "w").write("\n".join(out))
print("wrote", os.path.join(HERE, "llm_sudoku_results.svg"))

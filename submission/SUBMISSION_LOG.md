# Submission log

Target repo: https://github.com/cl-agi/HRM-Results (local clone: `../../HRM-Results`, branch `main`).
Each push copies this whole folder to `figure/<batch>/` there and links the figures from `draft.md`.

## 2026-09-21 — `figure/zhiyu_results_20260921/` (commit c270807)

| local file | pushed to | linked from draft.md |
|---|---|---|
| `scaling_tuned/image.png` | `figure/zhiyu_results_20260921/scaling_tuned/image.png` | S8 · Dataset scaling (Comment 1.5, anchor `#s8-dataset`) |
| `memorization_test/fig_copy_train.png` | `.../memorization_test/fig_copy_train.png` | new B19 (Comment 1.5, anchor `#b19`) |
| `llm_cot/llm_sudoku_results.svg` | `.../llm_cot/llm_sudoku_results.svg` + a rendered `.png` (not in this folder) | S10-c (Comment 1.6h, anchor `#s10`) |
| `flow_matching/6d7d096e85a76175ac1640a5f70a963e.png` | `.../flow_matching/6d7d096e85a76175ac1640a5f70a963e.png` | S16-a (Comment 4.3, anchor `#s16`) |
| `cycle_ff+rt/cycle_ff_architectures.{png,svg}`, `CYCLE_FF.md`, `CYCLE_FF_EXPLAINED.md` | `.../cycle_ff+rt/` | new B20 (Comment 4.2, anchor `#b20`) |

Also edited the "对应 subfigures / 缺项" lines of Comments 1.4, 1.5, 1.6h, 4.2, 4.3. `TODO.md` there was not updated.

If a file here changes: re-copy it to the same path in the clone, re-render the LLM PNG if the SVG changed
(`python -c "import cairosvg; cairosvg.svg2png(url='llm_sudoku_results.svg', write_to='llm_sudoku_results.png', output_width=2400)"`),
update the caption in `draft.md` if the numbers moved, and add a new row/section here.

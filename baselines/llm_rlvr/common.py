"""Shared pieces of the RLVR Sudoku baseline: puzzles, the hint curriculum, prompts, rewards.

The data is the same 1000 `sudoku-extreme-1k` training puzzles the from-scratch models see, with
the same band/stack/digit augmentation (`llm_reason.rollout.augmented_rows`). The one addition
is the *hint curriculum*: a training puzzle may have cells revealed from its solution until only
`n_empty` blanks remain, which makes it easier without importing any puzzle the other arms never
saw. Extreme puzzles have 17-27 clues (54-64 blanks); left with 8 blanks a small model succeeds
often enough for GRPO to have a gradient, and `train.py` raises `n_empty` back to the full puzzle
as the success rate allows. Evaluation is always the raw `test_hard` rows.

The prompt writes the grid as nine rows of nine digits (`INSTRUCTION_ROWS`) rather than the
81-digit line the `llm_reason` baseline used: a 4B model mis-transcribes an 81-digit line more
often than not, which is a tokenisation tax, not a reasoning result. `parse_grid` accepts both
answer layouts, and the metric is unchanged.
"""

from __future__ import annotations

import re

import numpy as np

from baselines.llm_reason.rollout import augmented_rows
from baselines.llm_sft.data import GRID, read_split
from baselines.llm_sft.eval import is_legal_solution
from baselines.llm_sft.prompt_eval import INSTRUCTION, parse_grid

THINK_END = "</think>"

INSTRUCTION_ROWS = (
    "Solve this Sudoku puzzle.\n\n"
    "The grid is given as nine rows of nine digits, where 0 marks an empty cell:\n"
    "{puzzle}\n\n"
    "Work it out, then give your final answer as the completed grid in the same layout: "
    "nine lines of nine digits, with no spaces or separators."
)


def as_rows(question: str) -> str:
    return "\n".join(question[i : i + 9] for i in range(0, GRID, 9))


def add_hints(question: str, answer: str, n_hints: int, rng: np.random.Generator) -> str:
    """Reveal `n_hints` empty cells of `question` from `answer` (fewer if the puzzle has fewer)."""
    if n_hints <= 0:
        return question
    empty = [i for i, c in enumerate(question) if c == "0"]
    reveal = rng.choice(empty, size=min(n_hints, len(empty)), replace=False)
    q = list(question)
    for i in reveal:
        q[i] = answer[i]
    return "".join(q)


def hinted_puzzle(question: str, answer: str, n_hints: int, seed: int, index: int) -> str:
    """`add_hints` addressed by (seed, index, n_hints) so every rank builds the same prompt."""
    rng = np.random.default_rng([seed, index, n_hints])
    return add_hints(question, answer, n_hints, rng)


def puzzle_with_empties(question: str, answer: str, n_empty: int, seed: int, index: int) -> str:
    """Reveal cells until at most `n_empty` blanks remain (the curriculum's difficulty knob)."""
    return hinted_puzzle(question, answer, question.count("0") - n_empty, seed, index)


def load_puzzles(data_dir: str, split: str) -> list[tuple[str, str]]:
    return read_split(data_dir, split)


def make_prompt(question: str, layout: str = "rows") -> list[dict]:
    """Chat-format prompt; the trainer / vLLM apply the model's own template (thinking on)."""
    if layout == "rows":
        content = INSTRUCTION_ROWS.format(puzzle=as_rows(question))
    elif layout == "line":
        content = INSTRUCTION.format(puzzle=question)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return [{"role": "user", "content": content}]


def eval_rows(data_dir: str, n: int, layout: str = "rows") -> list[dict]:
    """The first `n` raw `test_hard` puzzles -- the fixed subset the reasoning baseline reports on."""
    rows = load_puzzles(data_dir, "test_hard")[:n]
    return [{"prompt": make_prompt(q, layout), "question": q, "answer": a, "empties": q.count("0"), "index": i}
            for i, (q, a) in enumerate(rows)]


def train_rows(data_dir: str, seed: int, indices: range, n_empty: int, layout: str = "rows") -> list[dict]:
    """Augmented training puzzles cut down to `n_empty` blanks (for the diagnostic sweep)."""
    rows = load_puzzles(data_dir, "train")
    out = []
    for idx, q, a in augmented_rows(rows, indices, seed):
        hq = puzzle_with_empties(q, a, n_empty, seed, idx)
        out.append({"prompt": make_prompt(hq, layout), "question": hq, "answer": a,
                    "empties": hq.count("0"), "index": idx})
    return out


# ----------------------------------------------------------------------------------------- reward

def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") or "" for m in completion)


def grade(text: str, question: str, answer: str) -> dict:
    """Score one completion. `closed`: the model ended its reasoning by itself (the chat template
    opens `<think>`, so an answer only exists after `</think>`)."""
    closed = THINK_END in text
    grid = parse_grid(text, reasoning=True)
    gold = np.array([int(c) for c in answer])
    given = np.array([int(c) for c in question])
    if grid is None:
        return {"closed": closed, "parsed": False, "exact": False, "legal": False,
                "clues_kept": 0.0, "cell_acc": 0.0}
    clue = given > 0
    return {
        "closed": closed,
        "parsed": True,
        "exact": bool((grid == gold).all()),
        "legal": bool(is_legal_solution(grid[None])[0]),
        "clues_kept": float((grid[clue] == given[clue]).mean()),
        "cell_acc": float((grid == gold).mean()),
    }


def reward_exact(completions, question, answer, **kwargs) -> list[float]:
    """1 if the 81-digit grid after `</think>` is the solution, else 0. The verifiable reward."""
    return [float(grade(completion_text(c), q, a)["exact"])
            for c, q, a in zip(completions, question, answer)]


def reward_format(completions, question, answer, **kwargs) -> list[float]:
    """1 if the model closed its reasoning and wrote a parseable grid that keeps the clues.

    A small-weight shaping term: it is what a wrong-but-well-formed answer earns over a
    truncated trace, so "finish and answer" is rewarded before "be right" is reachable.
    """
    out = []
    for c, q, a in zip(completions, question, answer):
        g = grade(completion_text(c), q, a)
        out.append(float(g["parsed"] and g["clues_kept"] == 1.0))
    return out


def summarize(grades: list[dict], n_tokens: list[int] | None = None) -> dict:
    """Aggregate `grade` outputs the way `llm_reason.rollout.merge` reports them."""
    n = len(grades)
    closed = np.array([g["closed"] for g in grades])
    exact = np.array([g["exact"] for g in grades])
    m = {
        "n": n,
        "exact_match": float(exact.mean()),
        "legal_grid": float(np.mean([g["legal"] for g in grades])),
        "parse_rate": float(np.mean([g["parsed"] for g in grades])),
        "closed_rate": float(closed.mean()),
        "cell_accuracy": float(np.mean([g["cell_acc"] for g in grades])),
        "exact_match_when_closed": float(exact[closed].mean()) if closed.any() else 0.0,
    }
    if n_tokens is not None:
        t = np.asarray(n_tokens)
        m.update(tokens_mean=float(t.mean()), tokens_p50=float(np.median(t)), tokens_p90=float(np.percentile(t, 90)))
    return m

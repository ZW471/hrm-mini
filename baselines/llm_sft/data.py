"""Data plumbing for the fine-tuned-LLM Sudoku baseline.

A puzzle is encoded as a fixed-length token sequence

    [ instruction | 81 puzzle digits | SEP | 81 solution digits | EOS ]

with **exactly one token per cell**. That property is what makes this baseline comparable to
the from-scratch models in `arch/`: per-cell accuracy and 81-cell exact match then mean the
same thing they do in `train.py`. `build_codec` asserts it instead of trusting the tokenizer.

Note that we build token ids directly rather than tokenizing a digit string. Byte-level BPE
would otherwise merge digit runs ("769341" -> two or three tokens), which breaks the cell
alignment. Qwen's pre-tokenizer already splits numbers into single digits, so for Qwen models
the sequence we build is also the one the tokenizer would produce -- i.e. in-distribution.

The data is the repo's, not a copy of it: `read_split` reads the same
`downloaded-datasets/sudoku-extreme-1k` CSVs that `dataset.sudoku.create_dataloader` loads (same
1000 rows, same order), and augmentation calls `dataset.sudoku.shuffle_sudoku` itself, so the
band/stack/digit transform is literally the same code the HRM and transformer baselines use.
The one deliberate difference is bookkeeping: `shuffle_sudoku` draws from the global numpy RNG,
which the repo seeds per dataloader worker, while we seed it per sample index so that a given
(seed, index) reproduces regardless of world size or worker count. Same distribution of
augmentations, reproducible addressing.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from dataset.sudoku import shuffle_sudoku

GRID = 81

# A base model has no chat template, so the "system prompt" is just a text prefix. It states the
# encoding as well as the task, because the 81-digit row-major format is the part the model
# cannot guess.
DEFAULT_SYSTEM_PROMPT = (
    "Please solve this Sudoku task. The puzzle below is written as 81 digits in row-major "
    "order, where 0 marks an empty cell. Reply with the 81 digits of the completed grid.\n"
)


def read_split(data_dir: str, split: str) -> list[tuple[str, str]]:
    """Read a `sudoku-extreme` CSV into `(question, answer)` pairs, blanks written as '0'.

    Same rows in the same order as `dataset.sudoku.create_dataloader` gets from `load_dataset`.
    """
    path = os.path.join(data_dir, f"{split}.csv")
    with open(path, newline="") as f:
        return [(r["question"].replace(".", "0"), r["answer"]) for r in csv.DictReader(f)]


@dataclass
class Codec:
    """Maps between 81-character puzzle strings and model token ids."""

    digit_ids: list[int]  # token id of '0' .. '9'
    sep_ids: list[int]
    eos_id: int
    prefix_ids: list[int] = field(default_factory=list)  # the instruction
    system_prompt: str = ""

    @property
    def prompt_len(self) -> int:
        """Tokens before the first solution cell, for a zero-shot prompt."""
        return len(self.prefix_ids) + GRID + len(self.sep_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + GRID + 1  # + EOS

    @property
    def answer_ids(self) -> list[int]:
        """Token ids a solution cell may take: digits 1-9 (0 means 'blank', never an answer)."""
        return self.digit_ids[1:]

    def grid(self, digits: str) -> list[int]:
        return [self.digit_ids[int(c)] for c in digits]

    def answer(self, solution: str) -> list[int]:
        return self.grid(solution) + [self.eos_id]

    def prompt(self, question: str, demos: Sequence[tuple[str, str]] = ()) -> list[int]:
        """Instruction, then any in-context demonstrations, then the puzzle to solve."""
        ids = list(self.prefix_ids)
        for demo_q, demo_a in demos:
            ids += self.grid(demo_q) + self.sep_ids + self.answer(demo_a)
        return ids + self.grid(question) + self.sep_ids

    def example(self, question: str, solution: str) -> dict[str, list[int]]:
        """A training example with the loss masked off everything but the solution."""
        prompt = self.prompt(question)
        answer = self.answer(solution)
        return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}

    def cells(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode generated token ids to digits; anything that is not a digit becomes -1."""
        lut = np.full(max(self.digit_ids) + 1, -1, dtype=np.int64)
        for digit, tid in enumerate(self.digit_ids):
            lut[tid] = digit
        token_ids = np.asarray(token_ids)
        return np.where(token_ids < len(lut), lut[np.clip(token_ids, 0, len(lut) - 1)], -1)


def build_codec(tokenizer, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Codec:
    digit_ids = []
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"{tokenizer.name_or_path} encodes digit {d!r} as {len(ids)} tokens. This baseline "
                "needs one token per cell so its metrics line up with the from-scratch models. "
                "Use a tokenizer that splits digits individually (e.g. Qwen)."
            )
        digit_ids.append(ids[0])

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"{tokenizer.name_or_path} has no eos_token_id")

    return Codec(
        digit_ids=digit_ids,
        sep_ids=tokenizer.encode("\n", add_special_tokens=False),
        eos_id=eos_id,
        prefix_ids=tokenizer.encode(system_prompt, add_special_tokens=False) if system_prompt else [],
        system_prompt=system_prompt,
    )


def targets(rows: Iterable[tuple[str, str]]) -> np.ndarray:
    """`[n, 81]` array of ground-truth digits."""
    return np.array([[int(c) for c in answer] for _, answer in rows], dtype=np.int64)


class SudokuSFTDataset(torch.utils.data.Dataset):
    """`num_samples` augmented views of `rows`, addressed by index so it shards under DDP.

    Augmentation is `dataset.sudoku.shuffle_sudoku` -- the repo's own transform -- seeded from
    the index rather than from worker state, so a given (seed, index) always yields the same
    puzzle regardless of world size or dataloader workers.
    """

    def __init__(
        self,
        rows: list[tuple[str, str]],
        codec: Codec,
        num_samples: int,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.rows = rows
        self.codec = codec
        self.num_samples = num_samples
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        question, answer = self.rows[idx % len(self.rows)]
        if self.augment:
            np.random.seed((self.seed * 1_000_003 + idx) % (2**31 - 1))
            board = np.frombuffer(question.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            solution = np.frombuffer(answer.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            board, solution = shuffle_sudoku(board, solution)
            question = "".join(map(str, board.flatten().tolist()))
            answer = "".join(map(str, solution.flatten().tolist()))
        return self.codec.example(question, answer)


def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """Every sequence has the same length, so this is a stack -- no padding, no attention holes."""
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

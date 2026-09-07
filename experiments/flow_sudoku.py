"""Pixel-space flow matching that generates solved Sudoku boards.

The question this asks: can a plain (bidirectional) encoder transformer, trained only as a
continuous flow-matching velocity field, generate boards that actually satisfy all 27 Sudoku
constraints? No autoregressive factorization, no solver, no conditioning -- pure unconditional
generation of a 9x9 "image".

Pixel space
-----------
A board is treated as a 9x9 image with `in_channels` channels per cell, never as tokens:
  * `--repr onehot` (default): 9 channels holding the one-hot digit, whitened to zero mean and
    unit variance so the data distribution is scaled like the N(0, I) prior. Decoding is argmax.
  * `--repr scalar`: 1 channel holding the digit itself, whitened the same way. Decoding is
    un-whitening and rounding.
The flow is the straight (rectified / conditional-OT) path in that continuous space:
    x_t = (1 - t) * x_0 + t * x_1,   x_0 ~ N(0, I),   target velocity  u = x_1 - x_0,
and the model regresses u. Sampling is Euler integration from t=0 to t=1.

Data
----
The solved boards of sudoku-extreme (3.8M of them, all distinct), with HRM's exact augmentation
group -- transpose, band/stack and within-band row/column permutations, digit relabelling -- but
applied on the GPU (`augment_boards`) rather than per sample in a dataloader worker. The CPU path
capped out around 65 batches/s, which is the wrong bottleneck when the batch is large. Only the
*answer* side is used: the puzzle is thrown away, so the model sees solved boards drawn from an
effectively unlimited orbit. `--dataset-name` still accepts the 1k subset for comparison.

Backbone
--------
`arch.layers.Transformer` (the same encoder-only stack `arch/mae.py` uses) plus a sinusoidal
timestep embedding added to every cell.

Positions are 2D by default (`--rope-2d`): the board is a 9x9 grid, not a length-81 sequence, so
the head dim is split in half and rotated by the row index and the column index separately. With
the flattened 1D RoPE (`--no-rope-2d`) row neighbours sit 1 apart while column neighbours sit 9
apart, and column constraints end up markedly harder to satisfy than row constraints.

Usage
-----
    uv run torchrun --nproc-per-node 8 experiments/flow_sudoku.py
    uv run python experiments/flow_sudoku.py --train-steps 2000 --eval-interval 500   # quick smoke test
"""

from typing import Any, Optional
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
import wandb
import coolname

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adam_atan2 import AdamATan2
from arch.layers import (CastedLinear, CastedScaledEmbedding, MLP, Transformer,
                         TransformerConfig, trunc_normal_init_)
from dataset.sudoku import create_dataloader

WANDB_PROJECT = "sudoku-flow-matching"

# [Model]
def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal embedding of t in [0, 1]. `t`: [batch] -> [batch, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class RotaryEmbedding2D(nn.Module):
    """Axial 2D RoPE over an H x W grid.

    The head dim is split in half: the first half is rotated by the row index, the second by the
    column index. The returned (cos, sin) has exactly the same [seq_len, head_dim] layout as
    `arch.layers.RotaryEmbedding`, so this drops straight into `arch.layers.Transformer` -- which
    calls `self.rotary_emb()` and passes the result down to every `Attention` unchanged.
    """
    def __init__(self, dim: int, height: int, width: int, base: float):
        super().__init__()
        assert dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
        half = dim // 2  # dims allotted to each axis
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

        rows = torch.arange(height, dtype=torch.float32).repeat_interleave(width)
        cols = torch.arange(width, dtype=torch.float32).repeat(height)
        freqs = torch.cat([torch.outer(rows, inv_freq), torch.outer(cols, inv_freq)], dim=-1)

        emb = torch.cat((freqs, freqs), dim=-1)  # rotate_half pairs dim i with dim i + dim/2
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class NoRotaryEmbedding(nn.Module):
    """cos=1, sin=0, so `apply_rotary_pos_emb` is the identity.

    Lets the stock `Attention` run with no rotation at all, which is what `--pos-embed learned`
    needs: position then enters only once, as a vector added to the input.
    """
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.cos_cached = nn.Buffer(torch.ones(seq_len, dim), persistent=False)
        self.sin_cached = nn.Buffer(torch.zeros(seq_len, dim), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class FlowTransformerConfig(TransformerConfig):
    in_channels: int
    forward_dtype: str

    # "rope2d": axial RoPE over the 9x9 grid | "rope1d": stock RoPE over the flattened 81
    # positions | "learned": a free [seq_len, hidden] parameter added to the input, no rotation
    pos_embed: str = "rope2d"
    grid_height: int = 9
    grid_width: int = 9
    conditional: bool = False

class SudokuFlowTransformer(nn.Module):
    """Encoder-only transformer velocity field v_theta(x_t, t) over the 81 cells of a board.

    Input and output live in pixel space (`in_channels` per cell); the only extra machinery over
    `arch/mae.py` is a sinusoidal timestep embedding broadcast onto every cell.
    """
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = FlowTransformerConfig(**config_dict)
        self.config = config
        self.dtype = getattr(torch, config.forward_dtype)

        # Backbone
        self.core = Transformer(config)
        self.pos_embed = None
        if config.pos_embed == "rope2d":
            # Same interface, grid-aware positions
            self.core.rotary_emb = RotaryEmbedding2D(config.head_dim, config.grid_height,
                                                     config.grid_width, base=config.rope_theta)
        elif config.pos_embed == "learned":
            self.core.rotary_emb = NoRotaryEmbedding(config.head_dim, config.seq_len)
            self.pos_embed = nn.Parameter(
                trunc_normal_init_(torch.empty(config.seq_len, config.hidden_size), std=1.0))
        elif config.pos_embed != "rope1d":
            raise ValueError(f"unknown pos_embed: {config.pos_embed}")
        # I/O
        self.x_proj = CastedLinear(config.in_channels, config.hidden_size, bias=True)
        self.out_proj = CastedLinear(config.hidden_size, config.in_channels, bias=True)
        # Timestep conditioning
        self.t_mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)
        # Optional conditioning on a given puzzle: one embedding per cell value, 0 = blank.
        # Nothing here knows anything about Sudoku -- it is just "some cells are pinned".
        self.cond_embed = (CastedScaledEmbedding(10, config.hidden_size, cast_to=self.dtype)
                           if config.conditional else None)

    def forward(self, x_t: Tensor, t: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        # x_t: [batch, seq_len, in_channels], t: [batch], cond: [batch, seq_len] puzzle digits
        h = self.x_proj(x_t.to(self.dtype))
        if self.pos_embed is not None:
            h = h + self.pos_embed.to(self.dtype)
        if self.cond_embed is not None:
            h = h + self.cond_embed(cond)
        h = h + self.t_mlp(timestep_embedding(t, self.config.hidden_size, self.config.rope_theta).to(self.dtype))[:, None, :]
        return self.out_proj(self.core(h))

# [Data: solutions in memory, augmented on GPU]
def load_solutions(dataset_dir: str, split: str = "train") -> np.ndarray:
    """All solved boards from a sudoku-extreme CSV as [N, 81] uint8 digits, cached as .npy.

    Only the answer column is used -- the puzzle is irrelevant for unconditional generation.
    """
    csv_path = os.path.join(dataset_dir, f"{split}.csv")
    cache_path = os.path.join(dataset_dir, f"{split}_solutions.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    import pandas as pd
    # dtype=str matters: pandas otherwise infers a numeric dtype for some rows of this column
    answers = pd.read_csv(csv_path, usecols=["answer"], dtype={"answer": str})["answer"].dropna()
    answers = answers[answers.str.len() == 81].to_numpy()
    boards = (np.frombuffer("".join(answers).encode(), dtype=np.uint8) - ord("0")).reshape(-1, 81)
    np.save(cache_path, boards)
    return boards

def load_pairs(dataset_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(puzzles, solutions) as [N, 81] uint8. Blanks in the puzzle are 0."""
    csv_path = os.path.join(dataset_dir, f"{split}.csv")
    cache_path = os.path.join(dataset_dir, f"{split}_pairs.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["puzzles"], z["solutions"]

    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["question", "answer"], dtype={"question": str, "answer": str}).dropna()
    df = df[(df["question"].str.len() == 81) & (df["answer"].str.len() == 81)]
    puzzles = (np.frombuffer("".join(df["question"].str.replace(".", "0", regex=False)).encode(),
                             dtype=np.uint8) - ord("0")).reshape(-1, 81)
    solutions = (np.frombuffer("".join(df["answer"]).encode(), dtype=np.uint8) - ord("0")).reshape(-1, 81)
    np.savez(cache_path, puzzles=puzzles, solutions=solutions)
    return puzzles, solutions

def augment_grids(grids: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Apply ONE random element of HRM's symmetry group to each of the K grids in a sample.

    `grids`: [batch, K, 81]. Sharing the transform across K is what lets a puzzle and its solution
    be augmented together -- they must stay a matching pair. Blanks (0) survive the digit
    relabelling because the map fixes 0.
    """
    batch, k = grids.shape[:2]
    device = grids.device
    g = grids.view(batch, k, 9, 9).long()

    flip = torch.rand(batch, device=device, generator=generator) < 0.5
    g = torch.where(flip[:, None, None, None], g.transpose(-1, -2), g)

    def band_perm() -> Tensor:
        bands = torch.rand(batch, 3, device=device, generator=generator).argsort(dim=1)
        within = torch.rand(batch, 3, 3, device=device, generator=generator).argsort(dim=2)
        sel = torch.gather(within, 1, bands[:, :, None].expand(batch, 3, 3))
        return (bands[:, :, None] * 3 + sel).reshape(batch, 9)

    g = g.gather(2, band_perm()[:, None, :, None].expand(batch, k, 9, 9))
    g = g.gather(3, band_perm()[:, None, None, :].expand(batch, k, 9, 9))

    digit_map = torch.rand(batch, 9, device=device, generator=generator).argsort(dim=1) + 1
    digit_map = torch.cat([torch.zeros(batch, 1, dtype=digit_map.dtype, device=device), digit_map], dim=1)
    return digit_map[:, None, :].expand(batch, k, 10).gather(2, g.reshape(batch, k, 81))

def augment_boards(boards: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """HRM's `shuffle_sudoku`, vectorised on GPU so augmentation never bottlenecks the batch.

    Exactly the same symmetry group: transpose, permute the three bands and the rows within each
    band, the same for stacks and columns, and relabel the digits. Every draw gets a fresh random
    element of that group, so with 3.8M base boards the model effectively never sees a repeat.
    """
    return augment_grids(boards.unsqueeze(1), generator).squeeze(1)

# [Pixel-space board <-> digits]
class BoardCodec:
    """Maps digit boards (1..9) to/from the continuous pixel space the flow lives in.

    Both representations are whitened (zero mean, unit variance over a uniform digit) so the data
    end of the path is scaled like the N(0, I) noise end.
    """
    # one-hot: Bernoulli(1/9) per channel   |   scalar: uniform digit on 1..9
    ONEHOT_MEAN, ONEHOT_STD = 1.0 / 9.0, math.sqrt((1.0 / 9.0) * (8.0 / 9.0))
    SCALAR_MEAN, SCALAR_STD = 5.0, math.sqrt(60.0 / 9.0)

    def __init__(self, repr_name: str):
        assert repr_name in ("onehot", "scalar")
        self.repr_name = repr_name
        self.in_channels = 9 if repr_name == "onehot" else 1

    def encode(self, digits: Tensor) -> Tensor:
        # digits: [batch, 81] in 1..9  ->  [batch, 81, in_channels]
        if self.repr_name == "onehot":
            return (F.one_hot(digits.long() - 1, 9).float() - self.ONEHOT_MEAN) / self.ONEHOT_STD
        return ((digits.float() - self.SCALAR_MEAN) / self.SCALAR_STD).unsqueeze(-1)

    def decode(self, x: Tensor) -> Tensor:
        # x: [batch, 81, in_channels]  ->  [batch, 81] in 1..9
        if self.repr_name == "onehot":
            return x.argmax(dim=-1) + 1
        return (x.squeeze(-1).float() * self.SCALAR_STD + self.SCALAR_MEAN).round().clamp(1, 9).long()

# [Flow matching]
def sample_t(batch_size: int, schedule: str, device: torch.device) -> Tensor:
    t = torch.rand(batch_size, device=device)
    if schedule == "logit_normal":  # SD3-style: spend more capacity on the middle of the path
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    return t

def flow_matching_loss(model: nn.Module, x1: Tensor, t_schedule: str,
                       cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
    """Rectified-flow (conditional OT) objective. Returns (loss, x1_hat, t)."""
    x0 = torch.randn_like(x1)
    t = sample_t(x1.shape[0], t_schedule, x1.device)

    x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    v = model(x_t, t, cond).float()
    loss = F.mse_loss(v, target)
    # Where the model thinks the endpoint is: x_1_hat = x_t + (1 - t) * v
    x1_hat = x_t + (1.0 - t[:, None, None]) * v
    return loss, x1_hat, t

@torch.inference_mode()
def euler_sample(model: nn.Module, num_samples: int, seq_len: int, in_channels: int,
                 steps: int, device: torch.device, generator: Optional[torch.Generator] = None,
                 cond: Optional[Tensor] = None) -> Tensor:
    """Integrate dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (board) with uniform Euler steps."""
    x = torch.randn(num_samples, seq_len, in_channels, device=device, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((num_samples,), i * dt, device=device)
        x = x + dt * model(x, t, cond).float()
    return x

@torch.inference_mode()
def _drift(model, x, t, num, cond, guidance, noise_scale):
    """Drift of the marginal-preserving SDE: v + (g^2/2) * score, with g(t) = noise_scale (1-t).

    The (1-t)^2 in g^2 cancels the score's 1/(1-t), so this stays finite as t -> 1.
    """
    t_batch = torch.full((num, ), t, device=x.device)
    v = model(x, t_batch, cond).float()
    if guidance != 1.0 and cond is not None:
        v_uncond = model(x, t_batch, torch.zeros_like(cond)).float()
        v = v_uncond + guidance * (v - v_uncond)
    if noise_scale <= 0:
        return v
    return v + 0.5 * (noise_scale ** 2) * (1.0 - t) * (t * v - x)

@torch.inference_mode()
def sde_sample(model: nn.Module, num_samples: int, seq_len: int, in_channels: int, steps: int,
               device: torch.device, generator: Optional[torch.Generator] = None,
               noise_scale: float = 1.0, cond: Optional[Tensor] = None,
               clamp_x1: Optional[Tensor] = None, clamp_mask: Optional[Tensor] = None,
               guidance: float = 1.0, sampler: str = "euler") -> Tensor:
    """Stochastic sampler for the same trained velocity field.

    For the linear path x_t = (1-t) x_0 + t x_1 with x_0 ~ N(0, I) the score is recoverable from
    the velocity: E[x_0 | x_t] = x_t - t v, so

        score(x, t) = -E[x_0 | x_t] / (1 - t) = (t v(x, t) - x) / (1 - t).

    Any g(t) >= 0 then gives an SDE with the *same* marginals as the probability-flow ODE:

        dx = [v + (g^2 / 2) score] dt + g dW.

    Taking g(t) = noise_scale * (1 - t) makes both the drift correction and the injected noise
    vanish as t -> 1, which cancels the 1/(1-t) in the score. noise_scale=0 recovers Euler.
    """
    x = torch.randn(num_samples, seq_len, in_channels, device=device, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        t, t_next = i * dt, min((i + 1) * dt, 1.0)
        if clamp_mask is not None:
            # Replacement-style conditioning: cells whose value is already known are pinned to
            # their own forward-noised value, so the model only has to fill in the rest. Generic
            # conditional-diffusion practice (observed variables stay observed), not a Sudoku rule.
            known = (1.0 - t) * torch.randn(x.shape, device=device, generator=generator) + t * clamp_x1
            x = torch.where(clamp_mask[..., None], known, x)

        # One Wiener increment per step, shared by every stage of the integrator (the diffusion
        # coefficient depends only on t, so the noise is additive and this stays consistent).
        dw = (noise_scale * (1.0 - t) * math.sqrt(dt)
              * torch.randn(x.shape, device=device, generator=generator)) if noise_scale > 0 else 0.0

        k1 = _drift(model, x, t, num_samples, cond, guidance, noise_scale)
        if sampler == "euler":
            x = x + k1 * dt + dw
        elif sampler == "heun":                       # 2nd order: predictor, then trapezoid
            x_pred = x + k1 * dt + dw
            k2 = _drift(model, x_pred, t_next, num_samples, cond, guidance, noise_scale)
            x = x + 0.5 * (k1 + k2) * dt + dw
        elif sampler == "rk4":                        # classical RK4 on the drift
            half = t + 0.5 * dt
            k2 = _drift(model, x + 0.5 * dt * k1 + 0.5 * dw, half, num_samples, cond, guidance, noise_scale)
            k3 = _drift(model, x + 0.5 * dt * k2 + 0.5 * dw, half, num_samples, cond, guidance, noise_scale)
            k4 = _drift(model, x + dt * k3 + dw, t_next, num_samples, cond, guidance, noise_scale)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + dw
        else:
            raise ValueError(f"unknown sampler: {sampler}")
    return x

# [Sudoku validity metrics]
def group_view(boards: np.ndarray) -> np.ndarray:
    """[N, 81] digits -> [N, 27, 9]: the 9 rows, 9 columns and 9 boxes of each board."""
    g = boards.reshape(-1, 9, 9)
    rows = g
    cols = g.transpose(0, 2, 1)
    boxes = g.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    return np.concatenate([rows, cols, boxes], axis=1)

def board_metrics(boards: np.ndarray) -> dict[str, float]:
    """Constraint satisfaction of generated boards. `boards`: [N, 81] digits in 1..9."""
    groups = group_view(boards)  # [N, 27, 9]
    group_ok = (np.sort(groups, axis=-1) == np.arange(1, 10)).all(axis=-1)  # [N, 27]

    n = boards.shape[0]
    # How many distinct digits each group holds (9 = satisfied); a soft version of `group_ok`
    sorted_groups = np.sort(groups, axis=-1)
    num_distinct = 1 + (np.diff(sorted_groups, axis=-1) != 0).sum(axis=-1)  # [N, 27]

    return {
        "valid_board_rate": float(group_ok.all(axis=-1).mean()),
        "group_satisfaction": float(group_ok.mean()),
        "row_satisfaction": float(group_ok[:, :9].mean()),
        "col_satisfaction": float(group_ok[:, 9:18].mean()),
        "box_satisfaction": float(group_ok[:, 18:].mean()),
        "mean_distinct_per_group": float(num_distinct.mean()),
        "unique_sample_rate": float(len(np.unique(boards, axis=0)) / max(n, 1)),
    }

def render_boards(boards: np.ndarray, max_boards: int = 4) -> str:
    out = []
    for b in boards[:max_boards]:
        g = b.reshape(9, 9)
        lines = []
        for r in range(9):
            cells = " ".join("".join(str(int(v)) for v in g[r, c:c + 3]) for c in (0, 3, 6))
            lines.append(cells)
            if r in (2, 5):
                lines.append("--- --- ---")
        out.append("\n".join(lines))
    return "\n\n".join(out)

# [Training]
def update_lr(optim: torch.optim.Optimizer, step: int, args) -> float:
    if step < args.lr_warmup_steps:
        lr = args.lr * min(1.0, step / max(args.lr_warmup_steps, 1))
    else:
        progress = (step - args.lr_warmup_steps) / max(args.train_steps - args.lr_warmup_steps, 1)
        lr = args.lr * (args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in optim.param_groups:
        param_group["lr"] = tensor_lr
    return lr

def infinite_loader(loader):
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1

@torch.inference_mode()
def evaluate_conditional(net: nn.Module, codec: "BoardCodec", args, device: torch.device,
                         world_size: int, rank: int, puzzles: Tensor, solutions: Tensor) -> dict[str, float]:
    """Solve held-out puzzles: sample a solution conditioned on the puzzle, score exact match.

    `exact_match` is the same metric the HRM baselines in this repo report.
    """
    per_rank = max(args.eval_samples // world_size, 1)
    lo = rank * per_rank
    cond, truth = puzzles[lo:lo + per_rank], solutions[lo:lo + per_rank]
    generator = torch.Generator(device=device).manual_seed(1234 + rank)

    clamp_x1 = clamp_mask = None
    if args.clamp_givens:
        clamp_mask = cond > 0
        clamp_x1 = codec.encode(cond.clamp_min(1))
    x = sde_sample(net, cond.shape[0], 81, codec.in_channels, args.sample_steps, device,
                   generator, args.noise_scale, cond=cond, clamp_x1=clamp_x1, clamp_mask=clamp_mask,
                   guidance=args.guidance)
    pred = codec.decode(x)
    if args.clamp_givens:      # the givens are known; report them as given
        pred = torch.where(cond > 0, cond, pred)
    given = cond > 0
    boards = pred.cpu().numpy()

    stats = torch.stack([
        (pred == truth).all(dim=-1).sum(),
        (pred == truth).sum(),
        (pred[given] == cond[given]).sum(),
        torch.tensor(cond.shape[0], device=device),
        torch.tensor(cond.numel(), device=device),
        given.sum(),
    ]).double()
    if world_size > 1:
        dist.all_reduce(stats)
    n_exact, n_cell, n_given, n_boards, n_cells, n_givens = stats.tolist()
    return {
        "exact_match": n_exact / n_boards,
        "cell_accuracy": n_cell / n_cells,
        "givens_respected": n_given / max(n_givens, 1.0),
        "valid_board_rate": board_metrics(boards)["valid_board_rate"],
    }

@torch.inference_mode()
def evaluate(net: nn.Module, codec: BoardCodec, args, device: torch.device,
             world_size: int, rank: int) -> tuple[dict[str, float], np.ndarray]:
    """Sample boards and score them against the Sudoku constraints."""
    per_rank = max(args.eval_samples // world_size, 1)
    generator = torch.Generator(device=device).manual_seed(1234 + rank)

    x = euler_sample(net, per_rank, 81, codec.in_channels, args.sample_steps, device, generator)
    boards = codec.decode(x)

    if world_size > 1:
        gathered = [torch.empty_like(boards) for _ in range(world_size)]
        dist.all_gather(gathered, boards)
        boards = torch.cat(gathered, dim=0)

    boards_np = boards.cpu().numpy()
    return board_metrics(boards_np), boards_np

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data (HRM pipeline)
    parser.add_argument("--dataset-name", type=str, default="./downloaded-datasets/sudoku-extreme",
                        help="Directory holding train.csv (the full sudoku-extreme, or the 1k subset)")
    parser.add_argument("--no-augment", action="store_true", help="Disable HRM's band/stack/digit-permutation augmentation")
    parser.add_argument("--repr", dest="repr_name", type=str, default="onehot", choices=["onehot", "scalar"])
    # Model
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True,
                        help="RMS-normalise q/k per head; bounds attention logits and stops mid-training loss spikes")
    parser.add_argument("--pos-embed", type=str, default="rope2d", choices=["rope2d", "rope1d", "learned"],
                        help="How positions enter: axial 2D RoPE over the grid, stock 1D RoPE, or a learned input embedding")
    parser.add_argument("--forward-dtype", type=str, default="bfloat16")
    # Flow matching
    parser.add_argument("--t-schedule", type=str, default="uniform", choices=["uniform", "logit_normal"])
    parser.add_argument("--sample-steps", type=int, default=100, help="Integration steps used when sampling")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="Stochastic sampler noise level used at eval; 0 is the deterministic ODE")
    parser.add_argument("--conditional", action="store_true",
                        help="Condition on a puzzle and generate its solution (the HRM task)")
    parser.add_argument("--eval-split", type=str, default="test_hard", help="Split for conditional eval")
    parser.add_argument("--cond-dropout", type=float, default=0.0,
                        help="Probability of dropping the puzzle during training; enables guidance at sampling")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="Classifier-free guidance scale at sampling (1.0 = plain conditional)")
    parser.add_argument("--clamp-givens", action="store_true",
                        help="Pin the puzzle's known cells during sampling (replacement inpainting)")
    parser.add_argument("--eval-samples", type=int, default=512)
    # Optimization
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--local-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip; 0 disables")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--ema", type=float, default=0.999, help="0 disables the EMA of the weights")
    # Bookkeeping
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Distributed setup (mirrors train.py)
    world_size, rank, device_id = 1, 0, 0
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        device_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    run_name = args.run_name or os.environ.get("MLP_TASK_NAME") or f"flow_sudoku {coolname.generate_slug(2)}"
    codec = BoardCodec(args.repr_name)

    # Data: the HRM loader, of which we keep only the solution side
    # Every solved board held in memory; batches are drawn and augmented on the GPU, so the data
    # side costs nothing and the batch size is limited only by the model.
    if rank == 0:
        load_solutions(args.dataset_name)     # build the .npy cache once, before the other ranks read it
    if world_size > 1:
        dist.barrier()
    if args.conditional:
        train_q, train_a = load_pairs(args.dataset_name, "train")
        puzzles = torch.from_numpy(train_q.astype(np.int64)).to(device)
        solutions = torch.from_numpy(train_a.astype(np.int64)).to(device)
        test_q, test_a = load_pairs(args.dataset_name, args.eval_split)
        pick = np.random.default_rng(0).permutation(len(test_q))[:max(args.eval_samples, 1) * 4]
        test_puzzles = torch.from_numpy(test_q[pick].astype(np.int64)).to(device)
        test_solutions = torch.from_numpy(test_a[pick].astype(np.int64)).to(device)
    else:
        puzzles = None
        solutions = torch.from_numpy(load_solutions(args.dataset_name).astype(np.int64)).to(device)
    seq_len = 81

    # Model
    with torch.device(device):
        model: nn.Module = SudokuFlowTransformer(dict(
            seq_len=seq_len,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            head_dim=args.head_dim,
            is_causal=False,
            norm_eps=args.norm_eps,
            rope_theta=args.rope_theta,
            in_channels=codec.in_channels,
            forward_dtype=args.forward_dtype,
            pos_embed=args.pos_embed,
            conditional=args.conditional,
            qk_norm=args.qk_norm,
        ))
        num_params = sum(p.numel() for p in model.parameters())
        if not args.no_compile:
            model = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]
        net = model  # unwrapped view used for sampling
        if world_size > 1:
            model = DDP(model, static_graph=True)

    optim = AdamATan2(
        model.parameters(),
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
        ema=args.ema if args.ema and args.ema > 0 else None,
    )

    checkpoint_dir = os.path.join("checkpoints", run_name)
    use_wandb = (rank == 0) and not args.no_wandb
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
            json.dump(vars(args) | {"num_params": num_params, "seq_len": seq_len}, f, indent=2)
        print(f"[flow_sudoku] {run_name}: {num_params / 1e6:.2f}M params, pos_embed={args.pos_embed}, "
              f"{solutions.shape[0]} boards, {args.local_batch_size * world_size} global batch, "
              f"{args.train_steps} steps", flush=True)
        if use_wandb:
            wandb.init(project=WANDB_PROJECT, name=run_name, group=run_name,
                       config=vars(args) | {"num_params": num_params, "world_size": world_size},
                       settings=wandb.Settings(x_disable_stats=True))

    def run_eval_and_save(step: int, best_valid: float) -> float:
        # Everything below happens with the EMA weights swapped in, so the checkpoint on disk is
        # exactly the model that produced the metrics.
        optim.swap_ema()
        model.eval()
        if args.conditional:
            metrics = evaluate_conditional(net, codec, args, device, world_size, rank,
                                           test_puzzles, test_solutions)
            boards = None
        else:
            metrics, boards = evaluate(net, codec, args, device, world_size, rank)

        if rank == 0:
            print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            log = {f"eval/{k}": v for k, v in metrics.items()}
            if boards is not None:
                print(render_boards(boards, max_boards=1), flush=True)
                log["eval/samples"] = wandb.Html(f"<pre>{render_boards(boards)}</pre>")
            if use_wandb:
                wandb.log(log, step=step)

            state_dict = {k.replace("_orig_mod.", ""): v for k, v in
                          (model.module if world_size > 1 else model).state_dict().items()}
            torch.save(state_dict, os.path.join(checkpoint_dir, "last.pt"))
            score = metrics["exact_match"] if args.conditional else metrics["valid_board_rate"]
            if score > best_valid:
                best_valid = score
                torch.save(state_dict, os.path.join(checkpoint_dir, "best.pt"))
            del state_dict

        model.train()
        optim.swap_ema()
        return best_valid

    best_valid = -1.0
    progress_bar = tqdm.tqdm(total=args.train_steps, disable=rank != 0)
    model.train()
    # Each rank draws its own boards; the seed offset keeps the ranks from sampling in lockstep.
    data_gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)

    for step in range(1, args.train_steps + 1):
        idx = torch.randint(solutions.shape[0], (args.local_batch_size, ), device=device, generator=data_gen)
        if args.conditional:
            pair = torch.stack([puzzles[idx], solutions[idx]], dim=1)
            if not args.no_augment:
                pair = augment_grids(pair, data_gen)
            cond, boards = pair[:, 0], pair[:, 1]
            if args.cond_dropout > 0:
                # Drop the puzzle sometimes so the same network also learns the unconditional
                # field. An all-blank puzzle IS the unconditional case, so the null condition is
                # just "no cells given" -- no extra token needed.
                drop = torch.rand(cond.shape[0], device=device, generator=data_gen) < args.cond_dropout
                cond = torch.where(drop[:, None], torch.zeros_like(cond), cond)
        else:
            cond = None
            boards = solutions[idx]
            if not args.no_augment:
                boards = augment_boards(boards, data_gen)
        x1 = codec.encode(boards)

        lr = update_lr(optim, step, args)
        loss, x1_hat, t = flow_matching_loss(model, x1, args.t_schedule, cond)
        loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0 else torch.zeros((), device=device))
        optim.step()
        optim.zero_grad()

        if step % args.log_interval == 0:
            with torch.no_grad():
                # How often the model's current endpoint estimate decodes to the right digit
                cell_acc = (codec.decode(x1_hat) == codec.decode(x1)).float().mean()
                metrics = {"train/loss": loss.item(), "train/x1_cell_accuracy": cell_acc.item(),
                           "train/grad_norm": grad_norm.item(), "train/lr": lr}
            progress_bar.set_postfix(loss=f"{metrics['train/loss']:.4f}", acc=f"{metrics['train/x1_cell_accuracy']:.3f}",
                                     gnorm=f"{metrics['train/grad_norm']:.2f}")
            if use_wandb:
                wandb.log(metrics, step=step)

        if step % args.eval_interval == 0 or step == args.train_steps:
            best_valid = run_eval_and_save(step, best_valid)

        progress_bar.update(1)

    progress_bar.close()
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

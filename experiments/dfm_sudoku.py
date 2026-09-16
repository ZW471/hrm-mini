"""Discrete flow matching that solves Sudoku -- the categorical counterpart of `flow_sudoku.py`.

`flow_sudoku.py` relaxes a board to a point in R^{81 x 9} and learns a continuous velocity field.
This file keeps the board *discrete*: every cell is a token in {1..9}, the flow is a continuous-time
Markov chain (CTMC) on that space, and the model is a per-cell classifier p_theta(x_1 | x_t, t)
trained with cross-entropy. Same backbone, same conditioning, same budget; only the state space
and the loss change. (Campbell et al. 2024, "Generative Flows on Discrete State-Spaces"; Gat et al.
2024, "Discrete Flow Matching".)

Two priors, chosen with `--prior`:
  * `mask`     x_0 = all cells MASK (token 0). p_t(x_t | x_1) per cell: x_1 with prob t, else MASK.
               Unmasked cells are correct by construction, so the loss is on masked cells only and
               the generator can only *add* cells -- unless the noise term below re-masks some.
  * `uniform`  x_0 = uniform digits. p_t(x_t | x_1) per cell: x_1 with prob t, else a uniform digit.
               The model cannot tell which cells are wrong, so it predicts x_1 for every cell and
               the generator keeps *correcting* cells that disagree with its prediction. This is the
               discrete analogue of the continuous model's "guess, then repair".

Sampling is Euler on the CTMC. At each step, sample x_1_hat ~ p_theta(x_1 | x_t) per cell, then apply
the x_1-conditioned rate:
  mask:     MASK -> x_1_hat with prob dt (1 + eta t) / (1 - t);   x -> MASK with prob dt eta
  uniform:  x != x_1_hat -> x_1_hat with prob dt/(1-t) (1 + eta (tK + 1 - t)/K);   x -> Unif with prob dt eta
`eta` (`--noise-scale`) is the detailed-balance stochasticity of Campbell et al.: it adds noise
without changing the marginals, exactly as the SDE's g(t) does for the continuous model. eta = 0 is
the minimal (deterministic-rate) generator.

Conditioning is the same as the continuous model -- `cond_embed(puzzle)` on the input, 10% puzzle
dropout, classifier-free guidance on the logits -- except that given cells never need clamping:
they are simply never noised.

`--givens soft` drops that hard constraint: the given cells are noised, predicted and scored like
every other cell, and the sampler never writes the puzzle back into the state. The puzzle then
reaches the model only through `cond_embed`, as a hint it has to learn to honour rather than a
clamp. `eval_dfm_sudoku.py --clamp-givens` can still pin them at sampling time for such a model.

Usage
-----
    uv run torchrun --nproc-per-node 8 experiments/dfm_sudoku.py --prior mask
    uv run python experiments/dfm_sudoku.py --train-steps 2000 --eval-interval 500   # smoke test
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
from arch.layers import CastedLinear, CastedScaledEmbedding, MLP, Transformer, TransformerConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_sudoku import (RotaryEmbedding2D, NoRotaryEmbedding, timestep_embedding,
                                     load_pairs, augment_grids, board_metrics, update_lr, log_wandb)

K = 9          # digits
MASK = 0       # token id of the mask state (also "blank" in the puzzle encoding)

# [Model]
class DFMTransformerConfig(TransformerConfig):
    forward_dtype: str
    pos_embed: str = "rope2d"
    grid_height: int = 9
    grid_width: int = 9
    conditional: bool = False
    self_cond: bool = False

class SudokuDiscreteFlowTransformer(nn.Module):
    """Encoder-only transformer p_theta(x_1 | x_t, t, puzzle) over the 81 cells of a board.

    Identical to `SudokuFlowTransformer` except at the boundary: a token embedding in, per-cell
    digit logits out.
    """
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = DFMTransformerConfig(**config_dict)
        self.config = config
        self.dtype = getattr(torch, config.forward_dtype)

        self.core = Transformer(config)
        if config.pos_embed == "rope2d":
            self.core.rotary_emb = RotaryEmbedding2D(config.head_dim, config.grid_height,
                                                     config.grid_width, base=config.rope_theta)
        elif config.pos_embed != "rope1d":
            raise ValueError(f"unknown pos_embed: {config.pos_embed}")

        self.tok_embed = CastedScaledEmbedding(K + 1, config.hidden_size, cast_to=self.dtype)  # 0 = MASK
        self.lm_head = CastedLinear(config.hidden_size, K, bias=True)
        self.t_mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)
        self.cond_embed = (CastedScaledEmbedding(K + 1, config.hidden_size, cast_to=self.dtype)
                           if config.conditional else None)
        # Self-conditioning (Chen et al., "Analog Bits"): the model also sees its own previous
        # posterior over every cell, so soft information about undecided cells survives between
        # steps -- the discrete counterpart of what the continuous x_t carries. Zero-init so it
        # starts as the plain model.
        self.self_cond_proj = CastedLinear(K, config.hidden_size, bias=False) if config.self_cond else None
        if self.self_cond_proj is not None:
            with torch.no_grad():
                self.self_cond_proj.weight.zero_()

    def forward(self, x_t: Tensor, t: Tensor, cond: Optional[Tensor] = None,
                p_prev: Optional[Tensor] = None) -> Tensor:
        # x_t: [batch, 81] tokens in 0..9, t: [batch], cond: [batch, 81] puzzle digits (0 = blank)
        # p_prev: [batch, 81, K] previous posterior (self-conditioning), or None
        h = self.tok_embed(x_t)
        if self.cond_embed is not None:
            h = h + self.cond_embed(cond)
        if self.self_cond_proj is not None:
            if p_prev is None:
                p_prev = torch.zeros(*x_t.shape, K, device=x_t.device)
            h = h + self.self_cond_proj(p_prev.to(self.dtype))
        t_emb = self.t_mlp(timestep_embedding(t, self.config.hidden_size, self.config.rope_theta).to(self.dtype))
        return self.lm_head(self.core(h + t_emb[:, None, :]))  # [batch, 81, K]

# [Discrete flow matching]
def sample_prior(shape, prior: str, device, generator=None) -> Tensor:
    if prior == "mask":
        return torch.full(shape, MASK, dtype=torch.long, device=device)
    return torch.randint(1, K + 1, shape, device=device, generator=generator)

def corrupt(x1: Tensor, t: Tensor, prior: str, generator=None) -> tuple[Tensor, Tensor]:
    """Draw x_t ~ p_t(. | x_1) per cell. Returns (x_t, kept) where kept marks cells still at x_1."""
    kept = torch.rand(x1.shape, device=x1.device, generator=generator) < t[:, None]
    noise = sample_prior(x1.shape, prior, x1.device, generator)
    return torch.where(kept, x1, noise), kept

def dfm_loss(model: nn.Module, x1: Tensor, prior: str, cond: Optional[Tensor], given: Optional[Tensor],
             loss_weight: str, generator=None, self_cond: bool = False, self_cond_p: float = 0.5,
             self_cond_passes: int = 1, net: Optional[nn.Module] = None) -> tuple[Tensor, Tensor, Tensor]:
    """Cross-entropy on p_theta(x_1 | x_t). Returns (loss, x1_hat, t)."""
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    x_t, kept = corrupt(x1, t, prior, generator)
    if given is not None:
        x_t = torch.where(given, x1, x_t)      # the givens are known at every t
        kept = kept | given

    p_prev = None
    if self_cond:
        # Half the time, condition on a detached first-pass posterior (as at sampling time); the
        # other half on zeros, so the model also works on the first step.
        p_prev = torch.zeros(*x1.shape, K, device=x1.device)
        if torch.rand((), device=x1.device, generator=generator).item() < self_cond_p:
            with torch.no_grad():
                for _ in range(self_cond_passes):   # >1: refine the posterior before the graded pass
                    p_prev = torch.softmax((net if net is not None else model)(x_t, t, cond, p_prev).float(), dim=-1)
    logits = model(x_t, t, cond, p_prev).float()
    ce = F.cross_entropy(logits.reshape(-1, K), (x1 - 1).reshape(-1), reduction="none").view_as(x1)

    if prior == "mask":
        target_cells = ~kept                    # unmasked cells carry no information to learn
    else:
        target_cells = ~given if given is not None else torch.ones_like(kept)
    weight = target_cells.float()
    if loss_weight == "elbo":                   # 1/(1-t): the MDLM / DFM ELBO weighting
        weight = weight / (1.0 - t[:, None]).clamp_min(1e-3)
    loss = (ce * weight).sum() / weight.sum().clamp_min(1.0)
    return loss, logits.argmax(dim=-1) + 1, t

def guided_logits(model, x, t_batch, cond, guidance, p_prev=None):
    logits = model(x, t_batch, cond, p_prev).float()
    if guidance != 1.0 and cond is not None:
        logits_u = model(x, t_batch, torch.zeros_like(cond), p_prev).float()
        logits = logits_u + guidance * (logits - logits_u)
    return logits

@torch.inference_mode()
def ctmc_sample(model: nn.Module, num_samples: int, prior: str, steps: int, device, generator=None,
                cond: Optional[Tensor] = None, given: Optional[Tensor] = None, eta: float = 0.0,
                guidance: float = 1.0, temperature: float = 1.0, final: str = "argmax",
                trace: Optional[list] = None, self_cond: bool = False) -> Tensor:
    """Euler integration of the x_1-conditioned CTMC from t=0 (prior) to t=1 (board)."""
    x = sample_prior((num_samples, 81), prior, device, generator)
    if given is not None:
        x = torch.where(given, cond, x)
    dt = 1.0 / steps
    p_prev = torch.zeros(num_samples, 81, K, device=device) if self_cond else None
    for i in range(steps):
        t = i * dt
        last = i == steps - 1
        t_batch = torch.full((num_samples, ), t, device=device)
        logits = guided_logits(model, x, t_batch, cond, guidance, p_prev)
        if self_cond:
            p_prev = torch.softmax(logits, dim=-1)
        if last and final == "argmax":
            x1_hat = logits.argmax(dim=-1) + 1
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            x1_hat = torch.multinomial(probs.view(-1, K), 1, generator=generator).view(num_samples, 81) + 1
        if trace is not None:
            trace.append((x.clone(), x1_hat.clone()))

        u = torch.rand(x.shape, device=device, generator=generator)
        if prior == "mask":
            p_unmask = 1.0 if last else min(1.0, dt * (1.0 + eta * t) / (1.0 - t))
            is_mask = x == MASK
            new = torch.where(is_mask & (u < p_unmask), x1_hat, x)
        else:
            p_jump = 1.0 if last else min(1.0, dt / (1.0 - t) * (1.0 + eta * (t * K + 1.0 - t) / K))
            new = torch.where((x != x1_hat) & (u < p_jump), x1_hat, x)
        if eta > 0 and not last:
            # detailed-balance noise: re-noise a cell at rate eta (mask) / eta per uniform draw
            u2 = torch.rand(x.shape, device=device, generator=generator)
            noise = sample_prior(x.shape, prior, device, generator)
            new = torch.where(u2 < dt * eta, noise, new)
        if given is not None:
            new = torch.where(given, cond, new)
        x = new
    return x

# [Evaluation]
@torch.inference_mode()
def evaluate_conditional(net, args, device, world_size, rank, puzzles, solutions) -> dict[str, float]:
    """Solve held-out puzzles at every eta in `--eval-noise-scales`; the first one is the headline."""
    per_rank = max(args.eval_samples // world_size, 1)
    lo = rank * per_rank
    cond, truth = puzzles[lo:lo + per_rank], solutions[lo:lo + per_rank]
    given = (cond > 0) if args.givens == "hard" else None
    split = args.eval_split
    out: dict[str, float] = {}
    for i, eta in enumerate(args.eval_noise_scales):
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        pred = ctmc_sample(net, cond.shape[0], args.prior, args.sample_steps, device, generator,
                           cond=cond, given=given, eta=eta, guidance=args.guidance,
                           temperature=args.temperature, final=args.final, self_cond=args.self_cond)
        boards = pred.cpu().numpy()
        stats = torch.stack([
            (pred == truth).all(dim=-1).sum(),
            (pred == truth).sum(),
            torch.tensor(cond.shape[0], device=device),
            torch.tensor(cond.numel(), device=device),
        ]).double()
        if world_size > 1:
            dist.all_reduce(stats)
        n_exact, n_cell, n_boards, n_cells = stats.tolist()
        suffix = "" if i == 0 else f"_eta{eta:g}"
        out[f"{split}_exact_match{suffix}"] = n_exact / n_boards
        out[f"{split}_cell_accuracy{suffix}"] = n_cell / n_cells
        out[f"{split}_valid_board_rate{suffix}"] = board_metrics(boards)["valid_board_rate"]
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", type=str, default="./downloaded-datasets/sudoku-extreme")
    parser.add_argument("--eval-dataset-name", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default="test_hard")
    parser.add_argument("--no-augment", action="store_true")
    # Model (defaults = the 113m flow runs)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pos-embed", type=str, default="rope2d", choices=["rope2d", "rope1d"])
    parser.add_argument("--forward-dtype", type=str, default="bfloat16")
    # Discrete flow
    parser.add_argument("--prior", type=str, default="mask", choices=["mask", "uniform"])
    parser.add_argument("--loss-weight", type=str, default="uniform", choices=["uniform", "elbo"])
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--noise-scale", type=float, default=0.0, help="eta: detailed-balance stochasticity at sampling")
    parser.add_argument("--eval-noise-scales", type=float, nargs="+", default=None,
                        help="etas to evaluate during training; the first is the headline metric (default: --noise-scale)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--final", type=str, default="argmax", choices=["argmax", "sample"],
                        help="How the last step commits the remaining cells")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--givens", type=str, default="hard", choices=["hard", "soft"],
                        help="hard: given cells are never noised, never scored, always written back. "
                             "soft: they are generated like every other cell; the puzzle is only a hint")
    parser.add_argument("--self-cond", action="store_true", help="Self-conditioning on the previous posterior")
    parser.add_argument("--self-cond-p", type=float, default=0.5, help="Fraction of training steps that see a first-pass posterior")
    parser.add_argument("--self-cond-passes", type=int, default=1, help="No-grad refinement passes before the graded pass")
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=512)
    # Optimization (defaults = run_flow_seeds.sh)
    parser.add_argument("--train-steps", type=int, default=83_200, help="Length of the LR schedule")
    parser.add_argument("--stop-step", type=int, default=None, help="Stop early at this step, keeping the schedule")
    parser.add_argument("--local-batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--ema", type=float, default=0.999)
    # Bookkeeping
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--wandb-keys", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-project", type=str, default="sudoku")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.eval_noise_scales is None:
        args.eval_noise_scales = [args.noise_scale]

    world_size, rank, device_id = 1, 0, 0
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        device_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    run_name = args.run_name or f"dfm_sudoku {coolname.generate_slug(2)}"

    # Data: (puzzle, solution) pairs in memory, augmented on the GPU
    if rank == 0:
        load_pairs(args.dataset_name, "train")
        load_pairs(args.eval_dataset_name or args.dataset_name, args.eval_split)
    if world_size > 1:
        dist.barrier()
    train_q, train_a = load_pairs(args.dataset_name, "train")
    puzzles = torch.from_numpy(train_q.astype(np.int64)).to(device)
    solutions = torch.from_numpy(train_a.astype(np.int64)).to(device)
    test_q, test_a = load_pairs(args.eval_dataset_name or args.dataset_name, args.eval_split)
    pick = np.random.default_rng(0).permutation(len(test_q))[:max(args.eval_samples, 1) * 4]
    test_puzzles = torch.from_numpy(test_q[pick].astype(np.int64)).to(device)
    test_solutions = torch.from_numpy(test_a[pick].astype(np.int64)).to(device)

    with torch.device(device):
        model: nn.Module = SudokuDiscreteFlowTransformer(dict(
            seq_len=81, num_layers=args.num_layers, hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size, head_dim=args.head_dim, is_causal=False,
            norm_eps=args.norm_eps, rope_theta=args.rope_theta, forward_dtype=args.forward_dtype,
            pos_embed=args.pos_embed, conditional=args.conditional, qk_norm=args.qk_norm,
            self_cond=args.self_cond,
        ))
        num_params = sum(p.numel() for p in model.parameters())
        if not args.no_compile:
            model = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]
        net = model
        if world_size > 1:
            model = DDP(model, static_graph=True)

    optim = AdamATan2(model.parameters(), lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                      betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                      ema=args.ema if args.ema and args.ema > 0 else None)

    wandb_keys = set(args.wandb_keys) if args.wandb_keys else None
    checkpoint_dir = os.path.join("checkpoints", run_name, f"seed_{args.seed}")
    use_wandb = (rank == 0) and not args.no_wandb
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
            json.dump(vars(args) | {"num_params": num_params}, f, indent=2)
        print(f"[dfm_sudoku] {run_name}: {num_params / 1e6:.2f}M params, prior={args.prior}, "
              f"{solutions.shape[0]} boards, {args.local_batch_size * world_size} global batch, "
              f"{args.train_steps} steps", flush=True)
        if use_wandb:
            wandb.init(project=args.wandb_project, name=run_name, group=run_name,
                       config=vars(args) | {"num_params": num_params, "world_size": world_size},
                       settings=wandb.Settings(x_disable_stats=True))

    def run_eval_and_save(step: int, best: float) -> float:
        optim.swap_ema()
        model.eval()
        metrics = evaluate_conditional(net, args, device, world_size, rank, test_puzzles, test_solutions)
        if rank == 0:
            print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            if use_wandb:
                log_wandb({f"eval/{k}": v for k, v in metrics.items()}, step, wandb_keys)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in
                          (model.module if world_size > 1 else model).state_dict().items()}
            torch.save(state_dict, os.path.join(checkpoint_dir, "last.pt"))
            score = metrics[f"{args.eval_split}_exact_match"]
            if score > best:
                best = score
                torch.save(state_dict, os.path.join(checkpoint_dir, "best.pt"))
            del state_dict
        model.train()
        optim.swap_ema()
        return best

    best = -1.0
    progress_bar = tqdm.tqdm(total=args.train_steps, disable=rank != 0)
    model.train()
    data_gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)

    for step in range(1, args.train_steps + 1):
        idx = torch.randint(solutions.shape[0], (args.local_batch_size, ), device=device, generator=data_gen)
        pair = torch.stack([puzzles[idx], solutions[idx]], dim=1)
        if not args.no_augment:
            pair = augment_grids(pair, data_gen)
        cond, x1 = pair[:, 0], pair[:, 1]
        if args.conditional:
            if args.cond_dropout > 0:
                drop = torch.rand(cond.shape[0], device=device, generator=data_gen) < args.cond_dropout
                cond = torch.where(drop[:, None], torch.zeros_like(cond), cond)
            given = (cond > 0) if args.givens == "hard" else None
        else:
            cond, given = None, None

        lr = update_lr(optim, step, args)
        loss, x1_hat, t = dfm_loss(model, x1, args.prior, cond, given, args.loss_weight, data_gen,
                                   self_cond=args.self_cond, self_cond_p=args.self_cond_p,
                                   self_cond_passes=args.self_cond_passes, net=net)
        loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0 else torch.zeros((), device=device))
        optim.step()
        optim.zero_grad()

        if step % args.log_interval == 0:
            with torch.no_grad():
                correct = x1_hat == x1
                metrics = {"train/loss": loss.item(),
                           "train/per_position_accuracy": correct.float().mean().item(),
                           "train/exact_match": correct.all(dim=-1).float().mean().item(),
                           "train/grad_norm": grad_norm.item(), "train/lr": lr}
            progress_bar.set_postfix(loss=f"{metrics['train/loss']:.4f}",
                                     acc=f"{metrics['train/per_position_accuracy']:.3f}",
                                     em=f"{metrics['train/exact_match']:.3f}",
                                     gnorm=f"{metrics['train/grad_norm']:.2f}")
            if use_wandb:
                log_wandb(metrics, step, wandb_keys)

        if step % args.eval_interval == 0 or step == args.train_steps:
            best = run_eval_and_save(step, best)
        progress_bar.update(1)
        if args.stop_step is not None and step >= args.stop_step:
            break

    progress_bar.close()
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

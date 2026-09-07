"""Latent-space flow matching for solved Sudoku boards.

Two stages, both in one run:

1. **VAE.** An encoder transformer reads the 81 cells and pools them into a single latent vector
   z (`--latent-dim`), with the usual reparameterised Gaussian posterior. A decoder transformer
   expands z back to 81 positions and predicts a 9-way distribution per cell. Trained with
   cross-entropy (summed over cells, i.e. nats per board) + `--kl-beta` * KL (nats per board). The whole board -- all 81 digits -- has to fit through that
   one vector, so reconstruction accuracy is the first thing worth looking at.

2. **Flow.** With the VAE frozen, a rectified flow is trained on the (whitened) latents with an
   MLP velocity field. Sampling is z_0 ~ N(0, I) -> Euler -> unwhiten -> VAE decoder -> argmax.

The point of comparison against `flow_sudoku.py`: there the flow runs in pixel space over 81
positions and the transformer has to enforce the constraints itself. Here the constraints are the
decoder's job and the flow only has to model a low-dimensional latent distribution.

Diagnostics reported at the end of stage 1, which mostly determine whether stage 2 can work:
  * `recon_exact_match` -- can the decoder invert the latent at all?
  * `prior_valid_rate`  -- decode z ~ N(0, I) with no flow. If the aggregate posterior already
    matches the prior this is high, and the flow has little left to fix.
  * `latent_std`        -- how far the aggregate posterior is from the unit Gaussian.
  * decoder robustness  -- cell accuracy and valid rate after perturbing the latent by eps. A
    decoder that only accepts near-exact latents leaves the flow no room: a KL of thousands of
    nats reconstructs perfectly and still cannot be sampled from, because every small flow error
    decodes to garbage. Keeping the KL down to tens of nats is what makes stage 2 possible.

Usage
-----
    uv run python experiments/latent_flow_sudoku.py --latent-tokens 16 --latent-dim 32 --kl-beta 1e-4
    experiments/run_latent_sweep.sh          # 8 configurations, one per GPU
"""

from typing import Any, Optional
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import tqdm
import wandb
import coolname

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adam_atan2 import AdamATan2
from arch.layers import CastedLinear, CastedScaledEmbedding, MLP, Transformer, TransformerConfig, trunc_normal_init_
from dataset.sudoku import create_dataloader
from experiments.flow_sudoku import (RotaryEmbedding2D, SudokuFlowTransformer, board_metrics,
                                     euler_sample, infinite_loader, render_boards, sample_t)

WANDB_PROJECT = "sudoku-flow-matching"

# [Stage 1: VAE]
class SudokuVAE(nn.Module):
    """Encoder transformer -> latent grid [latent_tokens, latent_dim] -> decoder -> digit logits."""
    def __init__(self, hidden_size: int, intermediate_size: int, head_dim: int, num_layers: int,
                 latent_tokens: int, latent_dim: int, norm_eps: float, rope_theta: float,
                 forward_dtype: str, qk_norm: bool = True, rope_2d: bool = True,
                 logvar_init: float = -4.0):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.norm_eps = norm_eps
        self.hidden_size = hidden_size
        # The posterior starts nearly deterministic. Left at logvar=0 the sampled noise (std 1)
        # swamps mu, the decoder learns to ignore z, and the posterior collapses before it ever
        # becomes informative.
        self.logvar_init = logvar_init
        self.dtype = getattr(torch, forward_dtype)

        config = TransformerConfig(seq_len=81, num_layers=num_layers, hidden_size=hidden_size,
                                   intermediate_size=intermediate_size, head_dim=head_dim,
                                   is_causal=False, norm_eps=norm_eps, rope_theta=rope_theta,
                                   qk_norm=qk_norm)
        self.encoder = Transformer(config)
        self.decoder = Transformer(config)
        if rope_2d:
            for core in (self.encoder, self.decoder):
                core.rotary_emb = RotaryEmbedding2D(head_dim, 9, 9, base=rope_theta)

        self.embed = CastedScaledEmbedding(10, hidden_size, cast_to=self.dtype)
        # Learned weightings over the 81 positions, one row per latent token, and the mirror image
        # of that on the way out. A plain mean here does not work at all -- see the module docstring.
        self.pool_weight = nn.Parameter(trunc_normal_init_(torch.empty(latent_tokens, 81), std=1.0))
        self.unpool_weight = nn.Parameter(trunc_normal_init_(torch.empty(latent_tokens, 81), std=1.0))
        self.to_latent = CastedLinear(hidden_size, 2 * latent_dim, bias=True)
        self.from_latent = CastedLinear(latent_dim, hidden_size, bias=True)
        # The decoder input is the same vector at every position, so it needs its own positions
        self.decoder_pos = nn.Parameter(trunc_normal_init_(torch.empty(81, hidden_size), std=1.0))
        self.head = CastedLinear(hidden_size, 9, bias=True)

    def encode(self, digits: Tensor) -> tuple[Tensor, Tensor]:
        # Pooling shrinks magnitudes, so renormalise before projecting -- otherwise mu comes out
        # small relative to the sampled noise and the decoder learns to ignore the latent.
        h = self.encoder(self.embed(digits))                               # [batch, 81, hidden]
        h = torch.einsum("bph,lp->blh", h.float(), self.pool_weight.float())  # [batch, tokens, hidden]
        h = F.rms_norm(h, (self.hidden_size, ), eps=self.norm_eps).to(self.dtype)
        mu, logvar = self.to_latent(h).float().chunk(2, dim=-1)            # [batch, tokens, latent_dim]
        return mu, logvar + self.logvar_init

    def normalize(self, z: Tensor) -> Tensor:
        """Pin the latent to unit RMS per token.

        Without this, `decoder_noise` is meaningless: the encoder just scales mu up until the
        injected noise is negligible again (mu_std climbed to 2.1 for sigma=1.5), so the
        signal-to-noise ratio -- and with it the flow's real error budget -- stays wherever the
        encoder puts it. Fixing the latent scale makes sigma the actual budget.
        """
        return F.rms_norm(z.float(), (self.latent_dim, ), eps=self.norm_eps)

    def decode(self, z: Tensor) -> Tensor:
        g = self.from_latent(z.to(self.dtype))                             # [batch, tokens, hidden]
        h = torch.einsum("blh,lp->bph", g.float(), self.unpool_weight.float()).to(self.dtype)
        h = h + self.decoder_pos.to(self.dtype)                            # [batch, 81, hidden]
        return self.head(self.decoder(h))                                  # [batch, 81, 9]

    def forward(self, digits: Tensor, decoder_noise: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(digits)
        z = self.normalize(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar))
        # Extra noise seen only by the decoder, on the normalised latent. The posterior width sets
        # how much information the latent carries (and so how much signal the flow has to model);
        # this sets how much latent error the decoder can absorb. Tying them together, as a plain
        # VAE does, forces a choice between a latent the flow can learn and a decoder that accepts
        # the flow's output.
        if decoder_noise > 0:
            z = z + decoder_noise * torch.randn_like(z)
        return self.decode(z), mu, logvar

def vae_loss(logits: Tensor, digits: Tensor, mu: Tensor, logvar: Tensor, kl_beta: float, free_bits: float = 0.0):
    """Both terms are per-BOARD nats, so `kl_beta` means what it does in a beta-VAE.

    Reconstruction is summed over the 81 cells, not averaged: it is the log-likelihood of one whole
    board, which is what the per-board KL trades against. Averaging instead (the obvious thing to
    write) shrinks the reconstruction term 81x relative to the KL and collapses the posterior --
    the latent then carries nothing and every cell decodes at chance.
    """
    recon = F.cross_entropy(logits.reshape(-1, 9).float(), (digits - 1).reshape(-1).long(),
                            reduction="none").view(digits.shape[0], -1).sum(dim=-1).mean()
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())        # [batch, tokens, latent_dim]
    # Free bits: no pressure on a dimension until it carries more than `free_bits` nats
    kl_clamped = (kl_per_dim.clamp_min(free_bits) if free_bits > 0 else kl_per_dim).flatten(1).sum(dim=-1).mean()
    kl = kl_per_dim.flatten(1).sum(dim=-1).mean()
    return recon + kl_beta * kl_clamped, recon, kl

# [Stage 2: flow over the latent grid]
def build_flow(args, latent_dim: int) -> nn.Module:
    """The pixel-space flow transformer, pointed at the latent grid instead of the 9x9 board.

    The latent tokens are an unordered, unstructured set as far as the model is concerned: plain
    1D RoPE over the sequence, no grid geometry. Laying them out as a square (and especially as
    3x3) would hand the model the board's block structure, which is exactly the Sudoku-specific
    prior this experiment must not contain -- the latent layout has to be generic.
    """
    return SudokuFlowTransformer(dict(
        seq_len=args.latent_tokens,
        num_layers=args.flow_layers,
        hidden_size=args.flow_hidden_size,
        intermediate_size=args.flow_intermediate_size,
        head_dim=args.head_dim,
        is_causal=False,
        norm_eps=args.norm_eps,
        rope_theta=args.rope_theta,
        qk_norm=True,
        in_channels=latent_dim,
        forward_dtype=args.forward_dtype,
        pos_embed="rope1d",
    ))

# [Helpers]
def decode_boards(vae: SudokuVAE, z: Tensor, normalize: bool = True) -> np.ndarray:
    """Decode a latent. `normalize` projects onto the unit-RMS shell the decoder was trained on,
    which also removes one degree of freedom from whatever error the flow made."""
    inner = getattr(vae, "_orig_mod", vae)
    return (vae.decode(inner.normalize(z) if normalize else z).argmax(dim=-1) + 1).cpu().numpy()

def make_optimizer(module: nn.Module, args) -> AdamATan2:
    return AdamATan2(module.parameters(),
                     lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                     betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                     ema=args.ema if args.ema and args.ema > 0 else None)

def update_lr(optim: torch.optim.Optimizer, step: int, total_steps: int, args, base_lr: float) -> float:
    if step < args.lr_warmup_steps:
        lr = base_lr * min(1.0, step / max(args.lr_warmup_steps, 1))
    else:
        progress = (step - args.lr_warmup_steps) / max(total_steps - args.lr_warmup_steps, 1)
        lr = base_lr * (args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for group in optim.param_groups:
        group["lr"] = tensor_lr
    return lr

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data (HRM pipeline, solution side only -- same as flow_sudoku.py)
    parser.add_argument("--dataset-name", type=str, default="./downloaded-datasets/sudoku-extreme-1k")
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--no-augment", action="store_true")
    # VAE
    parser.add_argument("--latent-tokens", type=int, default=16, help="Number of latent tokens (a generic set, no grid geometry)")
    parser.add_argument("--latent-dim", type=int, default=32, help="Channels per latent token")
    parser.add_argument("--kl-beta", type=float, default=0.1,
                        help="beta-VAE weight; both loss terms are per-board nats, so 1.0 is the plain ELBO")
    parser.add_argument("--kl-warmup-steps", type=int, default=2000, help="Linearly anneal beta from 0")
    parser.add_argument("--free-bits", type=float, default=0.0, help="Nats per latent dim exempt from the KL penalty")
    parser.add_argument("--logvar-init", type=float, default=-4.0, help="Offset on the posterior log-variance head")
    parser.add_argument("--decoder-noise", type=float, default=0.6,
                        help="Extra Gaussian noise on the latent, decoder-side only: the error budget the flow gets")
    parser.add_argument("--vae-only", action="store_true", help="Stop after stage 1 (capability check)")
    parser.add_argument("--vae-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vae-steps", type=int, default=30000)
    # Latent flow (a SudokuFlowTransformer over the latent grid)
    parser.add_argument("--flow-hidden-size", type=int, default=512)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-intermediate-size", type=int, default=2048)
    parser.add_argument("--flow-steps", type=int, default=30000)
    parser.add_argument("--t-schedule", type=str, default="uniform", choices=["uniform", "logit_normal"])
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=1024)
    # Shared
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--forward-dtype", type=str, default="bfloat16")
    parser.add_argument("--local-batch-size", type=int, default=256)
    parser.add_argument("--vae-lr", type=float, default=3e-4)
    parser.add_argument("--flow-lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    run_name = args.run_name or f"latent_flow {coolname.generate_slug(2)}"

    train_loader, _metadata = create_dataloader(
        "train", args.local_batch_size, rank=0, world_size=1, dataset_name=args.dataset_name,
        augment=not args.no_augment, repeat=args.repeat, seed=args.seed)
    data = infinite_loader(train_loader)

    with torch.device(device):
        vae = SudokuVAE(args.hidden_size, args.intermediate_size, args.head_dim, args.vae_layers,
                        args.latent_tokens, args.latent_dim, args.norm_eps, args.rope_theta,
                        args.forward_dtype, logvar_init=args.logvar_init)
        flow = build_flow(args, args.latent_dim)
        num_params = sum(p.numel() for p in vae.parameters()), sum(p.numel() for p in flow.parameters())
        if not args.no_compile:
            vae = torch.compile(vae, dynamic=False, fullgraph=True)    # pyright: ignore[reportAssignmentType]
            flow = torch.compile(flow, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args) | {"vae_params": num_params[0], "flow_params": num_params[1]}, f, indent=2)
    print(f"[latent_flow] {run_name}: VAE {num_params[0]/1e6:.2f}M + flow {num_params[1]/1e6:.2f}M params, "
          f"latent={args.latent_tokens}x{args.latent_dim}, kl_beta={args.kl_beta}", flush=True)
    if not args.no_wandb:
        wandb.init(project=WANDB_PROJECT, name=run_name, group=run_name,
                   config=vars(args) | {"vae_params": num_params[0], "flow_params": num_params[1],
                                        "experiment": "latent_flow"},
                   settings=wandb.Settings(x_disable_stats=True))

    def next_boards() -> Tensor:
        _x, y = next(data)
        return y[:, 1:].to(device, non_blocking=True)  # drop BOS: [batch, 81] digits 1..9

    # ---------------- Stage 1: VAE ----------------
    optim = make_optimizer(vae, args)
    progress = tqdm.tqdm(total=args.vae_steps, desc="vae")
    vae.train()
    for step in range(1, args.vae_steps + 1):
        digits = next_boards()
        lr = update_lr(optim, step, args.vae_steps, args, args.vae_lr)

        logits, mu, logvar = vae(digits, args.decoder_noise)
        # Anneal beta in: a full-strength KL against an untrained decoder collapses the posterior
        # before the decoder can ever learn to use the latent.
        beta = args.kl_beta * min(1.0, step / max(args.kl_warmup_steps, 1))
        loss, recon, kl = vae_loss(logits, digits, mu, logvar, beta, args.free_bits)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
        optim.step()
        optim.zero_grad()

        if step % args.log_interval == 0:
            with torch.no_grad():
                acc = ((logits.argmax(-1) + 1) == digits).float().mean().item()
            progress.set_postfix(recon=f"{recon.item():.1f}", kl=f"{kl.item():.1f}", acc=f"{acc:.3f}")
            if not args.no_wandb:
                wandb.log({"vae/loss": loss.item(), "vae/recon_nats_per_board": recon.item(),
                           "vae/kl_nats": kl.item(), "vae/beta": beta,
                           "vae/cell_accuracy": acc, "vae/lr": lr}, step=step)
        progress.update(1)
    progress.close()

    # VAE diagnostics
    optim.swap_ema()
    vae.eval()
    with torch.inference_mode():
        digits = next_boards()
        mu, logvar = vae.encode(digits)
        recon_boards = decode_boards(vae, mu)
        truth = digits.cpu().numpy()
        recon_exact = float((recon_boards == truth).all(axis=-1).mean())
        recon_cell = float((recon_boards == truth).mean())

        # Whitening statistics over POSTERIOR SAMPLES, not means. Stage 2 trains on samples, and
        # a dimension the encoder does not use has mu std ~1e-3 while its samples still have the
        # posterior's std ~1 -- normalising samples by the std of mu blows those dimensions up by
        # ~1000x and the flow ends up modelling pure noise.
        inner = getattr(vae, "_orig_mod", vae)
        latents = torch.cat([inner.normalize(mu)]
                            + [inner.normalize(vae.encode(next_boards())[0]) for _ in range(7)], dim=0)
        latent_mean, latent_std = latents.mean(dim=0), latents.std(dim=0)
        mu_std = latent_std

        gen = torch.Generator(device=device).manual_seed(1234)
        prior_boards = decode_boards(vae, torch.randn(args.eval_samples, args.latent_tokens,
                                                      args.latent_dim, device=device, generator=gen))
    prior_metrics = board_metrics(prior_boards)

    # How far can a latent drift before the decoder stops producing a valid board? The flow will
    # never land exactly on the manifold, so this is the budget it has to work within.
    robustness = {}
    inner_vae = getattr(vae, "_orig_mod", vae)
    with torch.inference_mode():
        for eps in (0.1, 0.25, 0.5, 1.0, 1.5):
            noisy = decode_boards(vae, inner_vae.normalize(mu) + eps * torch.randn_like(mu), normalize=False)
            robustness[eps] = (float((noisy == truth).mean()), board_metrics(noisy)["valid_board_rate"])
    print("[vae] decoder robustness (noise on the unit-RMS latent -> cell acc / valid rate): "
          + "  ".join(f"eps={e}: {a:.3f}/{v:.3f}" for e, (a, v) in robustness.items()), flush=True)

    print(f"[vae] recon_exact_match={recon_exact:.4f}  recon_cell_accuracy={recon_cell:.4f}  "
          f"latent_std={latent_std.mean().item():.3f}  mu_std={mu_std.mean().item():.3f}  "
          f"prior_valid_rate={prior_metrics['valid_board_rate']:.4f}", flush=True)
    if not args.no_wandb:
        wandb.log({"vae/recon_exact_match": recon_exact, "vae/recon_cell_accuracy": recon_cell,
                   "vae/latent_std": latent_std.mean().item(), "vae/mu_std": mu_std.mean().item(),
                   "vae/latent_mean_abs": latent_mean.abs().mean().item(),
                   "vae/prior_valid_rate": prior_metrics["valid_board_rate"],
                   "vae/prior_group_satisfaction": prior_metrics["group_satisfaction"]}
                  | {f"vae/robust_eps{e}_cell_acc": a for e, (a, _v) in robustness.items()}
                  | {f"vae/robust_eps{e}_valid": v for e, (_a, v) in robustness.items()}, step=args.vae_steps)
    torch.save({k.replace("_orig_mod.", ""): v for k, v in vae.state_dict().items()},
               os.path.join(checkpoint_dir, "vae.pt"))
    # The EMA weights stay swapped in: stage 2 encodes with exactly the VAE we just measured.
    for p in vae.parameters():
        p.requires_grad_(False)

    if args.vae_only:
        print(f"[latent_flow] {run_name}: --vae-only, stopping after stage 1", flush=True)
        if not args.no_wandb:
            wandb.finish()
        return

    # ---------------- Stage 2: flow in latent space ----------------
    # Whiten with the aggregate posterior so the data end of the path is scaled like the prior.
    z_mean, z_std = latent_mean, latent_std.clamp_min(1e-2)

    optim = make_optimizer(flow, args)
    progress = tqdm.tqdm(total=args.flow_steps, desc="flow")
    flow.train()
    best_valid = -1.0
    with torch.no_grad():
        inner = getattr(vae, '_orig_mod', vae)
        real_latents = torch.cat([inner.normalize(vae.encode(next_boards())[0])
                                  for _ in range(8)], dim=0).flatten(1).float()

    def run_eval(step: int, best_valid: float) -> float:
        optim.swap_ema()
        flow.eval()
        gen = torch.Generator(device=device).manual_seed(1234)
        z = euler_sample(flow, args.eval_samples, args.latent_tokens, args.latent_dim,
                         args.sample_steps, device, gen)
        with torch.inference_mode():
            z_raw = getattr(vae, '_orig_mod', vae).normalize(z * z_std + z_mean)
            boards = decode_boards(vae, z_raw, normalize=False)
        with torch.inference_mode():
            # Per-dim RMS distance to the nearest real latent, in units of the decoder's noise
            # budget: below 1 the flow lands inside what the decoder absorbs, well above 1 it is
            # off-manifold and decoder robustness cannot help.
            d = torch.cdist(z_raw.flatten(1).float(), real_latents)
            nn_rms = (d.min(dim=1).values.pow(2) / real_latents.shape[1]).sqrt().mean().item()
        m = board_metrics(boards) | {"nn_per_dim": nn_rms,
                                     "nn_over_sigma": nn_rms / max(args.decoder_noise, 1e-9)}
        print(f"[flow step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if not args.no_wandb:
            wandb.log({f"eval/{k}": v for k, v in m.items()}
                      | {"eval/samples": wandb.Html(f"<pre>{render_boards(boards)}</pre>")},
                      step=args.vae_steps + step)
        state = {k.replace("_orig_mod.", ""): v for k, v in flow.state_dict().items()}
        torch.save(state, os.path.join(checkpoint_dir, "flow_last.pt"))
        if m["valid_board_rate"] > best_valid:
            best_valid = m["valid_board_rate"]
            torch.save(state, os.path.join(checkpoint_dir, "flow_best.pt"))
        flow.train()
        optim.swap_ema()
        return best_valid

    for step in range(1, args.flow_steps + 1):
        with torch.no_grad():
            # Target the means: with a narrow posterior these carry essentially all the signal, so
            # the flow is not spending its capacity fitting isotropic posterior noise.
            mu, _logvar = vae.encode(next_boards())
            x1 = (getattr(vae, "_orig_mod", vae).normalize(mu) - z_mean) / z_std

        lr = update_lr(optim, step, args.flow_steps, args, args.flow_lr)
        x0 = torch.randn_like(x1)
        t = sample_t(x1.shape[0], args.t_schedule, device)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = flow(x_t, t).float()
        loss = F.mse_loss(v, x1 - x0)
        loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
                     if args.grad_clip > 0 else torch.zeros((), device=device))
        optim.step()
        optim.zero_grad()

        if step % args.log_interval == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{grad_norm.item():.2f}")
            if not args.no_wandb:
                wandb.log({"flow/loss": loss.item(), "flow/grad_norm": grad_norm.item(), "flow/lr": lr},
                          step=args.vae_steps + step)
        if step % args.eval_interval == 0 or step == args.flow_steps:
            best_valid = run_eval(step, best_valid)
        progress.update(1)

    progress.close()
    print(f"[latent_flow] {run_name} done: best valid_board_rate={best_valid:.4f}", flush=True)
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

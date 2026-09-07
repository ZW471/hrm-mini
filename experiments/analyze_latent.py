"""Post-hoc diagnosis of a trained latent flow: is the flow landing on the latent manifold?

The decoder is trained to absorb per-dimension latent noise of size `--decoder-noise` (sigma), so
that is the flow's error budget. This measures what the flow actually delivers:

    nn_dist_per_dim      per-dimension RMS distance from a flow sample to the nearest real latent
    nn_dist_over_sigma   the same in units of the budget -- below 1 means the flow lands inside
                         the decoder's tolerance, well above 1 means it is off-manifold and no
                         amount of decoder robustness will rescue it

As a control it also reports the same distance for real latents (nearest *other* real latent),
which is the spacing of the manifold itself.

    uv run python experiments/analyze_latent.py --run-dir checkpoints/latent_t16x32_kl1e-3_dn1.0
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.sudoku import create_dataloader
from experiments.flow_sudoku import board_metrics, euler_sample, infinite_loader, render_boards
from experiments.latent_flow_sudoku import SudokuVAE, build_flow

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--real-batches", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=100)
    cli = parser.parse_args()

    a = SimpleNamespace(**json.load(open(os.path.join(cli.run_dir, "args.json"))))
    device = torch.device("cuda")
    with torch.device(device):
        vae = SudokuVAE(a.hidden_size, a.intermediate_size, a.head_dim, a.vae_layers,
                        a.latent_tokens, a.latent_dim, a.norm_eps, a.rope_theta, a.forward_dtype,
                        logvar_init=a.logvar_init)
        vae.load_state_dict(torch.load(os.path.join(cli.run_dir, "vae.pt"), map_location=device, weights_only=True))
        vae.eval()
        flow = build_flow(a, a.latent_dim)
        flow.load_state_dict(torch.load(os.path.join(cli.run_dir, "flow_best.pt"), map_location=device, weights_only=True))
        flow.eval()

    loader, _ = create_dataloader("train", a.local_batch_size, rank=0, world_size=1,
                                  dataset_name=a.dataset_name, augment=not a.no_augment,
                                  repeat=a.repeat, seed=a.seed)
    data = infinite_loader(loader)

    with torch.no_grad():
        real = torch.cat([vae.normalize(vae.encode(next(data)[1][:, 1:].to(device))[0])
                          for _ in range(cli.real_batches)], dim=0)
        z_mean, z_std = real.mean(dim=0), real.std(dim=0).clamp_min(1e-2)
        real_flat = real.flatten(1).float()
        dim = real_flat.shape[1]

        gen = torch.Generator(device=device).manual_seed(1234)
        z = euler_sample(flow, cli.num_samples, a.latent_tokens, a.latent_dim, cli.sample_steps, device, gen)
        z_raw = vae.normalize(z * z_std + z_mean)
        boards = (vae.decode(z_raw).argmax(dim=-1) + 1).cpu().numpy()

        gen_nn = torch.cdist(z_raw.flatten(1).float(), real_flat).min(dim=1).values
        gen_rms = (gen_nn.pow(2) / dim).sqrt().mean().item()
        # control: spacing of the manifold itself (nearest OTHER real latent)
        d_real = torch.cdist(real_flat[:cli.num_samples], real_flat)
        d_real.fill_diagonal_(float("inf"))
        real_rms = (d_real.min(dim=1).values.pow(2) / dim).sqrt().mean().item()

    m = board_metrics(boards)
    sigma = max(getattr(a, "decoder_noise", 0.0), 1e-9)
    print(f"{os.path.basename(cli.run_dir)}")
    print(f"  valid_board_rate={m['valid_board_rate']:.4f}  group_satisfaction={m['group_satisfaction']:.4f}")
    print(f"  flow sample -> nearest real latent : {gen_rms:.4f} per dim  = {gen_rms / sigma:.2f} x sigma({sigma})")
    print(f"  real latent -> nearest other real  : {real_rms:.4f} per dim  (manifold spacing)")
    print(f"  verdict: {'INSIDE the decoder budget' if gen_rms < sigma else 'OFF-MANIFOLD (beyond the budget)'}")

if __name__ == "__main__":
    main()

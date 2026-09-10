from typing import Any, Optional
import os

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, MLP, Carry, trunc_normal_init_

class CycleFFConfig(TransformerConfig):
    vocab_size: int

    cycles: int
    bptt: bool

    forward_dtype: str

    # --- Cycle FF layer ---
    # Number of layers in one cycle FF layer. 0 disables it, leaving the plain RecurrentTransformer.
    cff_layers: int = 0
    # "mlp": a post-norm feed-forward block (no attention), the cheap "dummy" H layer.
    # "block": a full transformer block, i.e. what HRM's H level actually is.
    cff_type: str = "mlp"
    # Defaults to the core's `intermediate_size`.
    cff_intermediate_size: Optional[int] = None
    # Share one cycle FF layer across all cycles (HRM ties its H level), or give each cycle its own.
    cff_tied: bool = True
    # Run a cycle FF layer every `cff_period` core cycles -- HRM's L_cycles per H cycle.
    cff_period: int = 1
    # "inject": the layer owns a slow state z_H fed back into the core's input (HRM-like).
    # "inline": the layer writes straight into the core's fast state, carrying nothing across cycles.
    cff_mode: str = "inject"
    # Wrap the layer in a zero-initialised per-channel gate. Required to make a run resume from a
    # pretrained core as an exact no-op; pointless (and a slower start) when training from scratch.
    cff_gated: bool = False
    # Learning rate multiplier for the cycle FF parameters relative to the rest of the model.
    cff_lr_mult: float = 1.0

    # Pretrained RecurrentTransformer to start from (keys core.* / embed.* / lm_head.* / z_init).
    pretrained_ckpt: Optional[str] = None
    # Freeze the pretrained core + embedding + readout, so only the cycle FF layers train.
    freeze_core: bool = False

class CycleFF(nn.Module):
    """A post-norm feed-forward block: the attention-free version of an HRM H-level block."""
    def __init__(self, config: CycleFFConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MLP(hidden_size=config.hidden_size,
                intermediate_size=config.cff_intermediate_size or config.intermediate_size)
            for _ in range(config.cff_layers)
        ])
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def forward(self, h: Tensor) -> Tensor:
        for layer in self.layers:
            h = self.norm(h + layer(h))
        return h

class RecurrentTransformerCFF(nn.Module):
    """A recurrent transformer with a cycle FF layer spliced between its cycles.

    The RecurrentTransformer applies one shared `core` to `z + x` for `cycles` iterations. Every
    iteration is identical, so there is no slow timescale of the kind HRM's H level provides -- and
    on Sudoku-Extreme 1k the RT tops out at ~70.7% against HRM's ~81.3% at matched parameters and
    matched block-forwards. This module adds that missing timescale as a small extra network run
    once per cycle:

        for i in 1..cycles:
            z   = core(z + z_H + x)
            if i % cff_period == 0:
                z_H = cff_i(z_H + z)                 # cff_gated: z_H + g * cff_i(z_H + z)

    `cff_tied` chooses whether every cycle reuses one layer (as HRM ties its H level) or each cycle
    gets its own. `cff_type` chooses a plain feed-forward block or a full transformer block.

    With `cff_gated` the layer is wrapped in a zero-initialised per-channel gate and z_H starts at
    exactly zero, so a run resuming from a pretrained core begins bit-for-bit identical to it
    (verified: max|logit diff| = 0). That makes "did the cycle FF layer help?" answerable against
    the checkpoint's own accuracy. Training from scratch there is nothing to preserve, so the gate
    is off by default and z_H gets HRM's random init.

    `cff_mode="inline"` is the ablation where the layer edits the fast state (`z = z + g * cff(z)`)
    instead of maintaining a slow one, separating "second timescale" from "more depth per cycle".
    """
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = CycleFFConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.cycles = config.cycles
        self.bptt = config.bptt
        self.cff_period = config.cff_period
        self.cff_mode = config.cff_mode
        self.cff_gated = config.cff_gated
        self.cff_lr_mult = config.cff_lr_mult
        assert self.cff_mode in ("inject", "inline")
        assert config.cff_type in ("mlp", "block")

        # Backbone Layers (identical to arch.rt.RecurrentTransformer, so its checkpoints load as-is)
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initial z
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

        # Cycle FF layers: one, or one per cycle that runs one.
        self.has_cff = config.cff_layers > 0
        if self.has_cff:
            num_cff = 1 if config.cff_tied else self.cycles // self.cff_period
            def build():
                if config.cff_type == "block":
                    return Transformer(config.model_copy(update={
                        "num_layers": config.cff_layers,
                        "intermediate_size": config.cff_intermediate_size or config.intermediate_size,
                    }))
                return CycleFF(config)
            self.cff = nn.ModuleList([build() for _ in range(num_cff)])
            # Zero-init gate, so a grafted run starts as an exact no-op. Absent when ungated.
            self.cff_gate = nn.Parameter(torch.zeros(config.hidden_size)) if config.cff_gated else None
            # Gated runs need z_H = 0 exactly, to leave the core's input untouched at init.
            zH = torch.zeros(config.hidden_size, dtype=dtype) if config.cff_gated \
                else trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype))
            self.zH_init = nn.Buffer(zH, persistent=True)

        if config.pretrained_ckpt is not None:
            self._load_pretrained(config.pretrained_ckpt)
        if config.freeze_core:
            for module in (self.core, self.embed, self.lm_head):
                module.requires_grad_(False)

    def _load_pretrained(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"pretrained_ckpt not found: {path}\n"
                "Checkpoints are gitignored and do not survive a clone. Train a recurrent "
                "transformer first (`torchrun --nproc-per-node 8 train.py --config-name tuned_rt "
                "seeds=[1]`), then point this arm at it:\n"
                "  export RT_CKPT=\"checkpoints/<tuned_rt run dir>/seed_1/best.pt\"\n"
                "See CYCLE_FF.md for the full procedure.")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # The checkpoint is a plain RT, so it may only be missing the cycle FF layers' own parameters.
        assert not unexpected, f"unexpected keys in {path}: {unexpected}"
        assert all(k.startswith(("cff.", "cff_gate", "zH_init")) for k in missing), f"missing keys in {path}: {missing}"

    def param_groups(self) -> list[dict[str, Any]]:
        """Split parameters so the cycle FF layers can train at their own learning rate."""
        is_cff = lambda name: name.startswith(("cff.", "cff_gate", "zH_init"))
        groups = [
            {"params": [p for n, p in self.named_parameters() if not is_cff(n) and p.requires_grad], "lr_mult": 1.0},
            {"params": [p for n, p in self.named_parameters() if is_cff(n) and p.requires_grad], "lr_mult": self.cff_lr_mult},
        ]
        return [g for g in groups if g["params"]]

    def _apply_cff(self, idx: int, z: Tensor, z_H: Tensor) -> tuple[Tensor, Tensor]:
        cff = self.cff[idx % len(self.cff)]
        if self.cff_mode == "inject":
            out = cff(z_H + z)
            if self.cff_gate is None:
                return z, out
            return z, z_H + self.cff_gate.to(z.dtype) * out
        out = cff(z)
        return (z + self.cff_gate.to(z.dtype) * out) if self.cff_gate is not None else out, z_H

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)
        z_H = carry.get("z_H", 0.0)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z = carry["z"]
            for _i in range(self.cycles - 1):
                z = self.core(z + z_H + x) if self.has_cff else self.core(z + x)
                if self.has_cff and (_i + 1) % self.cff_period == 0:
                    z, z_H = self._apply_cff((_i + 1) // self.cff_period - 1, z, z_H)

        # 1-step grad
        z = self.core(z + z_H + x) if self.has_cff else self.core(z + x)
        if self.has_cff and self.cycles % self.cff_period == 0:
            z, z_H = self._apply_cff(self.cycles // self.cff_period - 1, z, z_H)

        # Ensure no gradient moves across carry
        new_carry: Carry = dict(z=z.detach())
        if self.has_cff:
            new_carry["z_H"] = z_H.detach()
        return new_carry, self.lm_head(z)

    @property
    def initial_carry(self) -> Carry:
        carry: Carry = dict(z=self.z_init)
        if self.has_cff:
            carry["z_H"] = self.zH_init
        return carry

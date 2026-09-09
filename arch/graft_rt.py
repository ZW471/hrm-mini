from typing import Any, Optional

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry, trunc_normal_init_

class GraftedRTConfig(TransformerConfig):
    vocab_size: int

    cycles: int
    bptt: bool

    forward_dtype: str

    # --- Graft ---
    # Size of the grafted H-network. 0 disables the graft entirely, which makes this module
    # bit-for-bit the plain RecurrentTransformer (used for the SFT-only / frozen-only controls).
    h_num_layers: int = 1
    # Defaults to the core's `intermediate_size`; set smaller to keep the graft cheap.
    h_intermediate_size: Optional[int] = None
    # Run the graft every `h_period` core cycles, HRM's L_cycles per H cycle.
    h_period: int = 1
    # "inject": the graft owns a slow state z_H that is fed back into the core's input (HRM-like).
    # "inline": the graft writes directly into the core's fast state z.
    h_mode: str = "inject"

    # Pretrained RecurrentTransformer to graft onto. Keys must be core.* / embed.* / lm_head.* / z_init.
    pretrained_ckpt: Optional[str] = None
    # Freeze the pretrained core + embedding + readout, so only the graft trains.
    freeze_core: bool = True

class GraftedRecurrentTransformer(nn.Module):
    """A frozen recurrent transformer with a small trainable network spliced between its cycles.

    The RecurrentTransformer applies one shared `core` to `z + x` for `cycles` iterations; every
    iteration does the same thing, so there is no slow timescale of the kind HRM's H level provides.
    This arm asks whether that missing timescale can be *added after the fact*: take an RT that has
    already been trained to convergence, freeze it, and insert a small H-block between consecutive
    core cycles.

        z_H = 0
        for i in 1..cycles:
            z = core(z + z_H + x)                          # frozen
            if i % h_period == 0:
                z_H = z_H + g * h_net(z_H + z)             # trainable, g is a per-channel gate

    The graft is zero-initialised (`g = 0`, `z_H` starts at 0), so at step 0 the model computes
    *exactly* what the pretrained RT computes -- same logits, same accuracy. Any movement from the
    RT baseline is therefore attributable to the graft rather than to a different starting point,
    and training cannot be handicapped by a bad initialisation of the new block.

    `h_mode="inline"` is the ablation where the graft edits the fast state directly
    (`z = z + g * h_net(z)`) instead of maintaining a separate slow state; it is also identity at
    init, and isolates whether a *persistent* second state matters or merely extra depth per cycle.
    """
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = GraftedRTConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.cycles = config.cycles
        self.bptt = config.bptt
        self.h_period = config.h_period
        self.h_mode = config.h_mode
        assert self.h_mode in ("inject", "inline")

        # Backbone Layers (identical to arch.rt.RecurrentTransformer, so its checkpoints load as-is)
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initial z
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

        # Grafted H-network
        self.has_graft = config.h_num_layers > 0
        if self.has_graft:
            h_config = config.model_copy(update={
                "num_layers": config.h_num_layers,
                "intermediate_size": config.h_intermediate_size or config.intermediate_size,
            })
            self.h_net = Transformer(h_config)
            # Per-channel LayerScale gate, zero-init: the graft starts as a no-op.
            self.h_gate = nn.Parameter(torch.zeros(config.hidden_size))
            # Slow state, kept at exactly zero at init so the frozen core sees its pretrained input.
            self.zH_init = nn.Buffer(torch.zeros(config.hidden_size, dtype=dtype), persistent=True)

        if config.pretrained_ckpt is not None:
            self._load_pretrained(config.pretrained_ckpt)
        if config.freeze_core:
            for module in (self.core, self.embed, self.lm_head):
                module.requires_grad_(False)

    def _load_pretrained(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # The checkpoint is a plain RT, so it may only be missing the graft's own parameters.
        assert not unexpected, f"unexpected keys in {path}: {unexpected}"
        assert all(k.startswith(("h_net.", "h_gate", "zH_init")) for k in missing), f"missing keys in {path}: {missing}"

    def _graft(self, z: Tensor, z_H: Tensor) -> tuple[Tensor, Tensor]:
        gate = self.h_gate.to(z.dtype)
        if self.h_mode == "inject":
            return z, z_H + gate * self.h_net(z_H + z)
        return z + gate * self.h_net(z), z_H

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)
        z_H = carry.get("z_H", 0.0)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z = carry["z"]
            for _i in range(self.cycles - 1):
                z = self.core(z + z_H + x) if self.has_graft else self.core(z + x)
                if self.has_graft and (_i + 1) % self.h_period == 0:
                    z, z_H = self._graft(z, z_H)

        # 1-step grad
        z = self.core(z + z_H + x) if self.has_graft else self.core(z + x)
        if self.has_graft and self.cycles % self.h_period == 0:
            z, z_H = self._graft(z, z_H)

        # Ensure no gradient moves across carry
        new_carry: Carry = dict(z=z.detach())
        if self.has_graft:
            new_carry["z_H"] = z_H.detach()
        return new_carry, self.lm_head(z)

    @property
    def initial_carry(self) -> Carry:
        carry: Carry = dict(z=self.z_init)
        if self.has_graft:
            carry["z_H"] = self.zH_init
        return carry

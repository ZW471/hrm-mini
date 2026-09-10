from typing import Any

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry, trunc_normal_init_

class SharedDeeplySupervisedMaskedAutoEncoderConfig(TransformerConfig):
    vocab_size: int

    num_unrolls: int
    layers_per_segment: int
    detach_segments: bool = True
    inject_input: bool = False
    use_carry: bool = False

    forward_dtype: str

class SharedDeeplySupervisedMaskedAutoEncoder(nn.Module):
    """Weight-tied masked auto-encoder with HRM-style deep supervision along depth.

    Same task setup and readout scheme as `mae_deep_supervision@DeeplySupervisedMaskedAutoEncoder`,
    but the depth is produced by *reusing* a stack of `num_layers` blocks `num_unrolls` times
    instead of stacking `num_layers * num_unrolls` distinct blocks. Block `i` of the unrolled
    forward pass is `self.core.layers[i % num_layers]`, so all layers share the same set of
    parameters, exactly as HRM reuses its H/L blocks across cycles. This decouples parameter count
    from compute, which is what lets a single model be parameter- *and* FLOP-matched to HRM.

    As in HRM there is a single output head, read out every `layers_per_segment` block-forwards and
    supervised at each readout, and the hidden state is detached there (`detach_segments`), so
    gradient never crosses a supervision point. Because the blocks are shared, each block still
    receives gradient from every segment it appears in -- the segments accumulate into one set of
    weights, like HRM's 1-step gradient accumulating over its supervision segments.

    Forward returns the logits of *all* supervision points, stacked as
    `[batch, num_segments, seq_len, vocab]`; the last entry is the final-layer prediction that
    training metrics and inference use.

    With `inject_input`, the embedded input is re-added to the hidden state at every supervision
    point, so each segment refines a scratchpad while seeing the puzzle again -- the `z + x` of
    `rt@RecurrentTransformer` -- instead of having to carry the puzzle through all the tied blocks.

    With `use_carry`, the final hidden state is additionally carried (detached) across the
    `cycles_per_data` supervision steps of a batch and across the same number of rounds at eval, so
    the model refines its own previous state instead of restarting from the embedding every time.
    That is HRM/RT's deep supervision along *recurrence*, on top of this model's supervision along
    depth; `cycles_per_data` therefore also multiplies the compute spent per sample at inference.
    """
    is_autoregressive = False

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = SharedDeeplySupervisedMaskedAutoEncoderConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.num_unrolls = config.num_unrolls
        self.layers_per_segment = config.layers_per_segment
        self.detach_segments = config.detach_segments
        self.inject_input = config.inject_input
        self.use_carry = config.use_carry

        # Backbone Layers -- the shared stack, applied `num_unrolls` times per forward pass
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initial z, as in `rt@RecurrentTransformer` -- only used when the state is carried
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        cos_sin = self.core.rotary_emb()
        x = self.embed(input_ids)
        h = x + carry["z"] if self.use_carry else x

        logits = []
        num_layers = len(self.core.layers)
        num_forwards = num_layers * self.num_unrolls
        for _i in range(num_forwards):
            h = self.core.layers[_i % num_layers](h, cos_sin=cos_sin)
            # Supervise every `layers_per_segment` block-forwards, and always after the last one
            if (_i + 1) % self.layers_per_segment == 0 or _i + 1 == num_forwards:
                logits.append(self.lm_head(h))
                if self.detach_segments:
                    h = h.detach()  # Ensure no gradient moves across supervision points
            # Re-present the puzzle every unroll of the shared stack, as `rt@RecurrentTransformer`
            # does every cycle. Independent of where the readouts are, so `layers_per_segment` can
            # be varied without changing the state trajectory. Not after the last block: that state
            # is the carry, and the next forward re-adds `x` itself.
            if self.inject_input and (_i + 1) % num_layers == 0 and _i + 1 < num_forwards:
                h = h + x

        # Ensure no gradient moves across the carry
        return (dict(z=h.detach()) if self.use_carry else {}), torch.stack(logits, dim=1)

    @property
    def initial_carry(self) -> Carry:
        return dict(z=self.z_init) if self.use_carry else {}

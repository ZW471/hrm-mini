from typing import Any

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry

class DeeplySupervisedMaskedAutoEncoderConfig(TransformerConfig):
    vocab_size: int

    layers_per_segment: int
    detach_segments: bool = True

    forward_dtype: str

class DeeplySupervisedMaskedAutoEncoder(nn.Module):
    """Masked auto-encoder with HRM-style deep supervision along depth instead of along recurrence.

    Same task setup and backbone as `mae@MaskedAutoEncoder`, but the readout is taken every
    `layers_per_segment` blocks instead of only at the end, and every readout is trained against
    the target. As in HRM there is a single output head -- the one that reads out the final layer
    is the same one applied at the intermediate supervision points -- and the hidden state is
    detached at each of them (`detach_segments`), so a block is only trained by the readout that
    immediately follows it, never by the later ones.

    Forward returns the logits of *all* supervision points, stacked as
    `[batch, num_segments, seq_len, vocab]`; the last entry is the final-layer prediction that
    training metrics and inference use.
    """
    is_autoregressive = False

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = DeeplySupervisedMaskedAutoEncoderConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.layers_per_segment = config.layers_per_segment
        self.detach_segments = config.detach_segments

        # Backbone Layers
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        cos_sin = self.core.rotary_emb()
        h = self.embed(input_ids)

        logits = []
        num_layers = len(self.core.layers)
        for _i, layer in enumerate(self.core.layers):
            h = layer(h, cos_sin=cos_sin)
            # Supervise every `layers_per_segment` blocks, and always after the last one
            if (_i + 1) % self.layers_per_segment == 0 or _i + 1 == num_layers:
                logits.append(self.lm_head(h))
                if self.detach_segments:
                    h = h.detach()  # Ensure no gradient moves across supervision points

        return {}, torch.stack(logits, dim=1)

    @property
    def initial_carry(self) -> Carry:
        return {}

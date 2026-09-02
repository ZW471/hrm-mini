from typing import Any

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry

class MaskedAutoEncoderConfig(TransformerConfig):
    vocab_size: int

    forward_dtype: str

class MaskedAutoEncoder(nn.Module):
    """Encoder-only (bidirectional) transformer trained as a masked auto-encoder.

    Same task setup as HRM: the blanked-out puzzle is encoded in one shot and every position
    predicts its solution token. Unlike HRM there is no recurrence, so the carry is empty and a
    single forward pass is the whole computation.
    """
    is_autoregressive = False

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = MaskedAutoEncoderConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        # Backbone Layers
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        return {}, self.lm_head(self.core(self.embed(input_ids)))

    @property
    def initial_carry(self) -> Carry:
        return {}

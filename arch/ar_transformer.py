from typing import Any

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry

class AutoregressiveTransformerConfig(TransformerConfig):
    vocab_size: int

    forward_dtype: str

class AutoregressiveTransformer(nn.Module):
    """Decoder-only causal transformer trained with next-token prediction.

    The puzzle is flattened into a single sequence `[question (seq_len) | answer (seq_len)]`, so the
    model runs on `2 * seq_len` positions with causal attention. `forward` returns the logits that
    predict the answer block, i.e. `logits[:, i]` predicts answer token `i` from everything up to
    answer token `i - 1`. That alignment lets the trainer compute exactly the same loss / accuracy /
    exact-match metrics as the encoder-only models, against the very same target tensor `y`.

    Training is teacher-forced (the answer block is fed in). Evaluation must instead decode the
    answer one token at a time -- see `generate` in train.py.
    """
    is_autoregressive = True

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        # Dataset metadata describes a single (question or answer) block and is written for the
        # encoder-only models; a decoder-only model sees both blocks and must attend causally.
        self.block_len = config_dict["seq_len"]
        config = AutoregressiveTransformerConfig(**(dict(config_dict) | {
            "seq_len": 2 * self.block_len,
            "is_causal": True,
        }))
        dtype = getattr(torch, config.forward_dtype)

        # Backbone Layers
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        # input_ids: [batch, 2 * block_len] = question ++ answer
        h = self.core(self.embed(input_ids))
        # Shift by one: position `block_len - 1 + i` is the last one visible when predicting answer token `i`
        return {}, self.lm_head(h[:, self.block_len - 1: 2 * self.block_len - 1])

    @property
    def initial_carry(self) -> Carry:
        return {}

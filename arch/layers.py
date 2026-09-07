from typing import Sequence
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import pydantic

CosSin = tuple[Tensor, Tensor]
Carry = dict[str, Tensor]

def trunc_normal_init_(tensor: Tensor, std: float = 1.0):
    """Fast approximate truncated normal initialization. Fairly accurate."""
    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)

def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # q, k: [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, batch_output_dims: Sequence[int] = (), **kwargs):
        super().__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((*batch_output_dims, out_features, in_features), **kwargs), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_output_dims) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.view(-1, self.in_features).to(input.dtype), self.bias.to(input.dtype) if self.bias is not None else None)

class CastedScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Scale to the same std as most parameters
        self.scale = embedding_dim ** 0.5
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=1.0 / self.scale)
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.scale * self.weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, batch_output_dims=(2, ), **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, is_causal, qk_norm: bool = False, norm_eps: float = 1e-6, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_causal = is_causal
        # RMS-normalising q and k per head bounds the attention logits, which stops the slow growth
        # that shows up as mid-training loss spikes. Off by default: the HRM baselines predate it.
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps

        self.qkv_proj = CastedLinear(hidden_size, self.num_heads * self.head_dim, bias=False, batch_output_dims=(3, ), **kwargs)
        self.o_proj = CastedLinear(head_dim * num_heads, hidden_size, bias=False, **kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: CosSin) -> Tensor:
        # hidden_states, qkv: [..., seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        qkv = rearrange(qkv, "... (h hd) -> ... h hd", h=self.num_heads)
        query, key, value = qkv.chunk(3, dim=-1)
        if self.qk_norm:
            query = F.rms_norm(query, (self.head_dim, ), eps=self.norm_eps)
            key = F.rms_norm(key, (self.head_dim, ), eps=self.norm_eps)
        # Rotary embedding
        query = apply_rotary_pos_emb(query, cos_sin)
        key = apply_rotary_pos_emb(key, cos_sin)
        # PyTorch SDPA attention
        # query, key, value: [... x seq_len x num_heads x head_dim]
        attn_output = F.scaled_dot_product_attention(query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3), is_causal=self.is_causal).transpose(-2, -3)
        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(attn_output, "... h hd -> ... (h hd)")
        return self.o_proj(attn_output)

class TransformerConfig(pydantic.BaseModel):
    seq_len: int

    num_layers: int

    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool

    norm_eps: float
    rope_theta: float

    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256

    qk_norm: bool = False

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.is_mlp_mixer = config.is_mlp_mixer
        if self.is_mlp_mixer:
            self.mlp_t = SwiGLU(config.seq_len, config.mlp_mixer_intermediate_size)
            self.mlp = SwiGLU(config.hidden_size, config.mlp_mixer_intermediate_size)
        else:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.hidden_size // config.head_dim,
                is_causal=config.is_causal,
                qk_norm=config.qk_norm,
                norm_eps=config.norm_eps
            )
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size
            )

        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    def forward(self, x: Tensor, **kwargs) -> Tensor:  # PostNorm
        if self.is_mlp_mixer:
            x = x.transpose(-1, -2)
            x = self.norm(x + self.mlp_t(x))
            x = x.transpose(-1, -2)
        else:
            x = self.norm(x + self.attn(x, **kwargs))
        return self.norm(x + self.mlp(x))

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.seq_len, base=config.rope_theta)

        self.layers = nn.ModuleList([TransformerBlock(config) for _layer_idx in range(config.num_layers)])

    def forward(self, h: Tensor) -> Tensor:
        cos_sin = self.rotary_emb()

        for layer in self.layers:
            h = layer(h, cos_sin=cos_sin)
        return h

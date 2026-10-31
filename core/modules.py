"""Neural building blocks: RoPE, norms, attention, SwiGLU MLP."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32.to(dtype)) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding, computed on the fly so any length works."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rope dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Key/value cache keyed by (layer, recursion step)."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.seq_len = 0

    def get(self, layer: int, step: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self._data.get((layer, step))

    def put(self, layer: int, step: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._data[(layer, step)] = (k, v)
        self.seq_len = max(self.seq_len, k.shape[2])

    def __len__(self) -> int:
        return len(self._data)

    def reset(self) -> None:
        self._data.clear()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.head_dim = cfg.head_dim
        self.dropout = dropout
        bias = cfg.use_bias
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer: int = 0,
        step: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        n_cached = 0
        if cache is not None:
            previous = cache.get(layer, step)
            if previous is not None:
                n_cached = previous[0].shape[2]
                k = torch.cat((previous[0], k), dim=2)
                v = torch.cat((previous[1], v), dim=2)
            cache.put(layer, step, k, v)

        if n_cached == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        elif seq_len == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        else:
            mask = torch.ones(seq_len, n_cached + seq_len, dtype=torch.bool, device=x.device)
            mask[:, n_cached:] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg, dropout: float = 0.0) -> None:
        super().__init__()
        bias = cfg.use_bias
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

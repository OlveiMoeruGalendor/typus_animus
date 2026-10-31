"""NCM Transformer - generates hierarchy coarse to fine, parallel within level."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import NCMTokenizerConfig


@dataclass
class NCMConfig:
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_levels: int = 4
    codebook_size: int = 1024
    use_rope: bool = True
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class RoPE(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class NCMBlock(nn.Module):
    def __init__(self, cfg: NCMConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)
        self.rope = RoPE(cfg.d_model // cfg.n_head, cfg.rope_theta) if cfg.use_rope else None

    def forward(self, x, mask=None):
        B, T, D = x.shape
        x_norm = self.ln1(x)
        if self.rope is not None:
            cos, sin = self.rope(x, T)
            # apply rope to q,k conceptually simplified
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class NCMTransformer(nn.Module):
    """Coarse-to-fine level predictor. Generates levels sequentially, parallel within level."""
    def __init__(self, cfg: NCMConfig, tok_cfg: NCMTokenizerConfig = None):
        super().__init__()
        self.cfg = cfg
        self.tok_cfg = tok_cfg or NCMTokenizerConfig()
        self.level_embeds = nn.ModuleList([nn.Embedding(cfg.codebook_size, cfg.d_model) for _ in range(cfg.max_levels)])
        self.level_pos = nn.ParameterList([nn.Parameter(torch.zeros(1, 512, cfg.d_model)) for _ in range(cfg.max_levels)])
        self.cum_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.blocks = nn.ModuleList([NCMBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.codebook_size) for _ in range(cfg.max_levels)])
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward_level(self, cum_repr: torch.Tensor, level: int, mask=None) -> torch.Tensor:
        # cum_repr conditioned on, predict level codes in parallel
        x = self.cum_proj(cum_repr)
        # add level embedding
        x = x + self.level_embeds[level](torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device))  # placeholder
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.heads[level](x)

    def forward(self, cum_reprs: List[torch.Tensor], targets: Optional[List[torch.Tensor]] = None):
        losses = []
        logits_all = []
        for k in range(min(len(cum_reprs), self.cfg.max_levels)):
            logits = self.forward_level(cum_reprs[k], k)
            logits_all.append(logits)
            if targets is not None and k < len(targets):
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[k].reshape(-1))
                losses.append(loss)
        if losses:
            return logits_all, sum(losses)
        return logits_all, None

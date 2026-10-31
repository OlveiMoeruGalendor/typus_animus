"""The recursive GPT: weight-shared blocks applied to themselves."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .modules import CausalSelfAttention, KVCache, RMSNorm, RotaryEmbedding, SwiGLU


class RecursiveBlock(nn.Module):
    """A transformer block that is applied to its own output.

    The block owns one set of weights and reuses it for `n_recursions` steps,
    so depth is bought by recursion instead of by stacking distinct layers.
    Every recursion step carries its own gate and its own normalisation.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_recursions = cfg.n_recursions
        self.ln_attn = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, dropout=cfg.dropout)
        self.ln_mlp = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg, dropout=cfg.dropout)
        self.gates = nn.Parameter(torch.ones(cfg.n_recursions))
        self.step_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_recursions)])
        self.halt_head = nn.Linear(cfg.d_model, 1) if cfg.adaptive_recursion else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: Optional[KVCache] = None,
        recursion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        total = self.n_recursions if recursion_steps is None else max(1, recursion_steps)
        for step in range(total):
            idx = step % self.n_recursions
            attn_out = self.attn(self.ln_attn(x), cos, sin, cache, layer, step)
            mlp_out = self.mlp(self.ln_mlp(x))
            weight = 1.0
            if self.halt_head is not None:
                stop_prob = torch.sigmoid(self.halt_head(x)).mean()
                weight = 1.0 - stop_prob
            update = self.step_norms[idx](attn_out + mlp_out)
            x = x + self.gates[idx] * weight * update
            if self.halt_head is not None and weight < self.cfg.halt_threshold and step < total - 1:
                break
        return x


class RecursiveGPT(nn.Module):
    """Decoder-only GPT whose depth comes from recursion rather than depth."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.blocks = nn.ModuleList([RecursiveBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        cache: Optional[KVCache] = None,
        position_offset: Optional[int] = None,
        recursion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        batch, seq_len = idx.shape
        offset = position_offset if position_offset is not None else (cache.seq_len if cache is not None else 0)
        x = self.tok_emb(idx)
        cos, sin = self.rope(seq_len, idx.device, x.dtype, offset=offset)
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, layer, cache, recursion_steps)
        x = self.ln_f(x)
        return self.lm_head(x)

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        if new_vocab_size == self.cfg.vocab_size:
            return
        old_emb = self.tok_emb
        new_emb = nn.Embedding(new_vocab_size, self.cfg.d_model)
        nn.init.normal_(new_emb.weight, mean=0.0, std=self.cfg.init_std)
        keep = min(old_emb.weight.shape[0], new_vocab_size)
        with torch.no_grad():
            new_emb.weight[:keep] = old_emb.weight[:keep]
        self.tok_emb = new_emb
        if self.cfg.tie_embeddings:
            self.lm_head.weight = new_emb.weight
        else:
            old_head = self.lm_head
            new_head = nn.Linear(self.cfg.d_model, new_vocab_size, bias=False)
            nn.init.normal_(new_head.weight, mean=0.0, std=self.cfg.init_std)
            with torch.no_grad():
                keep = min(old_head.weight.shape[0], new_vocab_size)
                new_head.weight[:keep] = old_head.weight[:keep]
                if old_head.bias is not None:
                    new_head.bias[:keep] = old_head.bias[:keep]
            self.lm_head = new_head
        self.cfg.vocab_size = new_vocab_size

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def describe(self) -> str:
        cfg = self.cfg
        lines = [
            f"RecursiveGPT d_model={cfg.d_model} n_layer={cfg.n_layer} n_head={cfg.n_head} "
            f"n_recursions={cfg.n_recursions} vocab={cfg.vocab_size}",
            f"parameters: {self.num_parameters():,}",
            f"effective depth: {cfg.n_layer * cfg.n_recursions} recursive applications",
        ]
        return "\n".join(lines)


def build_model(cfg: ModelConfig) -> RecursiveGPT:
    return RecursiveGPT(cfg)


def estimate_flops(cfg: ModelConfig, tokens: int) -> float:
    """Rough forward-pass FLOPs for a token count."""
    per_token = 12 * cfg.n_layer * cfg.d_model * cfg.d_model + 12 * cfg.n_layer * cfg.d_model * cfg.d_ff
    return float(per_token * cfg.n_recursions * tokens)


# Alias
TypusAnimus = RecursiveGPT

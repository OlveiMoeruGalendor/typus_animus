"""Multiscale Residual VQ-VAE tokenizer for Next-Concept Modeling.
Coarse-to-fine discrete code hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NCMTokenizerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 6
    n_levels: int = 4
    codebook_size: int = 1024
    codebook_dim: int = 64
    level_lengths: Optional[List[int]] = None  # coarse to fine, last should be seq_len
    commitment_weight: float = 0.25
    dropout: float = 0.1


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: (B, T, D)
        inputs_flat = inputs.view(-1, self.embedding_dim)
        # distances
        distances = (torch.sum(inputs_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, encoding_indices.view(inputs.shape[:-1])


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.block(self.norm(x))


class DownUp(nn.Module):
    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.factor = factor
        if factor > 1:
            self.down = nn.Linear(dim * factor, dim)
            self.up = nn.Linear(dim, dim * factor)
        else:
            self.down = nn.Identity()
            self.up = nn.Identity()

    def downsample(self, x):
        if self.factor == 1:
            return x
        B, T, D = x.shape
        T2 = (T // self.factor) * self.factor
        x = x[:, :T2, :]
        x = x.view(B, T2 // self.factor, D * self.factor)
        return self.down(x)

    def upsample(self, x, T):
        if self.factor == 1:
            return x
        B, T2, D = x.shape
        x = self.up(x)
        x = x.view(B, T2 * self.factor, D)
        return x[:, :T, :]


class MultiscaleResidualTokenizer(nn.Module):
    """Multiscale residual VQ-VAE - encodes to coarse-to-fine hierarchy."""
    def __init__(self, cfg: NCMTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.level_lengths is None:
            # geometric progression coarse to fine
            factors = []
            rem = cfg.n_levels
            # build levels
            lengths = []
            t = 64  # start coarse
            for i in range(cfg.n_levels - 1):
                lengths.append(t)
                t = t * 2
            lengths.append(1024)  # fine = full length placeholder
            cfg.level_lengths = lengths
        self.level_lengths = cfg.level_lengths
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder_layers = nn.ModuleList([ResidualBlock(cfg.d_model, cfg.dropout) for _ in range(cfg.n_layers)])
        self.downups = nn.ModuleList([DownUp(cfg.d_model, max(1, cfg.level_lengths[i+1] // max(1, cfg.level_lengths[i]) if i+1 < len(cfg.level_lengths) else 1)) for i in range(cfg.n_levels)])
        # simpler: fixed ratios
        self.vqs = nn.ModuleList([VectorQuantizer(cfg.codebook_size, cfg.codebook_dim, cfg.commitment_weight) for _ in range(cfg.n_levels)])
        self.vq_proj_in = nn.ModuleList([nn.Linear(cfg.d_model, cfg.codebook_dim) for _ in range(cfg.n_levels)])
        self.vq_proj_out = nn.ModuleList([nn.Linear(cfg.codebook_dim, cfg.d_model) for _ in range(cfg.n_levels)])
        self.decoder_layers = nn.ModuleList([ResidualBlock(cfg.d_model, cfg.dropout) for _ in range(cfg.n_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def encode_hierarchy(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        B, T = x.shape
        h = self.embed(x)
        for layer in self.encoder_layers:
            h = layer(h)
        z = h
        codes_list = []
        vq_losses = []
        residual = z
        cum = torch.zeros_like(z)
        for k in range(self.cfg.n_levels):
            # downsample to level length
            target_t = self.level_lengths[k] if k < len(self.level_lengths) else T
            if target_t < T:
                # simple pooling downsample
                ratio = T // target_t
                if ratio > 1:
                    uk = residual[:, :target_t*ratio, :].reshape(B, target_t, ratio * residual.shape[-1])
                    uk = uk.mean(dim=2)
                else:
                    uk = residual
            else:
                uk = residual
            uk_in = self.vq_proj_in[k](uk)
            q, vq_loss, idx = self.vqs[k](uk_in)
            q = self.vq_proj_out[k](q)
            # upsample back
            if q.shape[1] != T:
                q_up = q.unsqueeze(2).repeat(1,1, ratio if 'ratio' in dir() else T//q.shape[1], 1).reshape(B, T, -1) if False else q.repeat_interleave(max(1,T//q.shape[1]), dim=1)[:, :T, :]
            else:
                q_up = q
            cum = cum + q_up
            residual = residual - q_up
            codes_list.append(idx)
            vq_losses.append(vq_loss)
        return codes_list, vq_losses, cum

    def decode(self, cum: torch.Tensor) -> torch.Tensor:
        h = cum
        for layer in self.decoder_layers:
            h = layer(h)
        return self.lm_head(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        codes, vq_losses, cum = self.encode_hierarchy(x)
        logits = self.decode(cum)
        loss = sum(vq_losses)
        return logits, loss

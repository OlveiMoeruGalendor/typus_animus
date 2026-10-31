"""CoE Router - routes queries to appropriate expert based on aspect/domain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterConfig:
    d_model: int = 512
    n_experts: int = 6
    hidden_dim: int = 256
    temperature: float = 1.0
    top_k: int = 1
    uncertainty_threshold: float = 0.5  # route to base if uncertain
    use_uncertainty: bool = True


class CoERouter(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.n_experts),
        )
        self.base_expert_idx = cfg.n_experts  # fallback

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden: (B, T, D) -> pool
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        probs = F.softmax(logits / self.cfg.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        # uncertainty routing
        if self.cfg.use_uncertainty:
            route_to_base = entropy > self.cfg.uncertainty_threshold
        else:
            route_to_base = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        if self.cfg.top_k == 1:
            expert_idx = torch.argmax(probs, dim=-1)
        else:
            expert_idx = torch.topk(probs, self.cfg.top_k, dim=-1).indices
        return expert_idx, probs, entropy

    def route(self, hidden: torch.Tensor):
        expert_idx, probs, entropy = self.forward(hidden)
        return expert_idx, probs, entropy

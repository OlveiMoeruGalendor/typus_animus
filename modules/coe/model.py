"""CoE Model - Composition of Experts with LLM/NCM mix."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .router import CoERouter, RouterConfig
from .expert import Expert, ExpertConfig


@dataclass
class CoEConfig:
    d_model: int = 512
    vocab_size: int = 32000
    n_experts: int = 8
    ncm_ratio: float = 0.25  # percentage of experts that are NCM
    router_config: Optional[RouterConfig] = None
    expert_aspects: List[str] = field(default_factory=lambda: [
        "reasoning", "coding", "creative", "factual", "emotional", 
        "conversational", "analytical", "general"
    ])
    use_streaming: bool = True
    stream_batch_size: int = 1


class CoEModel(nn.Module):
    def __init__(self, cfg: CoEConfig):
        super().__init__()
        self.cfg = cfg
        router_cfg = cfg.router_config or RouterConfig(d_model=cfg.d_model, n_experts=cfg.n_experts + 1)
        self.router = CoERouter(router_cfg)
        self.experts = nn.ModuleList()
        n_ncm = max(1, int(cfg.n_experts * cfg.ncm_ratio))
        for i in range(cfg.n_experts):
            aspect = cfg.expert_aspects[i % len(cfg.expert_aspects)]
            expert_type = "ncm" if i < n_ncm else "llm"
            ecfg = ExpertConfig(name=f"expert_{i}", expert_type=expert_type, 
                                d_model=cfg.d_model, vocab_size=cfg.vocab_size, aspect=aspect,
                                use_streaming=cfg.use_streaming)
            self.experts.append(Expert(ecfg))
        self.base_expert = Expert(ExpertConfig(name="base", expert_type="llm", 
                                                d_model=cfg.d_model, vocab_size=cfg.vocab_size, aspect="general"))
        self.shared_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embed.weight

    def forward(self, idx: torch.Tensor, route: bool = True):
        x = self.shared_embed(idx)
        if route:
            expert_idx, probs, entropy = self.router(x)
            # route to base if uncertain
            route_to_base = entropy > self.router.cfg.uncertainty_threshold
            # for now use first expert or base
            out = self.base_expert(x)
        else:
            out = self.base_expert(x)
        # adapt output shape
        if hasattr(out, 'shape') and out.shape[-1] == self.cfg.d_model:
            logits = self.lm_head(out)
            return logits
        return out

    def get_expert_info(self):
        info = []
        for i, exp in enumerate(self.experts):
            info.append({"idx": i, "type": exp.expert_type, "aspect": exp.aspect})
        info.append({"idx": -1, "type": "llm", "aspect": "general", "base": True})
        return info

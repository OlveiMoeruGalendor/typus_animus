"""CoE Expert - individual expert, can be LLM or NCM type."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn


@dataclass
class ExpertConfig:
    name: str = "expert"
    expert_type: Literal["llm", "ncm"] = "llm"
    d_model: int = 512
    n_layer: int = 4
    n_head: int = 8
    vocab_size: int = 32000
    aspect: str = "general"  # e.g., reasoning, coding, creative, factual, emotional, etc.
    trainable: bool = True
    use_streaming: bool = True


class Expert(nn.Module):
    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.aspect = cfg.aspect
        self.expert_type = cfg.expert_type
        # placeholder modules - will be replaced by actual implementations
        if cfg.expert_type == "llm":
            from ...core.model import RecursiveGPT
            from ...core.config import ModelConfig
            mcfg = ModelConfig(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head)
            self.model = RecursiveGPT(mcfg)
        else:  # ncm
            from ...core.ncm.model import NCMTransformer, NCMConfig
            ncfg = NCMConfig(d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head)
            self.model = NCMTransformer(ncfg)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

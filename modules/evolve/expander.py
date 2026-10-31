"""Model expander - grow model during training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import copy

import torch
import torch.nn as nn


@dataclass
class ExpansionConfig:
    expand_every_steps: int = 1000
    expand_factor_width: float = 1.25  # grow d_model
    expand_factor_depth: float = 1.0   # grow n_layer
    max_d_model: int = 2048
    max_n_layer: int = 32
    growth_method: str = "function_preserving"  # net2net style
    noise_scale: float = 0.01


class ModelExpander:
    def __init__(self, model: nn.Module, cfg: ExpansionConfig = None):
        self.model = model
        self.cfg = cfg or ExpansionConfig()
        self.expansion_count = 0
        self.history = []

    def can_expand(self, current_step: int, perf_ok: bool = True) -> bool:
        if current_step % self.cfg.expand_every_steps != 0:
            return False
        # don't expand if lagging too much
        return perf_ok

    def expand_width(self, module: nn.Module, factor: float = None) -> bool:
        factor = factor or self.cfg.expand_factor_width
        expanded = False
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear):
                if m.out_features >= self.cfg.max_d_model:
                    continue
                old_out, old_in = m.out_features, m.in_features
                new_out = min(int(old_out * factor), self.cfg.max_d_model)
                new_in = min(int(old_in * factor), self.cfg.max_d_model)
                if new_out <= old_out and new_in <= old_in:
                    continue
                # function preserving expansion (Net2Net style)
                new_m = nn.Linear(new_in, new_out, bias=m.bias is not None).to(m.weight.device)
                with torch.no_grad():
                    # copy old weights
                    new_m.weight[:old_out, :old_in] = m.weight
                    # replicate for new neurons
                    if new_out > old_out:
                        for j in range(old_out, new_out):
                            src = j % old_out
                            new_m.weight[j, :old_in] = m.weight[src] * (1 + torch.randn(old_in, device=m.weight.device) * self.cfg.noise_scale)
                    if new_in > old_in:
                        for j in range(old_in, new_in):
                            src = j % old_in
                            new_m.weight[:new_out, j] = new_m.weight[:new_out, src]
                    if m.bias is not None and new_out > old_out:
                        new_m.bias[:old_out] = m.bias
                        for j in range(old_out, new_out):
                            new_m.bias[j] = m.bias[j % old_out]
                # replace
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = module.get_submodule(parent_name) if parent_name else module
                setattr(parent, child_name, new_m)
                expanded = True
        return expanded

    def expand_depth(self, module: nn.Module) -> bool:
        # try to clone blocks
        for name, m in module.named_modules():
            if isinstance(m, nn.ModuleList):
                if len(m) >= self.cfg.max_n_layer:
                    continue
                # clone last block
                if len(m) > 0:
                    last = m[-1]
                    clone = copy.deepcopy(last)
                    with torch.no_grad():
                        for p in clone.parameters():
                            p.add_(torch.randn_like(p) * self.cfg.noise_scale)
                    m.append(clone)
                    self.expansion_count += 1
                    self.history.append({"type": "depth", "blocks": len(m), "step": self.expansion_count})
                    return True
        return False

    def expand(self, step: int = 0, perf_lagging: bool = False):
        if perf_lagging:
            return False
        w = self.expand_width(self.model, self.cfg.expand_factor_width)
        d = self.expand_depth(self.model)
        self.history.append({"step": step, "width": w, "depth": d})
        return w or d

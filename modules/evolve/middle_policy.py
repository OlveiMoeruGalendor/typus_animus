"""Middle policy repeater - repeat middle layers/policy multiple times."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import copy

import torch
import torch.nn as nn


@dataclass
class MiddlePolicyConfig:
    repeat_times: int = 3  # repeat middle policy 3x
    select_middle_ratio: float = 0.4  # middle 40% of layers
    share_weights: bool = True  # tie weights for repeated blocks
    interpolate: bool = True


class MiddlePolicyRepeater:
    def __init__(self, model: nn.Module, cfg: MiddlePolicyConfig = None):
        self.model = model
        self.cfg = cfg or MiddlePolicyConfig()
        self.repeated = False

    def find_middle_blocks(self, module: nn.Module) -> List:
        for name, m in module.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 3:
                n = len(m)
                start = int(n * (1 - self.cfg.select_middle_ratio) / 2)
                end = int(n * (1 + self.cfg.select_middle_ratio) / 2)
                return list(range(start, end)), name, m
        return [], None, None

    def repeat(self):
        indices, name, blocks = self.find_middle_blocks(self.model)
        if not indices or self.repeated:
            return False
        parent_name = '.'.join(name.split('.')[:-1])
        parent = self.model.get_submodule(parent_name) if parent_name else self.model
        new_blocks = list(blocks)
        middle_blocks = [blocks[i] for i in indices]
        for _ in range(self.cfg.repeat_times - 1):
            for mb in middle_blocks:
                if self.cfg.share_weights:
                    new_blocks.append(mb)  # same instance
                else:
                    clone = copy.deepcopy(mb)
                    new_blocks.append(clone)
        setattr(parent, name.split('.')[-1], nn.ModuleList(new_blocks))
        self.repeated = True
        return True

    def unrepeat(self):
        # restore original structure if needed
        self.repeated = False
        return True

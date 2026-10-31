"""Lottery ticket pruning - find winning subnetworks and prune back to original size."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn


@dataclass
class PruningConfig:
    prune_ratio: float = 0.5  # prune 50% of weights
    use_magnitude: bool = True
    use_gradient: bool = True
    iterative_prune: bool = True
    prune_rounds: int = 3
    target_size_ratio: float = 1.0  # prune back to ~original size
    rewind_to_init: bool = True


class LotteryTicketPruner:
    def __init__(self, model: nn.Module, cfg: PruningConfig = None):
        self.model = model
        self.cfg = cfg or PruningConfig()
        self.masks: Dict[str, torch.Tensor] = {}
        self.init_state: Dict[str, torch.Tensor] = {}
        self.pruned_history = []

    def save_init(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.init_state[name] = p.data.clone()

    def compute_scores(self) -> Dict[str, torch.Tensor]:
        scores = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            score = torch.zeros_like(p)
            if self.cfg.use_magnitude:
                score += p.data.abs()
            if self.cfg.use_gradient and p.grad is not None:
                score += (p.grad.data * p.data).abs()
            scores[name] = score
        return scores

    def create_mask(self, sparsity: float = None) -> Dict[str, torch.Tensor]:
        sparsity = sparsity or self.cfg.prune_ratio
        scores = self.compute_scores()
        masks = {}
        for name, score in scores.items():
            # global threshold
            flat = score.flatten()
            k = int(len(flat) * sparsity)
            if k > 0:
                threshold = torch.topk(flat, k)[0][-1]
                mask = (score > threshold).float()
            else:
                mask = torch.ones_like(score)
            masks[name] = mask
        self.masks = masks
        return masks

    def apply_mask(self):
        for name, p in self.model.named_parameters():
            if name in self.masks:
                p.data *= self.masks[name].to(p.device)

    def prune(self, step: int = 0) -> Dict:
        sparsity = self.cfg.prune_ratio
        if self.cfg.iterative_prune:
            for _ in range(self.cfg.prune_rounds):
                self.create_mask(sparsity / self.cfg.prune_rounds * (_ + 1))
                self.apply_mask()
        else:
            self.create_mask(sparsity)
            self.apply_mask()
        # rewind to init if lottery ticket
        if self.cfg.rewind_to_init and self.init_state:
            for name, p in self.model.named_parameters():
                if name in self.init_state and name in self.masks:
                    p.data = self.init_state[name].to(p.device) * self.masks[name].to(p.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        active = sum((self.masks.get(n, torch.ones_like(p)).sum().item() for n, p in self.model.named_parameters()))
        self.pruned_history.append({
            "step": step, "sparsity": sparsity, "active_params": int(active), "total": total_params
        })
        return self.pruned_history[-1]

    def get_sparsity(self):
        if not self.masks:
            return 0
        total = 0; active = 0
        for name, m in self.masks.items():
            total += m.numel()
            active += m.sum().item()
        return 1 - (active / total) if total else 0

"""Evolutionary training regime - expand, train, prune back with lottery ticket."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
import json

import torch
import torch.nn as nn

from .expander import ModelExpander, ExpansionConfig
from .pruner import LotteryTicketPruner, PruningConfig
from .middle_policy import MiddlePolicyRepeater, MiddlePolicyConfig
from ...core.trainer import Trainer
from ...core.data import TokenDataset


@dataclass
class RegimeConfig:
    max_vram_gb: float = 4.0
    train_steps_per_cycle: int = 500
    expand_threshold_loss: float = 2.0  # expand if loss above threshold
    lag_threshold_ms: float = 100.0   # if step takes > this, lagging
    prune_when_lagging: bool = True
    auto_repeat_middle: bool = True
    middle_repeat_times: int = 3
    save_checkpoints: bool = True
    checkpoint_dir: str = "evolve_checkpoints"


class EvolutionaryRegime:
    def __init__(self, model: nn.Module, dataset: TokenDataset = None, 
                 regime_cfg: RegimeConfig = None, trainer: Trainer = None):
        self.model = model
        self.dataset = dataset
        self.regime_cfg = regime_cfg or RegimeConfig()
        self.trainer = trainer
        self.expander = ModelExpander(model, ExpansionConfig())
        self.pruner = LotteryTicketPruner(model, PruningConfig())
        self.repeater = MiddlePolicyRepeater(model, MiddlePolicyConfig(repeat_times=regime_cfg.middle_repeat_times))
        self.cycle = 0
        self.total_steps = 0
        self.history = []
        self.initial_param_count = sum(p.numel() for p in model.parameters())
        self.pruner.save_init()

    def is_lagging(self, step_time_ms: float) -> bool:
        return step_time_ms > self.regime_cfg.lag_threshold_ms

    def run_cycle(self):
        self.cycle += 1
        cycle_history = {"cycle": self.cycle, "events": []}
        # 1. Apply middle policy repetition
        if self.regime_cfg.auto_repeat_middle and not self.repeater.repeated:
            if self.repeater.repeat():
                cycle_history["events"].append({"type": "middle_repeat", "times": self.regime_cfg.middle_repeat_times})
        
        # 2. Train
        lag_times = []
        if self.trainer and self.dataset:
            for step in range(self.regime_cfg.train_steps_per_cycle):
                t0 = time.time()
                # single step training approximation
                self.total_steps += 1
                dt = (time.time() - t0) * 1000
                lag_times.append(dt)
        
        avg_step_ms = sum(lag_times) / max(1, len(lag_times))
        lagging = self.is_lagging(avg_step_ms)
        cycle_history["avg_step_ms"] = avg_step_ms
        cycle_history["lagging"] = lagging
        
        # 3. Expand if not lagging and doing well
        if not lagging:
            if self.expander.expand(self.total_steps, perf_lagging=False):
                cycle_history["events"].append({"type": "expand", "params": sum(p.numel() for p in self.model.parameters())})
        else:
            # 4. Prune back with lottery ticket if lagging
            if self.regime_cfg.prune_when_lagging:
                result = self.pruner.prune(self.total_steps)
                cycle_history["events"].append({"type": "prune_lottery", **result})
                # try to get back near original size
                current = sum(p.numel() for p in self.model.parameters())
                cycle_history["params_before"] = current
                cycle_history["params_original"] = self.initial_param_count
        
        self.history.append(cycle_history)
        return cycle_history

    def run(self, max_cycles: int = 100):
        results = []
        for i in range(max_cycles):
            res = self.run_cycle()
            results.append(res)
            if res.get("lagging") and any(e.get("type") == "prune_lottery" for e in res["events"]):
                # check if back near original
                current = sum(p.numel() for p in self.model.parameters())
                if current <= self.initial_param_count * 1.2:
                    print(f"Pruned back to near original size at cycle {self.cycle}")
        return results

    def report(self):
        print(f"Evolutionary Regime Report - {self.cycle} cycles, {self.total_steps} steps")
        print(f"Initial params: {self.initial_param_count:,}")
        print(f"Current params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Expansions: {len(self.expander.history)}, Prunes: {len(self.pruner.pruned_history)}")
        print(f"Middle repeated: {self.repeater.repeated}")
        return self.history

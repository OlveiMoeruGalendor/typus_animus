"""Training loop. Ready for a language, but ships without one."""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import TokenDataset
from .model import RecursiveGPT


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


class Trainer:
    def __init__(self, model: RecursiveGPT, cfg: Optional[TrainConfig] = None) -> None:
        self.cfg = cfg or TrainConfig()
        self.device = resolve_device(self.cfg.device)
        self.dtype = resolve_dtype(self.cfg.dtype)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
            eps=1e-8,
        )
        self.step = 0

    def lr_at(self, step: int, total_steps: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)

    def loss_for(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.dtype == torch.float32:
            logits = self.model(x)
        else:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                logits = self.model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))

    def train(
        self,
        dataset: TokenDataset,
        epochs: Optional[int] = None,
        on_step: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        if dataset.is_empty:
            raise ValueError(
                "dataset is empty: provide a language corpus before training "
                "(python -m recursive_gpt prepare --corpus <path>)"
            )
        cfg = self.cfg
        epochs = epochs if epochs is not None else cfg.epochs
        generator = torch.Generator().manual_seed(cfg.seed)
        total_steps = max(1, math.ceil(len(dataset) / cfg.batch_size) * epochs)
        history: list[dict] = []
        model = self.model
        model.train()
        start_time = time.time()
        for epoch in range(epochs):
            for step, (x, y) in enumerate(dataset.iter_batches(cfg.batch_size, shuffle=True, generator=generator)):
                lr = self.lr_at(self.step, total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.loss_for(x, y) / cfg.grad_accum
                loss.backward()
                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                if self.step % cfg.log_every == 0:
                    record = {
                        "epoch": epoch,
                        "step": self.step,
                        "loss": float(loss.item()) * cfg.grad_accum,
                        "lr": lr,
                        "elapsed": time.time() - start_time,
                    }
                    history.append(record)
                    if on_step is not None:
                        on_step(record)
                if cfg.eval_every and self.step % cfg.eval_every == 0:
                    eval_loss = self.evaluate(dataset)
                    record = {"epoch": epoch, "step": self.step, "eval_loss": eval_loss, "elapsed": time.time() - start_time}
                    history.append(record)
                    if on_step is not None:
                        on_step(record)
        model.eval()
        return {"history": history, "steps": self.step}

    def evaluate(self, dataset: TokenDataset) -> float:
        if dataset.is_empty:
            return float("nan")
        self.model.eval()
        generator = torch.Generator().manual_seed(self.cfg.seed)
        losses: list[float] = []
        with torch.no_grad():
            for index, (x, y) in enumerate(dataset.iter_batches(self.cfg.batch_size, shuffle=False, generator=generator)):
                if index >= self.cfg.eval_batches:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                losses.append(float(self.loss_for(x, y).item()))
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.cfg.checkpoint_dir, f"step_{self.step:06d}.pt")
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.model.cfg.to_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.step = payload.get("step", 0)


def load_model(path: str, device: str = "auto") -> RecursiveGPT:
    payload = torch.load(path, map_location=resolve_device(device), weights_only=False)
    from .config import ModelConfig

    cfg = ModelConfig.from_dict(payload["config"])
    model = RecursiveGPT(cfg)
    model.load_state_dict(payload["model"])
    return model.to(resolve_device(device))

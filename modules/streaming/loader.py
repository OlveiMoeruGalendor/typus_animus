"""Layer-wise loading like AirLLM - load parts one by one, not all at once."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn as nn


@dataclass
class StreamingConfig:
    device: str = "auto"
    max_vram_gb: float = 4.0  # target VRAM budget
    prefetch: bool = True
    pin_memory: bool = True
    use_mmap: bool = True
    compression: Optional[str] = None  # 4bit, 8bit
    layer_timeout: float = 30.0
    keep_embeddings_on_gpu: bool = True
    keep_lm_head_on_gpu: bool = True


class LayerStreamer:
    """
    AirLLM-style streaming: load model layers one by one into VRAM.
    Allows running large models (12B+) in small VRAM (4GB) by keeping only
    current layer + embeddings + head in memory.
    """
    def __init__(self, model: nn.Module, checkpoint_path: str = None, cfg: StreamingConfig = None):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.cfg = cfg or StreamingConfig()
        self.device = self._resolve_device()
        self.loaded_layers: Dict[str, torch.Tensor] = {}
        self.layer_order: List[str] = []
        self.hooks = []
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1) if cfg.prefetch else None
        self.prefetch_future = None
        self.lock = threading.Lock()
        self.vram_used = 0
        self.stats = {"loads": 0, "evicts": 0, "bytes_moved": 0}

    def _resolve_device(self):
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def split_model(self, layer_prefix: str = "blocks", embed_name: str = "tok_emb", 
                    norm_name: str = "ln_f", head_name: str = "lm_head"):
        """Identify layer structure for streaming."""
        self.layer_order = []
        for name, module in self.model.named_modules():
            if layer_prefix in name and name.count('.') <= 2:  # top-level layers
                self.layer_order.append(name)
        self.layer_order = sorted(set(self.layer_order))
        self.embed_name = embed_name
        self.norm_name = norm_name
        self.head_name = head_name
        return self.layer_order

    def load_shard(self, name: str) -> Dict[str, torch.Tensor]:
        if self.checkpoint_path is None:
            return {}
        shard_path = Path(self.checkpoint_path) / f"{name.replace('.', '_')}.pt"
        if shard_path.exists():
            sd = torch.load(shard_path, map_location='cpu')
            self.stats["loads"] += 1
            return sd
        # try safetensors
        try:
            from safetensors.torch import load_file
            st_path = Path(self.checkpoint_path) / f"{name.replace('.', '_')}.safetensors"
            if st_path.exists():
                sd = load_file(st_path)
                self.stats["loads"] += 1
                return sd
        except:
            pass
        return {}

    def evict_layer(self, name: str):
        for pname, param in self.model.named_parameters():
            if name in pname:
                param.data = param.data.to('meta')
        for bname, buf in self.model.named_buffers():
            if name in bname:
                buf.data = buf.data.to('meta')
        self.stats["evicts"] += 1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def stream_layer(self, name: str, state_dict: Dict[str, torch.Tensor]):
        for pname, param in self.model.named_parameters():
            if pname in state_dict:
                param.data = state_dict[pname].to(self.device)
                self.stats["bytes_moved"] += param.data.numel() * param.data.element_size()
        return True

    def install_hooks(self):
        """Install forward hooks for automatic streaming."""
        def make_pre_hook(layer_name):
            def hook(module, input):
                sd = self.load_shard(layer_name)
                self.stream_layer(layer_name, sd)
                return input
            return hook

        def make_post_hook(layer_name):
            def hook(module, input, output):
                self.evict_layer(layer_name)
                return output
            return hook

        for i, lname in enumerate(self.layer_order):
            # find module
            module = dict(self.model.named_modules()).get(lname)
            if module is not None:
                h1 = module.register_forward_pre_hook(make_pre_hook(lname))
                h2 = module.register_forward_hook(make_post_hook(lname))
                self.hooks.extend([h1, h2])
        return len(self.hooks)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def save_shards(self, output_dir: str):
        """Save model split into layer shards for streaming."""
        os.makedirs(output_dir, exist_ok=True)
        for name, module in self.model.named_modules():
            if any(x in name for x in ['block', 'layer']) and '.' not in name.split('block')[-1].split('layer')[-1].replace(name.split('.')[-1],''):
                pass
        # save by layer order
        for i, lname in enumerate(self.layer_order):
            sd = {}
            module = dict(self.model.named_modules()).get(lname)
            if module:
                for pname, param in module.named_parameters():
                    sd[pname.split('.')[-1] if False else pname] = param.cpu()
                for bname, buf in module.named_buffers():
                    sd[bname] = buf.cpu()
            torch.save(sd, os.path.join(output_dir, f"layer_{i:04d}.pt"))
        print(f"saved {len(self.layer_order)} layer shards to {output_dir}")

    def get_vram_estimate(self):
        total = 0
        for _, p in self.model.named_parameters():
            if p.device.type != 'meta':
                total += p.numel() * p.element_size()
        return total / (1024**3)

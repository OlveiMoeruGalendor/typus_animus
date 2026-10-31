"""VRAM Manager for layer-wise loading."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class VRAMStats:
    total_gb: float = 0
    free_gb: float = 0
    used_gb: float = 0
    reserved_gb: float = 0


class VRAMManager:
    def __init__(self, max_vram_gb: float = 4.0):
        self.max_vram_gb = max_vram_gb
        self.current_usage = 0

    def get_stats(self) -> VRAMStats:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            free = total - allocated
            return VRAMStats(total_gb=total, free_gb=free, used_gb=allocated, reserved_gb=reserved)
        return VRAMStats(total_gb=self.max_vram_gb, free_gb=self.max_vram_gb, used_gb=0, reserved_gb=0)

    def can_load(self, size_gb: float) -> bool:
        stats = self.get_stats()
        return stats.free_gb >= size_gb * 1.1  # 10% headroom

    def estimate_model_fit(self, param_count: int, bits: int = 16) -> dict:
        gb = (param_count * bits / 8) / (1024**3)
        stats = self.get_stats()
        return {
            "model_gb_full": gb,
            "available_gb": stats.total_gb,
            "fits_full": gb <= stats.total_gb * 0.9,
            "fits_streaming": gb / 100 <= self.max_vram_gb,  # rough estimate
            "recommended": "streaming" if gb > stats.total_gb * 0.5 else "direct"
        }

    def clear(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

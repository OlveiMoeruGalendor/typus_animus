"""Recursive generation.

Two engines are provided:

* `generate` - plain autoregressive decoding with a key/value cache.
* `generate_recursive` - multi-resolution generation. A coarse pass drafts the
  segment, the draft is refined by recursive re-sampling passes over
  progressively finer lattices, and a final autoregressive pass realises the
  result while staying anchored to the refined draft.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .config import GenerationConfig
from .model import RecursiveGPT
from .modules import KVCache
from .tokenizer import Tokenizer


class Generator:
    def __init__(self, model: RecursiveGPT, tokenizer: Optional[Tokenizer] = None, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def _filter_logits(self, logits: torch.Tensor, cfg: GenerationConfig, generated: Sequence[int]) -> torch.Tensor:
        logits = logits.clone()
        if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
            for token in set(generated):
                value = logits[token]
                logits[token] = value / cfg.repetition_penalty if value > 0 else value * cfg.repetition_penalty
        if cfg.temperature <= 0:
            return logits
        logits = logits / cfg.temperature
        if cfg.top_k is not None and cfg.top_k > 0:
            k = min(cfg.top_k, logits.shape[-1])
            threshold = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if cfg.top_p is not None and 0 < cfg.top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > cfg.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
        return logits

    def _sample(self, logits: torch.Tensor, cfg: GenerationConfig, generator: Optional[torch.Generator]) -> torch.Tensor:
        if cfg.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    def _seed_generator(self, cfg: GenerationConfig) -> Optional[torch.Generator]:
        if cfg.seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.seed)
        return generator

    def generate(
        self,
        prompt: Sequence[int],
        cfg: Optional[GenerationConfig] = None,
        forced: Optional[dict[int, int]] = None,
    ) -> list[int]:
        """Autoregressive decoding. `forced` maps segment-relative index -> token id."""
        cfg = cfg or GenerationConfig()
        generator = self._seed_generator(cfg)
        cache = KVCache()
        eos_id = self.tokenizer.eos_id if self.tokenizer is not None else None
        produced: list[int] = []
        with torch.no_grad():
            next_logits = None
            if len(prompt) > 0:
                tensor = torch.tensor(list(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.model(tensor, cache=cache, recursion_steps=cfg.recursion_steps)
                next_logits = logits[:, -1, :].squeeze(0)
            else:
                next_logits = torch.zeros(self.model.cfg.vocab_size, device=self.device)
            for step in range(cfg.max_new_tokens):
                filtered = self._filter_logits(next_logits, cfg, produced)
                token = int(self._sample(filtered, cfg, generator).item())
                if forced and step in forced:
                    token = int(forced[step])
                produced.append(token)
                if eos_id is not None and token == eos_id and step + 1 < cfg.max_new_tokens:
                    break
                tensor = torch.tensor([[token]], dtype=torch.long, device=self.device)
                logits = self.model(tensor, cache=cache, recursion_steps=cfg.recursion_steps)
                next_logits = logits[:, -1, :].squeeze(0)
        return produced

    def generate_recursive(
        self,
        prompt: Sequence[int],
        cfg: Optional[GenerationConfig] = None,
    ) -> list[int]:
        """Draft coarse, refine recursively, then realise anchored to the draft."""
        cfg = cfg or GenerationConfig()
        if cfg.max_new_tokens <= 0:
            return []
        draft = self.generate(prompt, cfg)
        if not draft:
            return []
        full = list(prompt) + draft
        refined, refined_positions = self._refine(full, len(prompt), cfg)
        segment = refined[len(prompt):]
        if not cfg.realize or not refined_positions:
            return segment
        forced = {j - len(prompt): token for j, token in enumerate(refined) if j in refined_positions}
        return self.generate(prompt, cfg, forced=forced)

    def _refine(self, full: Sequence[int], prompt_len: int, cfg: GenerationConfig) -> tuple[list[int], set[int]]:
        """Re-sample the draft on a hierarchy of lattices, coarse lattice first."""
        current = list(full)
        refined: set[int] = set()
        depth = max(0, cfg.recursion_depth)
        granularity = max(1, cfg.granularity)
        for level in range(depth):
            spacing = max(1, granularity >> level)
            positions = [
                prompt_len + j
                for j in range(len(current) - prompt_len)
                if j % spacing == 0 and (prompt_len + j) not in refined
            ]
            if not positions:
                continue
            current, _ = self._resample_positions(current, positions, cfg, self.mask_id)
            refined.update(positions)
        return current, refined

    def _resample_positions(
        self,
        ids: Sequence[int],
        positions: Sequence[int],
        cfg: GenerationConfig,
        mask_id: int,
    ) -> tuple[list[int], float]:
        branches = max(1, cfg.num_branches)
        best_ids: list[int] = list(ids)
        best_score = float("-inf")
        for _ in range(branches):
            candidate, score = self._resample_once(ids, positions, cfg, mask_id)
            if score > best_score:
                best_score = score
                best_ids = candidate
        return best_ids, best_score

    def _resample_once(
        self,
        ids: Sequence[int],
        positions: Sequence[int],
        cfg: GenerationConfig,
        mask_id: int,
    ) -> tuple[list[int], float]:
        generator = self._seed_generator(cfg)
        out = list(ids)
        for p in positions:
            out[p] = mask_id
        if cfg.refill == "sequential":
            total = 0.0
            count = 0
            for p in positions:
                prefix = torch.tensor([out[:p]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(prefix, recursion_steps=cfg.recursion_steps)[0, -1, :]
                filtered = self._filter_logits(logits, cfg, [])
                token = int(self._sample(filtered, cfg, generator).item())
                out[p] = token
                total += float(_logprob(logits, token))
                count += 1
            return out, total / max(1, count)
        tensor = torch.tensor([out], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor, recursion_steps=cfg.recursion_steps)[0]
        total = 0.0
        for p in positions:
            filtered = self._filter_logits(logits[p - 1], cfg, [])
            token = int(self._sample(filtered, cfg, generator).item())
            out[p] = token
            total += float(_logprob(logits[p - 1], token))
        return out, total / max(1, len(positions))

    @property
    def mask_id(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.mask_id
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(i) for i in ids)
        return self.tokenizer.decode(ids)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)


def _logprob(logits: torch.Tensor, token: int) -> float:
    return float(F.log_softmax(logits.float(), dim=-1)[token].item())


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return float(math.exp(loss.item()))

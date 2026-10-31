"""NCM Generator - coarse to fine sampling."""
from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn.functional as F

from .model import NCMTransformer


class NCMGenerator:
    def __init__(self, model: NCMTransformer, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt_codes_or_embeds=None, levels=4, temperature=1.0, top_k=50):
        self.model.eval()
        cum = None
        all_codes = []
        for k in range(levels):
            if cum is None:
                cum = torch.zeros(1, 64, self.model.cfg.d_model, device=self.device)
            logits = self.model.forward_level(cum, k)
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[..., [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                codes = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
            else:
                codes = logits.argmax(dim=-1)
            all_codes.append(codes)
            # update cum (simplified)
            emb = self.model.level_embeds[k](codes)
            if cum.shape[1] != emb.shape[1]:
                emb = emb.repeat_interleave(cum.shape[1] // emb.shape[1] + 1, dim=1)[:, :cum.shape[1], :]
            cum = cum + emb
        return all_codes

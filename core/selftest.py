"""Self test: exercises the network end to end without any real language."""

from __future__ import annotations

import torch

from .config import GenerationConfig, ModelConfig
from .data import TokenDataset
from .generation import Generator
from .model import RecursiveGPT
from .modules import KVCache
from .tokenizer import ByteTokenizer


def _tiny_config(vocab_size: int = 64) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, n_layer=2, n_head=2, d_model=32, n_recursions=3, max_seq_len=64)


def run_selftest() -> int:
    torch.manual_seed(0)
    tokenizer = ByteTokenizer()
    cfg = _tiny_config(tokenizer.vocab_size)
    model = RecursiveGPT(cfg)
    print(model.describe())

    ids = torch.randint(0, cfg.vocab_size, (2, 9))
    logits = model(ids)
    assert logits.shape == (2, 9, cfg.vocab_size), logits.shape
    print(f"forward ok: logits {tuple(logits.shape)}")

    target = torch.randint(0, cfg.vocab_size, (1, 6))
    full = model(target)[0]
    cache = KVCache()
    stepwise = []
    for t in range(target.shape[1]):
        out = model(target[:, t:t + 1], cache=cache)
        stepwise.append(out[0, -1])
    stepped = torch.stack(stepwise)
    drift = (full - stepped).abs().max().item()
    print(f"kv-cache drift: {drift:.2e}")
    assert drift < 1e-3, drift

    tokenizer = ByteTokenizer()
    generator = Generator(model, tokenizer)
    gen_cfg = GenerationConfig(max_new_tokens=8, temperature=0.8, seed=7, recursion_depth=2, granularity=2)
    ar = generator.generate([tokenizer.bos_id], gen_cfg)
    rec = generator.generate_recursive([tokenizer.bos_id], gen_cfg)
    print(f"ar tokens: {ar}")
    print(f"recursive tokens: {rec}")
    assert len(ar) == 8 and len(rec) == 8

    dataset = TokenDataset.from_texts(["ka tu vesh", "tu vesh ka", "vesh ka tu"], tokenizer, seq_len=4)
    print(f"dataset windows: {len(dataset)}")
    x, y = dataset.batch([0])
    loss = torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, cfg.vocab_size), y.reshape(-1))
    print(f"loss: {loss.item():.4f}")
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    deeper = model(ids, recursion_steps=cfg.n_recursions + 2)
    print(f"depth extrapolation ok: {tuple(deeper.shape)}")
    print("selftest passed")
    return 0

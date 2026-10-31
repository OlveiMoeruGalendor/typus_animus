"""Command line interface for the recursive GPT."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from .config import GenerationConfig, ModelConfig, TrainConfig
from .data import TokenDataset
from .generation import Generator
from .model import TypusAnimus
from .tokenizer import Tokenizer, build_tokenizer
from .trainer import Trainer, load_model, resolve_device


def read_corpus(paths: Sequence[str]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def cmd_info(args: argparse.Namespace) -> int:
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        n_recursions=args.n_recursions,
        max_seq_len=args.max_seq_len,
    )
    model = TypusAnimus(cfg)
    print(model.describe())
    print(f"neurons (d_model): {cfg.d_model}")
    print(f"vocabulary slots: {cfg.vocab_size} (use `prepare` to fit a language)")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    corpus = read_corpus(args.corpus)
    if not corpus:
        print("no text found in the given corpus files", file=sys.stderr)
        return 2
    tokenizer = build_tokenizer(args.kind, corpus, vocab_size=args.vocab_size)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    dataset = TokenDataset.from_texts(corpus, tokenizer, seq_len=args.seq_len)
    dataset_path = os.path.join(args.out_dir, "dataset.pt")
    dataset.save(dataset_path)
    print(f"tokenizer: {tokenizer_path} ({tokenizer.vocab_size} tokens)")
    print(f"dataset:   {dataset_path} ({len(dataset)} windows of {args.seq_len} tokens)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.load(args.tokenizer)
    dataset = TokenDataset.load(args.data)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        n_recursions=args.n_recursions,
        max_seq_len=args.max_seq_len,
    )
    model = TypusAnimus(cfg)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.out_dir,
        device=args.device,
        dtype=args.dtype,
    )
    trainer = Trainer(model, train_cfg)

    def report(record: dict) -> None:
        print(json.dumps(record), flush=True)

    result = trainer.train(dataset, epochs=args.epochs, on_step=report)
    path = os.path.join(args.out_dir, "model.pt")
    trainer.save_checkpoint(path)
    print(f"checkpoint: {path}")
    print(f"steps: {result['steps']}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model = load_model(args.model, device=str(device))
    tokenizer = Tokenizer.load(args.tokenizer) if args.tokenizer else None
    generator = Generator(model, tokenizer, device=device)
    prompt_ids = tokenizer.encode(args.prompt) if tokenizer else [int(t) for t in args.prompt.split()]
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        recursion_depth=args.recursion_depth,
        granularity=args.granularity,
        recursion_steps=args.recursion_steps,
    )
    if args.mode == "recursive":
        produced = generator.generate_recursive(prompt_ids, cfg)
    else:
        produced = generator.generate(prompt_ids, cfg)
    print(generator.decode(produced))
    if args.show_ids:
        print(produced)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    return run_selftest()


def cmd_coe_info(args: argparse.Namespace) -> int:
    from .coe import CoEModel, CoEConfig
    cfg = CoEConfig(n_experts=args.n_experts, ncm_ratio=args.ncm_ratio, d_model=args.d_model)
    model = CoEModel(cfg)
    info = model.get_expert_info()
    ncm_count = sum(1 for e in info if e.get('type') == 'ncm' and not e.get('base'))
    llm_count = sum(1 for e in info if e.get('type') == 'llm')
    print(f"CoE Model: {len(info)} components")
    print(f"Experts: {cfg.n_experts}, NCM: {ncm_count} ({args.ncm_ratio*100:.0f}%), LLM: {llm_count}")
    for e in info:
        print(f"  [{e['idx']}] {e['type'].upper()}: {e['aspect']}{' (base)' if e.get('base') else ''}")
    return 0


def cmd_ncm_info(args: argparse.Namespace) -> int:
    from .ncm import NCMTransformer, NCMConfig, MultiscaleResidualTokenizer, NCMTokenizerConfig
    ncfg = NCMConfig(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head)
    tcfg = NCMTokenizerConfig(d_model=args.d_model, n_levels=args.n_levels)
    model = NCMTransformer(ncfg)
    tok = MultiscaleResidualTokenizer(tcfg)
    print(model)
    print(f"NCM levels: {args.n_levels}, coarse-to-fine parallel generation")
    print(f"Tokenizer params: {sum(p.numel() for p in tok.parameters()):,}")
    return 0


def cmd_stream_info(args: argparse.Namespace) -> int:
    from .streaming import LayerStreamer, StreamingConfig, VRAMManager
    vm = VRAMManager(max_vram_gb=args.max_vram)
    stats = vm.get_stats()
    print(f"VRAM Manager - Max budget: {args.max_vram}GB")
    print(f"System: total={stats.total_gb:.1f}GB free={stats.free_gb:.1f}GB used={stats.used_gb:.1f}GB")
    print("Mode: AirLLM-style layer-wise streaming - load 1 layer at a time")
    print("Benefit: Run 12B+ models in 4GB VRAM")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    from .evolve import EvolutionaryRegime, RegimeConfig
    from .model import TypusAnimus
    from .config import ModelConfig
    cfg = ModelConfig(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head)
    model = TypusAnimus(cfg)
    regime_cfg = RegimeConfig(max_vram_gb=args.max_vram, train_steps_per_cycle=args.steps,
                               prune_when_lagging=True, auto_repeat_middle=True,
                               middle_repeat_times=args.repeat_middle)
    regime = EvolutionaryRegime(model, regime_cfg=regime_cfg)
    print(f"Initial params: {regime.initial_param_count:,}")
    print("Starting evolutionary regime...")
    results = regime.run(max_cycles=args.cycles)
    regime.report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typus_animus", description="Typus Animus")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show the model shape and parameter budget")
    info.add_argument("--d-model", type=int, default=512)
    info.add_argument("--n-layer", type=int, default=8)
    info.add_argument("--n-head", type=int, default=8)
    info.add_argument("--n-recursions", type=int, default=4)
    info.add_argument("--vocab-size", type=int, default=512)
    info.add_argument("--max-seq-len", type=int, default=1024)
    info.set_defaults(func=cmd_info)

    prepare = sub.add_parser("prepare", help="tokenise a language corpus")
    prepare.add_argument("--corpus", nargs="+", required=True)
    prepare.add_argument("--out-dir", default="data")
    prepare.add_argument("--kind", choices=["bpe", "word", "byte"], default="bpe")
    prepare.add_argument("--vocab-size", type=int, default=4096)
    prepare.add_argument("--seq-len", type=int, default=256)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="train the network on a prepared dataset")
    train.add_argument("--data", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--out-dir", default="runs")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--d-model", type=int, default=512)
    train.add_argument("--n-layer", type=int, default=8)
    train.add_argument("--n-head", type=int, default=8)
    train.add_argument("--n-recursions", type=int, default=4)
    train.add_argument("--max-seq-len", type=int, default=1024)
    train.add_argument("--device", default="auto")
    train.add_argument("--dtype", default="float32")
    train.set_defaults(func=cmd_train)

    generate = sub.add_parser("generate", help="sample from a checkpoint")
    generate.add_argument("--model", required=True)
    generate.add_argument("--tokenizer")
    generate.add_argument("--prompt", default="")
    generate.add_argument("--mode", choices=["ar", "recursive"], default="recursive")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--recursion-depth", type=int, default=2)
    generate.add_argument("--granularity", type=int, default=4)
    generate.add_argument("--recursion-steps", type=int, default=None)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--show-ids", action="store_true")
    generate.set_defaults(func=cmd_generate)

    selftest = sub.add_parser("selftest", help="run the built-in sanity checks")
    selftest.set_defaults(func=cmd_selftest)

    coe = sub.add_parser("coe", help="Composition of Experts (CoE) with LLM/NCM mix")
    coe.add_argument("--n-experts", type=int, default=8)
    coe.add_argument("--ncm-ratio", type=float, default=0.25)
    coe.add_argument("--d-model", type=int, default=512)
    coe.set_defaults(func=cmd_coe_info)

    ncm = sub.add_parser("ncm", help="Next-Concept Modeling info")
    ncm.add_argument("--d-model", type=int, default=512)
    ncm.add_argument("--n-layer", type=int, default=8)
    ncm.add_argument("--n-head", type=int, default=8)
    ncm.add_argument("--n-levels", type=int, default=4)
    ncm.set_defaults(func=cmd_ncm_info)

    stream = sub.add_parser("stream", help="AirLLM-style layer-wise streaming info")
    stream.add_argument("--max-vram", type=float, default=4.0)
    stream.set_defaults(func=cmd_stream_info)

    evolve = sub.add_parser("evolve", help="Evolutionary training: expand, train, prune with lottery ticket, repeat middle policy")
    evolve.add_argument("--d-model", type=int, default=256)
    evolve.add_argument("--n-layer", type=int, default=4)
    evolve.add_argument("--n-head", type=int, default=4)
    evolve.add_argument("--max-vram", type=float, default=4.0)
    evolve.add_argument("--steps", type=int, default=100)
    evolve.add_argument("--cycles", type=int, default=5)
    evolve.add_argument("--repeat-middle", type=int, default=3)
    evolve.set_defaults(func=cmd_evolve)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

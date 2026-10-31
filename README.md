# Typus Animus

**Version:** 0.3.1 | **Status:** Experimental (living prototype)

**Project:** Anima AI, the first step toward *ANIMA AETERNA*

Typus Animus is a modular language model framework exploring recursive depth, Next-Concept Modeling (NCM), Composition of Experts (CoE), AirLLM-style layer-wise streaming, and evolutionary training. It is language-agnostic: the network consumes integer token ids, and the tokenizer is the only place a language needs to be plugged in.

## Context

- **Anima AI:** a research project exploring modular, adaptive neural systems and pathways toward persistent, evolving intelligence.
- **ANIMA AETERNA:** the long-term goal of a persistent, evolving model system (self-modifying architecture, continual growth and pruning). Currently a guiding vision, not implemented.
- **Typus Animus:** Latin-inspired, roughly "living type/form". The name of this evolving prototype.

## Quickstart

```bash
pip install -e .
python -m typus_animus selftest
python -m typus_animus prepare --corpus data/corpus.txt --out-dir data --kind bpe --vocab-size 4096
python -m typus_animus train --data data/dataset.pt --tokenizer data/tokenizer.json --out-dir runs --epochs 1 --d-model 256 --n-layer 4 --n-recursions 4
python -m typus_animus generate --model runs/model.pt --tokenizer data/tokenizer.json --prompt "hello" --mode ar
```

## Installation and dependencies

```bash
git clone <repo-url>
cd typus_animus
pip install -e .
pip install "torch>=2.1"   # skip if torch is already declared in pyproject.toml
```

- **Python:** >= 3.10
- **PyTorch:** >= 2.1. The default shape (about 26M parameters) trains on CPU. CUDA is recommended for larger training and inference.
- **License:** MIT ([LICENSE](LICENSE))

## Module status

| Module | Status | Notes |
| --- | --- | --- |
| `core/model`, `core/modules`, `core/generation` | **Stable** | Recursive core. Covered by tests. |
| `core/tokenizer`, `core/data`, `core/trainer`, `core/config` | **Stable** | Standard pipeline. Covered by tests. |
| `core/draekonic_tokenizer`, `tokendict.json` | **Experimental** | Draekonic-specific. See [docs/draekonic.md](docs/draekonic.md). |
| `modules/ncm` | **Experimental** | Coarse-to-fine. API present; CLI prints config only. Limited training and evaluation. |
| `modules/coe` | **Experimental** | Router plus LLM/NCM experts with uncertainty routing. CLI prints config only. Training and execution in progress. |
| `modules/streaming` | **Experimental** | Layer-wise sharding and streaming. CLI prints config only. Integration evolving. |
| `modules/evolve` | **Experimental** | Grow, train, repeat middle, lottery-ticket prune. Lag timing is simulated in parts; checkpointing and masks evolving. |

## Layout

```
typus_animus/
  core/        # stable
  modules/     # experimental (ncm, coe, streaming, evolve)
  tests/       # core tests
  docs/        # docs
  tokendict.json
```

## 1. Recursive core (stable)

Each `RecursiveBlock` owns **one** set of weights (attention + MLP) and applies it to its own output `n_recursions` times. Each recursion step has its own gate and its own normalisation.

```
block(x):
    for step in 0 .. n_recursions:
        a = attn(ln_attn[step](x))      # attention weights shared across steps
        m = mlp(ln_mlp[step](x))        # MLP weights shared across steps
        x = x + gate[step] * rmsnorm(a + m)
```

With `n_layer = 8` and `n_recursions = 4`, the network performs 32 block applications while storing 8 blocks of parameters.

**What recursion does and does not save.** It reduces parameter count, not compute or activation memory. FLOPs scale with effective depth (32 in the example). The KV cache is keyed by `(layer, recursion step)`, so it scales with `n_layer * n_recursions` as well.

**Features**
- RoPE positions, computed on the fly for any sequence length
- Exact incremental decoding with a KV cache
- Tied input/output embeddings; `resize_token_embeddings` for late vocabulary changes
- Optional adaptive recursion with a halting head
- Test-time recursion depth can exceed training depth (`recursion_steps=`)

```python
from typus_animus import Generator, GenerationConfig, TypusAnimus, ModelConfig

model = TypusAnimus(ModelConfig(d_model=512, n_layer=8, n_head=8, n_recursions=4))
gen = Generator(model)
cfg = GenerationConfig(max_new_tokens=64, recursion_depth=2, granularity=4)
out = gen.generate_recursive([1], cfg)
```

### Recursive generation

`generate_recursive` runs three stages:

1. **Draft:** a normal autoregressive pass produces a coarse segment.
2. **Refine:** the draft is re-sampled over a hierarchy of position lattices, coarsest first. Tokens not yet refined are hidden behind `<mask>` so the model cannot copy the draft. With `num_branches > 1`, the highest-scoring candidates are kept.
3. **Realise:** a final autoregressive pass regenerates the text with the refined lattice points forced.

`recursion_depth` is the number of lattice levels, `granularity` is `[TODO: define]`, and `num_branches` is the number of candidates kept.

> **Caveat:** the refine stage depends on the model handling `<mask>` inputs. `[TODO: state whether train includes a masking objective. If it does not, say that refinement quality is untested on models trained with plain next-token loss.]`

## 2. NCM (experimental)

Multiscale residual VQ-VAE plus a level-parallel transformer. Text is encoded into a hierarchy of discrete codes, and generation runs coarse to fine, predicting all codes within a level in parallel (inspired by VAR).

```python
from typus_animus import NCMTransformer, NCMConfig

model = NCMTransformer(NCMConfig(d_model=512, n_layer=8, n_head=8))
```

**CLI:** `python -m typus_animus ncm` (prints config/info only)

**Training:** the tokenizer and transformer exist, but the full training recipe is in progress.

## 3. CoE (experimental)

Uncertainty-routed experts, a mix of LLM (`TypusAnimus`) and NCM experts, with a base fallback. The router is embedding-based and uses entropy as its uncertainty signal. Default: 8 aspect experts (reasoning, coding, creative, factual, emotional, conversational, analytical, general) at a 25% NCM ratio, plus a base fallback `[TODO: confirm total is 8 or 9]`.

```python
from typus_animus import CoEModel, CoEConfig

model = CoEModel(CoEConfig(n_experts=8, ncm_ratio=0.25))
```

**CLI:** `python -m typus_animus coe` (prints config/info only)

**Training:** the router and expert structure are present. Joint training and end-to-end execution are in progress.

**Open design questions:** LLM experts produce token logits and NCM experts produce level codes, so `[TODO: describe how outputs are made comparable or combined]`. Routing is per query, and `[TODO: describe how the entropy threshold is chosen or calibrated]`.

## 4. Streaming (experimental)

Layer-wise sharding and streaming for constrained VRAM. Layers are loaded on demand, prefetched, and evicted after use.

```python
from typus_animus import LayerStreamer, StreamingConfig

streamer = LayerStreamer(model, cfg=StreamingConfig(max_vram_gb=4.0))
```

**CLI:** `python -m typus_animus stream` (prints config/info only)

**Caveats**
- Throughput is bound by disk and PCIe bandwidth, so generation will be slow. No measurements yet.
- Activations and the KV cache also need VRAM. The budget must cover more than one layer's weights.
- Because recursive blocks reuse weights, a block's weights could be loaded once per pass and reused across its recursion steps. `[TODO: confirm whether the streamer does this.]`

## 5. Evolutionary regime (experimental)

Cycle: grow, train, repeat the middle policy, then prune when lagging.

1. **Train** the current model for N steps.
2. **Expand** (when step time is under the lag threshold) with Net2Net-style growth plus small noise. The noise makes this approximately, not exactly, function-preserving.
3. **Repeat the middle policy:** repeat the middle blocks N times, with shared or cloned weights.
4. **Monitor** step time. If it exceeds the threshold, trigger pruning.
5. **Prune** with a lottery-ticket procedure: score weights by magnitude plus gradient, prune iteratively, and rewind surviving weights to initial values under the mask.
6. **Repeat.**

**Current limitations**
- Lag timing is simulated in parts of the implementation, so "lag-monitored pruning" is not yet driven by real hardware measurements everywhere. Wall-clock step time is also noisy and hardware-dependent, so the threshold should be configurable `[TODO: confirm]`.
- Pruning is `[TODO: structured or unstructured]`. Unstructured masks reduce nonzero parameters but do not reduce compute or memory without sparse kernels.
- After expansion, the saved "initial weights" belong to a model of a different shape. `[TODO: state what is rewound to.]`

```python
from typus_animus import EvolutionaryRegime, RegimeConfig, TypusAnimus, ModelConfig

model = TypusAnimus(ModelConfig(d_model=256, n_layer=4, n_head=4))
regime = EvolutionaryRegime(model, RegimeConfig(
    train_steps_per_cycle=500, prune_when_lagging=True,
    auto_repeat_middle=True, middle_repeat_times=3))
regime.run(max_cycles=10)
```

**CLI:** `python -m typus_animus evolve`

## Using your own language

Any language works in three steps: tokenize, train, sample (see the Quickstart for the commands).

- `--kind bpe`: byte-level, lossless, learns merges
- `--kind word`: whitespace and punctuation
- `--kind byte`: fixed 256-slot vocabulary, no training needed

**Custom tokenizers:** subclass `typus_animus.tokenizer.Tokenizer` and implement `encode`, `decode`, `vocab_size`, and `save`/`load`. `LanguageSpec` in `config.py` is the placeholder record for the language owner.

## Language: Draekonic

`tokendict.json` is the dictionary of Draekonic, a constructed language designed for Anima AI. Entries are `word: [id, gloss]`, with category-based id ranges (roots 1000+, affixes 2000+, up to tense suffixes at 12000+). Tokenization is greedy longest-match. Full grammar and rules: [docs/draekonic.md](docs/draekonic.md).

The ids are sparse and reach 12000+, which is larger than the `--vocab-size 4096` used in the examples above. `[TODO: state whether core/draekonic_tokenizer remaps ids to a contiguous range, and how to use it from the CLI.]`

## CLI reference

| Command | Description |
| --- | --- |
| `info` | Model shape and parameter budget |
| `prepare` | Tokenize a corpus (byte / word / bpe) |
| `train` | Train TypusAnimus |
| `generate` | Sample (`ar` / `recursive`) |
| `selftest` | Sanity checks |
| `coe` | CoE config/info (**experimental**, prints config only) |
| `ncm` | NCM config/info (**experimental**, prints config only) |
| `stream` | Streaming config/info (**experimental**, prints config only) |
| `evolve` | Evolutionary regime (**experimental**) |

## Tests and benchmarks

```bash
python -m typus_animus selftest
python -m unittest discover -s tests -t .
```

- Core tests cover forward shapes, causality, KV-cache equivalence, recursion-depth extrapolation, tokenizer round trips, dataset packing, a training step, both generation engines, and the `prepare -> train -> generate` pipeline.
- **No dedicated tests for NCM, CoE, streaming, or evolve yet.**
- **No benchmarks or results published.** Toy runs only.

## Notes

- Vocabulary size is not fixed: build the tokenizer first and pass `tokenizer.vocab_size` to `ModelConfig`, or call `model.resize_token_embeddings(...)`.
- `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<mask>` are reserved. `<mask>` is used by the recursive refiner.
- `max_seq_len` is for bookkeeping only; RoPE positions are computed on demand.
- CoE and NCM share patterns but are independent modules.

## Prior work and credits

**Borrowed or inspired:** Universal Transformers / looped transformers and ALBERT (weight sharing); VAR (next-scale prediction); Net2Net (function-preserving growth); the Lottery Ticket Hypothesis (Frankle & Carbin); AirLLM (layer-wise streaming); GPT, RoPE, SwiGLU, RMSNorm.

**Novel in combination:** per-step gating and KV caching inside a recursive-depth block, within one framework; CoE mixing recursive LLM and NCM experts under uncertainty-based routing; and an evolutionary regime that chains growth, middle-policy repetition, and lag-monitored lottery-ticket pruning. The individual ingredients are established; the contribution is how they are combined.

## License

MIT. See [LICENSE](LICENSE).
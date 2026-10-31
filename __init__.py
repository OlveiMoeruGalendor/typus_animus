"""Typus Animus: a decoder-only language model built on recursive generation.

The package has four layers that can be used independently:

* `tokenizer` - turns any language into token ids (byte, word or byte-level BPE).
* `model` - `TypusAnimus`, a GPT-style decoder whose depth comes from
  weight-shared blocks that recurse on their own output.
* `generation` - `Generator`, offering autoregressive decoding and the
  recursive multi-resolution refinement decoder.
* `trainer` - training loop that stays empty until a language is supplied.

Also includes:
* `ncm` - Next-Concept Modeling (NCM) with coarse-to-fine hierarchical generation
* `coe` - Composition of Experts (CoE) routing across LLM and NCM experts
* `streaming` - AirLLM-style layer-wise VRAM streaming
* `evolve` - Evolutionary training with expansion, lottery ticket pruning, middle policy repetition
"""

from .core.config import GenerationConfig, LanguageSpec, ModelConfig, TrainConfig
from .core.data import TokenDataset
from .core.generation import Generator
from .core.model import RecursiveBlock, RecursiveGPT, TypusAnimus, build_model
from .core.tokenizer import BPETokenizer, ByteTokenizer, Tokenizer, WordTokenizer, build_tokenizer
from .core.draekonic_tokenizer import DraekonicTokenizer
from .core.trainer import Trainer, load_model

# New modules
from . import modules
from .modules import ncm, coe, streaming, evolve
from .modules.ncm import MultiscaleResidualTokenizer, NCMTokenizerConfig, NCMTransformer, NCMConfig, NCMGenerator
from .modules.coe import CoERouter, RouterConfig, Expert, ExpertConfig, CoEModel, CoEConfig
from .modules.streaming import LayerStreamer, StreamingConfig, VRAMManager
from .modules.evolve import ModelExpander, ExpansionConfig, LotteryTicketPruner, PruningConfig, EvolutionaryRegime, RegimeConfig, MiddlePolicyRepeater, MiddlePolicyConfig

__all__ = [
    "GenerationConfig",
    "LanguageSpec",
    "ModelConfig",
    "TrainConfig",
    "TokenDataset",
    "Generator",
    "RecursiveBlock",
    "TypusAnimus",
    "build_model",
    "BPETokenizer",
    "ByteTokenizer",
    "Tokenizer",
    "WordTokenizer",
    "DraekonicTokenizer",
    "build_tokenizer",
    "Trainer",
    "load_model",
    # ncm
    "ncm", "MultiscaleResidualTokenizer", "NCMTokenizerConfig", "NCMTransformer", "NCMConfig", "NCMGenerator",
    # coe
    "coe", "CoERouter", "RouterConfig", "Expert", "ExpertConfig", "CoEModel", "CoEConfig",
    # streaming
    "streaming", "LayerStreamer", "StreamingConfig", "VRAMManager",
    # evolve
    "evolve", "ModelExpander", "ExpansionConfig", "LotteryTicketPruner", "PruningConfig", "EvolutionaryRegime", "RegimeConfig", "MiddlePolicyRepeater", "MiddlePolicyConfig",
]

__version__ = "0.3.1"

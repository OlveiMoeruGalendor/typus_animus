"""Configuration objects for the recursive GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    vocab_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: Optional[int] = None
    n_recursions: int = 4
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    init_std: float = 0.02
    use_bias: bool = False
    adaptive_recursion: bool = False
    halt_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.d_ff is None:
            hidden = int(8 * self.d_model / 3)
            self.d_ff = max(64, ((hidden + 63) // 64) * 64)
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_recursions < 1:
            raise ValueError("n_recursions must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    recursion_depth: int = 2
    granularity: int = 4
    num_branches: int = 1
    realize: bool = True
    recursion_steps: Optional[int] = None
    refill: str = "sequential"


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 20
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 1337
    compile: bool = False


@dataclass
class LanguageSpec:
    """Describes a language to be tokenised; filled in by the language owner."""

    name: str = "unnamed"
    corpus_paths: list[str] = field(default_factory=list)
    tokenizer_kind: str = "bpe"
    vocab_size: int = 4096
    special_tokens: list[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"])

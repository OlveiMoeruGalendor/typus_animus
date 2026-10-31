"""Datasets. No language ships with this package; data arrives from outside."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import torch

from .tokenizer import Tokenizer


class TokenDataset:
    """Token sequences packed into fixed-length training windows."""

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        seq_len: int = 256,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.sequences = [list(s) for s in sequences if len(s) > 0]
        self.chunks: list[list[int]] = []
        self._pack()

    def _pack(self) -> None:
        stream: list[int] = []
        for seq in self.sequences:
            stream.extend(seq)
            if self.eos_id is not None:
                stream.append(self.eos_id)
        usable = (len(stream) // self.seq_len) * self.seq_len
        self.chunks = [stream[i:i + self.seq_len] for i in range(0, usable, self.seq_len)]

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.chunks[i] for i in indices]
        x = torch.tensor(rows, dtype=torch.long)
        y = torch.tensor([row[1:] + [row[-1]] for row in rows], dtype=torch.long)
        return x, y

    def iter_batches(self, batch_size: int, shuffle: bool = True, generator: Optional[torch.Generator] = None):
        order = torch.randperm(len(self.chunks), generator=generator) if shuffle else torch.arange(len(self.chunks))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size].tolist()
            if idx:
                yield self.batch(idx)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        tokenizer: Tokenizer,
        seq_len: int = 256,
    ) -> "TokenDataset":
        sequences = tokenizer.encode_batch(texts)
        return cls(sequences, seq_len=seq_len, eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save({"chunks": self.chunks, "seq_len": self.seq_len, "eos_id": self.eos_id}, path)

    @classmethod
    def load(cls, path: str) -> "TokenDataset":
        payload = torch.load(path, weights_only=False)
        dataset = cls.__new__(cls)
        dataset.seq_len = payload["seq_len"]
        dataset.eos_id = payload.get("eos_id")
        dataset.pad_id = None
        dataset.sequences = []
        dataset.chunks = payload["chunks"]
        return dataset

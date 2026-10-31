"""Tokenizers for arbitrary languages.

The network is language agnostic: it consumes integer token ids. This module
supplies the bridge between raw text and ids, and is the only place a new
language needs to be plugged in.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Iterable, Sequence

DEFAULT_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]


def build_tokenizer(kind: str, corpus: Sequence[str] = (), vocab_size: int = 4096, **kwargs):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "word":
        if corpus:
            return WordTokenizer.train(corpus, vocab_size=vocab_size)
        return WordTokenizer()
    if kind == "bpe":
        if corpus:
            return BPETokenizer.train(corpus, vocab_size=vocab_size)
        return BPETokenizer([], vocab_size=vocab_size)
    if kind == "draekonic":
        from .draekonic_tokenizer import DraekonicTokenizer
        return DraekonicTokenizer(**kwargs)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

GPT_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)
DEFAULT_SPLIT_RE = re.compile(GPT_SPLIT_PATTERN)


class Tokenizer:
    """Base class: vocab_size + encode/decode + json persistence."""

    kind = "base"

    def __init__(self, special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS)) -> None:
        self.special_tokens = list(special_tokens)
        self.special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.special_to_id.get("<pad>", 0)

    @property
    def unk_id(self) -> int:
        return self.special_to_id.get("<unk>", 1)

    @property
    def bos_id(self) -> int:
        return self.special_to_id.get("<bos>", 2)

    @property
    def eos_id(self) -> int:
        return self.special_to_id.get("<eos>", 3)

    @property
    def mask_id(self) -> int:
        return self.special_to_id.get("<mask>", 4)

    @property
    def n_special(self) -> int:
        return len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, sequences: Iterable[Sequence[int]]) -> list[str]:
        return [self.decode(seq) for seq in sequences]

    def state(self) -> dict:
        return {"kind": self.kind, "special_tokens": self.special_tokens}

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        kind = data.get("kind", "base")
        if kind == "byte":
            return ByteTokenizer.from_state(data)
        if kind == "word":
            return WordTokenizer.from_state(data)
        if kind == "bpe":
            return BPETokenizer.from_state(data)
        if kind == "draekonic":
            from .draekonic_tokenizer import DraekonicTokenizer
            return DraekonicTokenizer.from_state(data)
        raise ValueError(f"unknown tokenizer kind: {kind}")


class ByteTokenizer(Tokenizer):
    """Lossless byte-level tokenizer; accepts any language without training."""

    kind = "byte"

    def __init__(self, special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS)) -> None:
        super().__init__(special_tokens)
        self.byte_offset = self.n_special

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256

    def encode(self, text: str) -> list[int]:
        return [self.byte_offset + b for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytearray()
        for token in ids:
            if token < self.n_special:
                continue
            raw.append((token - self.byte_offset) & 0xFF)
        return raw.decode("utf-8", errors="replace")

    def state(self) -> dict:
        data = super().state()
        data["byte_offset"] = self.byte_offset
        return data

    @classmethod
    def from_state(cls, data: dict) -> "ByteTokenizer":
        return cls(data.get("special_tokens", DEFAULT_SPECIALS))


class WordTokenizer(Tokenizer):
    """Whitespace/punctuation tokenizer for languages with clear word forms."""

    kind = "word"

    def __init__(self, special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS), words: Sequence[str] = ()) -> None:
        super().__init__(special_tokens)
        self.words = list(words)
        self.word_to_id = {word: self.n_special + i for i, word in enumerate(self.words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return self.n_special + len(self.words)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in DEFAULT_SPLIT_RE.findall(text):
            token = self.word_to_id.get(piece)
            if token is None:
                ids.extend(self.fallback_ids(piece))
            else:
                ids.append(token)
        return ids

    def fallback_ids(self, piece: str) -> list[int]:
        return [self.unk_id]

    def decode(self, ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for token in ids:
            if token < self.n_special:
                continue
            piece = self.id_to_word.get(token)
            if piece is not None:
                pieces.append(piece)
        return "".join(pieces)

    @classmethod
    def train(cls, texts: Iterable[str], special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS)) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(DEFAULT_SPLIT_RE.findall(text))
        return cls(special_tokens, sorted(counter))

    def state(self) -> dict:
        data = super().state()
        data["words"] = self.words
        return data

    @classmethod
    def from_state(cls, data: dict) -> "WordTokenizer":
        return cls(data.get("special_tokens", DEFAULT_SPECIALS), data.get("words", []))


class BPETokenizer(Tokenizer):
    """Byte-level BPE, trained on a corpus of the target language."""

    kind = "bpe"

    def __init__(
        self,
        special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS),
        merges: Sequence[tuple[str, str]] = (),
        vocab: Sequence[str] = (),
    ) -> None:
        super().__init__(special_tokens)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = list(vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self._cache: dict[str, list[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS),
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        word_freq: Counter[str] = Counter()
        for text in texts:
            for piece in DEFAULT_SPLIT_RE.findall(text):
                word_freq[piece] += 1

        pieces: dict[str, list[str]] = {}
        weights: dict[str, int] = {}
        for word, freq in word_freq.items():
            if freq < min_frequency:
                continue
            encoded = [BYTE_ENCODER[b] for b in word.encode("utf-8")]
            if not encoded:
                continue
            pieces[word] = encoded
            weights[word] = freq

        merges: list[tuple[str, str]] = []
        n_merges = max(0, vocab_size - len(special_tokens))
        for _ in range(n_merges):
            pair_freq: Counter[tuple[str, str]] = Counter()
            for word, symbols in pieces.items():
                weight = weights[word]
                for pair in zip(symbols, symbols[1:]):
                    pair_freq[pair] += weight
            if not pair_freq:
                break
            best, count = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))
            if count < min_frequency:
                break
            merges.append(best)
            merged = best[0] + best[1]
            for word, symbols in pieces.items():
                if best[0] in symbols and best[1] in symbols:
                    pieces[word] = _merge_pair(symbols, best, merged)

        tokens: list[str] = list(special_tokens)
        seen: set[str] = set(tokens)
        for symbol in sorted({s for symbols in pieces.values() for s in symbols}):
            if symbol not in seen:
                tokens.append(symbol)
                seen.add(symbol)
        for a, b in merges:
            merged = a + b
            if merged not in seen:
                tokens.append(merged)
                seen.add(merged)
        return cls(special_tokens, merges, tokens)

    def _bpe(self, word: str) -> list[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = [BYTE_ENCODER[b] for b in word.encode("utf-8")]
        if len(symbols) < 2:
            self._cache[word] = symbols
            return symbols
        while True:
            best_pair = None
            best_rank = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            symbols = _merge_pair(symbols, best_pair, merged)
            if len(symbols) == 1:
                break
        self._cache[word] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in DEFAULT_SPLIT_RE.findall(text):
            for symbol in self._bpe(piece):
                token = self.token_to_id.get(symbol)
                ids.append(self.unk_id if token is None else token)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytearray()
        for token in ids:
            if token < self.n_special:
                continue
            symbol = self.id_to_token.get(token)
            if symbol is None:
                continue
            for char in symbol:
                byte = BYTE_DECODER.get(char)
                if byte is not None:
                    raw.append(byte)
        return raw.decode("utf-8", errors="replace")

    def state(self) -> dict:
        data = super().state()
        data["merges"] = [list(m) for m in self.merges]
        data["vocab"] = self.vocab
        return data

    @classmethod
    def from_state(cls, data: dict) -> "BPETokenizer":
        merges = [tuple(m) for m in data.get("merges", [])]
        vocab = data.get("vocab", [])
        return cls(data.get("special_tokens", DEFAULT_SPECIALS), merges, vocab)


def _merge_pair(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def build_tokenizer(kind: str, corpus: Sequence[str] | None = None, vocab_size: int = 4096) -> Tokenizer:
    if kind == "byte":
        return ByteTokenizer()
    if kind == "word":
        return WordTokenizer.train(corpus or [])
    if kind == "bpe":
        return BPETokenizer.train(corpus or [], vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer kind: {kind}")

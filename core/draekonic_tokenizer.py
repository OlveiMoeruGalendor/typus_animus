"""Draekonic tokenizer backed by tokendict.json.

Loads vocabulary from tokendict.json and auto-updates when the file changes.
Supports greedy longest-match, root+suffix splitting, plural doubling,
and integrates with the recursive_gpt tokenizer framework.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .tokenizer import Tokenizer, DEFAULT_SPECIALS


class DraekonicTokenizer(Tokenizer):
    kind = "draekonic"

    def __init__(self, dict_path: str = None, special_tokens: Sequence[str] = tuple(DEFAULT_SPECIALS),
                 auto_reload: bool = True):
        super().__init__(special_tokens)
        self.dict_path = dict_path or self._default_path()
        self.auto_reload = auto_reload
        self._last_mtime = 0
        self._load_dict()

    def _default_path(self):
        return str(Path(__file__).parent / "tokendict.json")

    def _load_dict(self):
        if not os.path.exists(self.dict_path):
            self.tokendict = {"_meta": {}, "roots": {}}
            self.word_to_id = {}
            self.id_to_word = {}
            self.entries = {}
            self._build_maps()
            return
        mtime = os.path.getmtime(self.dict_path)
        with open(self.dict_path, "r", encoding="utf-8") as f:
            self.tokendict = json.load(f)
        self._last_mtime = mtime
        self.entries = {}
        for cat, items in self.tokendict.items():
            if cat.startswith("_"):
                continue
            if isinstance(items, dict):
                for word, val in items.items():
                    self.entries[word] = val  # val is [id, gloss]
        self._build_maps()

    def _check_reload(self):
        if not self.auto_reload or not os.path.exists(self.dict_path):
            return
        try:
            mtime = os.path.getmtime(self.dict_path)
            if mtime > self._last_mtime:
                self._load_dict()
        except:
            pass

    def _build_maps(self):
        self.word_to_id = dict(self.special_to_id)
        self.id_to_word = dict(self.id_to_special)
        # build sorted by length for greedy longest match
        words = []
        for word, val in self.entries.items():
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                tok_id = val[0]
            elif isinstance(val, int):
                tok_id = val
            else:
                continue
            words.append((word, tok_id))
        # assign any missing ids
        used = set(self.word_to_id.values())
        base = self.n_special
        for word, tok_id in sorted(words, key=lambda x: x[1]):
            if tok_id in used and tok_id < base + 20000:
                pass
            self.word_to_id[word] = tok_id
            self.id_to_word[tok_id] = word
            used.add(tok_id)
        # also add entries not yet mapped
        for word, val in self.entries.items():
            if word in self.word_to_id:
                continue
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                tid = val[0]
            else:
                continue
            if tid not in self.word_to_id.values():
                self.word_to_id[word] = tid
                self.id_to_word[tid] = word
        self._sorted_words = sorted(self.word_to_id.keys(), key=len, reverse=True)
        # meta info
        self.meta = self.tokendict.get("_meta", {})

    @property
    def vocab_size(self):
        if not self.id_to_word:
            return self.n_special
        return max(self.id_to_word.keys()) + 1

    def _pluralize(self, word: str) -> str:
        # plural = repeat first syllable of last root
        if len(word) < 2:
            return word + word[0]
        # simple: repeat first syllable (2 chars heuristic)
        return word + word[:2]

    def tokenize_greedy(self, text: str) -> List[str]:
        self._check_reload()
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            # try greedy longest match
            matched = False
            for wlen in range(min(20, n - i), 0, -1):
                substr = text[i:i + wlen]
                if substr in self.word_to_id:
                    tokens.append(substr)
                    i += wlen
                    matched = True
                    break
            if not matched:
                # try to split as root + suffix
                # look for known suffixes
                suffixes = [k for k in self.entries.keys() if k.startswith('-') or k.endswith('ah') or k.endswith('ahy')]
                found_split = False
                for s in sorted(suffixes, key=len, reverse=True):
                    s_clean = s.strip('-')
                    if text[i:].endswith(s_clean) or any(text[i:].find(s_clean) > 0 for _ in [0]):
                        # try root + suffix
                        idx = text.find(s_clean, i)
                        if idx > i:
                            root = text[i:idx]
                            if root in self.word_to_id:
                                tokens.append(root)
                                tokens.append(s_clean)
                                i = idx + len(s_clean)
                                found_split = True
                                break
                if not found_split:
                    tokens.append('<unk>')
                    i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        self._check_reload()
        toks = self.tokenize_greedy(text)
        ids = []
        for t in toks:
            if t == '<unk>':
                ids.append(self.unk_id)
            elif t in self.word_to_id:
                ids.append(self.word_to_id[t])
            elif t.endswith('s') or t.endswith('es'):  # try plural
                base = t.rstrip('es').rstrip('s')
                if base in self.word_to_id:
                    ids.append(self.word_to_id[base])
                    # repeat first syllable conceptually handled
                else:
                    ids.append(self.unk_id)
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        self._check_reload()
        parts = []
        for tid in ids:
            if tid in self.id_to_special:
                continue
            if tid in self.id_to_word:
                parts.append(self.id_to_word[tid])
            else:
                parts.append('<unk>')
        return ' '.join(parts)

    def add_entry(self, word: str, token_id: int, gloss: str, category: str = "roots"):
        """Add/update entry in dict and save - will auto-update on next use."""
        self._check_reload()
        if category not in self.tokendict:
            self.tokendict[category] = {}
        self.tokendict[category][word] = [token_id, gloss]
        self.entries[word] = [token_id, gloss]
        self._build_maps()
        self.save_dict()
        return True

    def save_dict(self):
        with open(self.dict_path, "w", encoding="utf-8") as f:
            json.dump(self.tokendict, f, ensure_ascii=False, indent=2)
        self._last_mtime = os.path.getmtime(self.dict_path)

    def reload(self):
        self._load_dict()
        return len(self.entries)

    def state(self):
        data = super().state()
        data["dict_path"] = self.dict_path
        data["num_entries"] = len(self.entries)
        return data

    @classmethod
    def from_state(cls, data):
        return cls(data.get("dict_path"))

"""
Simple word-level tokenizer.
Builds a vocab from the training corpus, encodes/decodes text.
Keeps things lightweight — no external dependencies beyond Python builtins.
"""

import re
import json
import os
from collections import Counter
from typing import List, Optional


class WordTokenizer:
    """
    Word-level tokenizer with a fixed vocabulary.

    Special tokens:
      <pad>  → 0
      <unk>  → 1
      <bos>  → 2
      <eos>  → 3
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self):
        self.word2idx: dict = {}
        self.idx2word: dict = {}

    # ──────────────────────────────────────────────
    #  Build vocabulary
    # ──────────────────────────────────────────────

    def build_vocab(self, texts: List[str], max_vocab: int = 10_000) -> None:
        """
        Build vocabulary from a list of raw text strings.
        Keeps the `max_vocab` most-frequent tokens.
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Always include specials first
        vocab = list(self.SPECIAL_TOKENS)
        for word, _ in counter.most_common(max_vocab - len(self.SPECIAL_TOKENS)):
            vocab.append(word)

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"[Tokenizer] Vocab size: {len(self.word2idx):,}")

    # ──────────────────────────────────────────────
    #  Encode / Decode
    # ──────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        if add_bos:
            ids = [self.word2idx[self.BOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.word2idx[self.EOS_TOKEN]]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        specials = set(self.word2idx[t] for t in self.SPECIAL_TOKENS)
        words = []
        for idx in ids:
            if skip_special and idx in specials:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    # ──────────────────────────────────────────────
    #  Internal tokenisation
    # ──────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lower-case, split on whitespace/punctuation."""
        text = text.lower()
        # Keep apostrophes inside words (e.g. don't → don't)
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\w\s]", text)
        return tokens

    # ──────────────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved vocab to {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        print(f"[Tokenizer] Loaded vocab ({len(self.word2idx):,} tokens) from {path}")

    # ──────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.word2idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

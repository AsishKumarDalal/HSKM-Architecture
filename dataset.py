"""
Dataset utilities for HSKM training.

Uses WikiText-2 (≈2M train tokens) via the `datasets` library —
small enough to train to convergence in ~20-30 min on a single GPU.
Falls back to a tiny synthetic corpus if HuggingFace is unavailable.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from tokenizer import WordTokenizer


# ─────────────────────────────────────────────
#  Raw data loading
# ─────────────────────────────────────────────

def load_wikitext2() -> Tuple[List[str], List[str], List[str]]:
    """
    Loads WikiText-2 from HuggingFace datasets.
    Returns (train_texts, val_texts, test_texts) as lists of strings.
    """
    try:
        from datasets import load_dataset
        print("[Dataset] Downloading WikiText-2 …")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        def extract(split):
            return [
                row["text"].strip()
                for row in ds[split]
                if len(row["text"].strip()) > 10
            ]

        train = extract("train")
        val   = extract("validation")
        test  = extract("test")
        print(f"[Dataset] WikiText-2 — train: {len(train):,}  val: {len(val):,}  test: {len(test):,} paragraphs")
        return train, val, test

    except Exception as e:
        print(f"[Dataset] Could not load WikiText-2 ({e}). Falling back to synthetic corpus.")
        return _synthetic_corpus()


def _synthetic_corpus() -> Tuple[List[str], List[str], List[str]]:
    """
    Very small in-memory corpus used when no internet / datasets package.
    Generates ~10k sentence-like strings for smoke-testing.
    """
    import random
    random.seed(42)

    subjects    = ["the cat", "a dog", "the bird", "the fox", "a man", "the woman", "a child"]
    verbs       = ["ran", "jumped", "saw", "chased", "found", "carried", "watched"]
    objects     = ["the ball", "a tree", "the river", "a mountain", "the forest", "the road"]
    adjectives  = ["quick", "lazy", "small", "large", "bright", "dark", "ancient"]
    connectors  = ["and", "but", "while", "because", "although", "so"]

    def make_sentence():
        s  = random.choice(subjects)
        v  = random.choice(verbs)
        o  = random.choice(objects)
        adj = random.choice(adjectives)
        con = random.choice(connectors)
        s2 = random.choice(subjects)
        v2 = random.choice(verbs)
        return f"The {adj} {s} {v} {o} {con} {s2} {v2} quickly ."

    sentences = [make_sentence() for _ in range(15_000)]
    train = sentences[:12_000]
    val   = sentences[12_000:13_500]
    test  = sentences[13_500:]
    print(f"[Dataset] Synthetic corpus — {len(train)} train / {len(val)} val / {len(test)} test sentences")
    return train, val, test


# ─────────────────────────────────────────────
#  Token-ID array helpers
# ─────────────────────────────────────────────

def texts_to_ids(texts: List[str], tokenizer: WordTokenizer, max_len: int) -> np.ndarray:
    """Encode all texts and concatenate into one flat array of token IDs."""
    all_ids: List[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=max_len)
        all_ids.extend(ids)
    return np.array(all_ids, dtype=np.int32)


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Fixed-length chunk dataset.
    Slices a flat token array into (input, label) pairs of length `seq_len`.
    Labels are identical to inputs (shifted by 1 inside the model's forward pass).
    """

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        # Drop the tail that doesn't fill a full chunk
        n_chunks = len(token_ids) // (seq_len + 1)
        token_ids = token_ids[: n_chunks * (seq_len + 1)]
        self.data = torch.from_numpy(token_ids).long()

    def __len__(self) -> int:
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        x = chunk[:-1]          # input  tokens  [seq_len]
        y = chunk[1:]           # target tokens  [seq_len]
        return x, y


# ─────────────────────────────────────────────
#  High-level data builder
# ─────────────────────────────────────────────

def build_dataloaders(
    seq_len:    int  = 128,
    batch_size: int  = 64,
    max_vocab:  int  = 10_000,
    num_workers: int = 0,
    tokenizer_save_path: str = "checkpoints/vocab.json",
) -> Tuple[DataLoader, DataLoader, WordTokenizer]:
    """
    End-to-end pipeline:
      1. Load raw text
      2. Build / load tokenizer
      3. Encode to token IDs
      4. Wrap in DataLoaders
    Returns (train_loader, val_loader, tokenizer)
    """
    # ── 1. Raw text ──
    train_texts, val_texts, _ = load_wikitext2()

    # ── 2. Tokenizer ──
    tokenizer = WordTokenizer()
    if os.path.exists(tokenizer_save_path):
        tokenizer.load(tokenizer_save_path)
    else:
        tokenizer.build_vocab(train_texts, max_vocab=max_vocab)
        tokenizer.save(tokenizer_save_path)

    # ── 3. Encode ──
    print("[Dataset] Encoding training tokens …")
    train_ids = texts_to_ids(train_texts, tokenizer, max_len=seq_len * 4)
    print("[Dataset] Encoding validation tokens …")
    val_ids   = texts_to_ids(val_texts,   tokenizer, max_len=seq_len * 4)
    print(f"[Dataset] Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

    # ── 4. Datasets ──
    train_ds = TokenDataset(train_ids, seq_len)
    val_ds   = TokenDataset(val_ids,   seq_len)
    print(f"[Dataset] Train batches: {len(train_ds):,}  |  Val batches: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    return train_loader, val_loader, tokenizer

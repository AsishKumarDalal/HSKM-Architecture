"""
Dataset utilities for HSKM training with BPE.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tokenizer import BPETokenizer


def load_wikitext2() -> Tuple[List[str], List[str]]:
    """Loads WikiText-2 from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("[Dataset] Loading WikiText-2 …")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        train = [row["text"].strip() for row in ds["train"] if len(row["text"].strip()) > 10]
        val   = [row["text"].strip() for row in ds["validation"] if len(row["text"].strip()) > 10]
        return train, val
    except Exception as e:
        print(f"[Dataset] Load failed: {e}. Use synthetic.")
        return _synthetic()

def _synthetic():
    sents = ["the quick brown fox jumps over the lazy dog . " * 10] * 1000
    return sents[:800], sents[800:]

def texts_to_ids(texts: List[str], tokenizer: BPETokenizer) -> np.ndarray:
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text))
    return np.array(all_ids, dtype=np.int32)

class TokenDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        n_chunks = len(token_ids) // (seq_len + 1)
        self.data = torch.from_numpy(token_ids[:n_chunks * (seq_len + 1)]).long()

    def __len__(self) -> int:
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

def build_dataloaders(seq_len: int = 512, batch_size: int = 8):
    tokenizer = BPETokenizer()
    train_texts, val_texts = load_wikitext2()
    
    print("[Dataset] Tokenizing training set …")
    train_ids = texts_to_ids(train_texts, tokenizer)
    print("[Dataset] Tokenizing validation set …")
    val_ids   = texts_to_ids(val_texts, tokenizer)
    
    train_ds = TokenDataset(train_ids, seq_len)
    val_ds   = TokenDataset(val_ids, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer

"""
Dataset utilities for HSKM training with TinyStories.
TinyStories provides simple, coherent narratives ideal for small model training.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tokenizer import BPETokenizer


def load_tinystories(max_samples: int = 50000) -> Tuple[List[str], List[str]]:
    """
    Loads TinyStories from HuggingFace.
    We limit the sample count to ensure training fits within a reasonable window.
    """
    try:
        from datasets import load_dataset
        print(f"[Dataset] Loading TinyStories (max {max_samples} samples) …")
        # TinyStories is large, so we use streaming or just select a subset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
        
        # Take a subset for rapid training
        train_data = ds.select(range(min(len(ds), max_samples)))
        val_data   = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)
        
        train_texts = [row["text"].strip() for row in train_data]
        val_texts   = [row["text"].strip() for row in val_data.select(range(min(len(val_data), 2000)))]
        
        print(f"[Dataset] TinyStories loaded: {len(train_texts)} train, {len(val_texts)} val")
        return train_texts, val_texts
    except Exception as e:
        print(f"[Dataset] TinyStories load failed: {e}. Falling back to synthetic.")
        return _synthetic()

def _synthetic():
    sents = ["Once upon a time, there was a little bird who loved to sing . " * 5] * 1000
    return sents[:800], sents[800:]

def texts_to_ids(texts: List[str], tokenizer: BPETokenizer) -> np.ndarray:
    all_ids = []
    for text in texts:
        # For TinyStories, we often want to separate stories with <eos>
        all_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
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

def build_dataloaders(seq_len: int = 512, batch_size: int = 8, max_samples: int = 50000):
    tokenizer = BPETokenizer()
    train_texts, val_texts = load_tinystories(max_samples=max_samples)
    
    print("[Dataset] Tokenizing training set (this may take a minute) …")
    train_ids = texts_to_ids(train_texts, tokenizer)
    print("[Dataset] Tokenizing validation set …")
    val_ids   = texts_to_ids(val_texts, tokenizer)
    
    print(f"[Dataset] Total train tokens: {len(train_ids):,}")
    
    train_ds = TokenDataset(train_ids, seq_len)
    val_ds   = TokenDataset(val_ids, seq_len)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=0 # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

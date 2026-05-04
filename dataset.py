"""
Streaming Dataset utilities for HSKM training with TinyStories.
Uses HuggingFace streaming to provide fresh stories continuously.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Tuple
from tokenizer import BPETokenizer


def get_streaming_dataset():
    from datasets import load_dataset
    print("[Dataset] Initializing TinyStories Stream …")
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds_val   = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    return ds_train, ds_val


class StreamingTokenDataset(IterableDataset):
    def __init__(self, hf_stream, tokenizer: BPETokenizer, seq_len: int):
        self.hf_stream = hf_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        buffer = []
        for example in self.hf_stream:
            text = example["text"].strip()
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def build_dataloaders(seq_len: int = 512, batch_size: int = 8, seed: int = None):
    tokenizer = BPETokenizer()
    ds_train, ds_val = get_streaming_dataset()
    
    # Use a random seed if none provided to ensure different data on restart
    actual_seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()
    ds_train = ds_train.shuffle(buffer_size=10000, seed=actual_seed)
    
    train_ds = StreamingTokenDataset(ds_train, tokenizer, seq_len)
    val_ds   = StreamingTokenDataset(ds_val, tokenizer, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader, tokenizer

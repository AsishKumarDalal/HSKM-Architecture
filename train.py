"""
HSKM Training Script (Scaled Edition)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import HSKM, HSKMConfig
from dataset import build_dataloaders
from tokenizer import BPETokenizer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    # Scaling up configuration
    config = HSKMConfig(
        d_model=512,      # Medium scale
        n_layers=6,       # Multi-layer
        max_seq_len=args.seq_len,
    )
    
    train_loader, val_loader, tokenizer = build_dataloaders(seq_len=args.seq_len, batch_size=args.batch_size)
    config.vocab_size = tokenizer.vocab_size
    
    model = HSKM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(enabled=use_amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss, _ = model(x, labels=y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss, _ = model(x, labels=y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model': model.state_dict(),
                'config': config.__dict__,
            }, "checkpoints/best.pt")
            print("Best model saved!")

    # Final generation test
    prompt = "Artificial intelligence is"
    input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], device=device)
    output = model.generate(input_ids, max_new_tokens=50)
    print(f"\nGenerated: {tokenizer.decode(output[0].tolist())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    train(parser.parse_args())

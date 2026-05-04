"""
HSKM Training Script (Improved Stability & Step-wise Logging)
- Adds LR warmup
- Gradient clipping
- Weight decay tuning
- Step-wise loss logging to artifacts folder
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
import math
import json

from model import HSKM, HSKMConfig
from dataset import build_dataloaders
from tokenizer import BPETokenizer

def get_lr_scheduler(optimizer, warmup_steps, total_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    # Configuration
    config = HSKMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    
    train_loader, val_loader, tokenizer = build_dataloaders(seq_len=args.seq_len, batch_size=args.batch_size)
    config.vocab_size = tokenizer.vocab_size
    
    model = HSKM(config).to(device)
    
    # Optimizer groups
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler(enabled=use_amp)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True) # Create artifacts folder
    
    best_val = float('inf')
    all_step_losses = []

    print(f"Starting training on {device}...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_step_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss, _ = model(x, labels=y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Log step loss
            l_val = loss.item()
            epoch_step_losses.append({
                "step": step,
                "global_step": epoch * len(train_loader) + step,
                "loss": round(l_val, 4)
            })
            
            pbar.set_postfix(loss=f"{l_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        # Save epoch step-wise losses to artifacts
        epoch_log_path = f"artifacts/epoch_{epoch+1}_steps.json"
        with open(epoch_log_path, "w") as f:
            json.dump(epoch_step_losses, f, indent=2)
        print(f"Step-wise losses for epoch {epoch+1} saved to {epoch_log_path}")
        
        all_step_losses.extend(epoch_step_losses)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss, _ = model(x, labels=y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Save global history
        with open("artifacts/history.json", "w") as f:
            json.dump(all_step_losses, f, indent=2)
            
        print(f"Val Loss: {val_loss:.4f} | PPL: {math.exp(min(val_loss, 20)):.2f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model': model.state_dict(),
                'config': config.__dict__,
            }, "checkpoints/best.pt")
            print("★ New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    train(parser.parse_args())

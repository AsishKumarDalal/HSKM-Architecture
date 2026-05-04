"""
HSKM Training Script (Streaming Edition)
- Uses Infinite Streaming: Model sees new stories every step
- Fixed steps per epoch for consistent checkpointing/graphing
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
import matplotlib.pyplot as plt
from itertools import islice

from model import HSKM, HSKMConfig
from dataset import build_dataloaders
from tokenizer import BPETokenizer

STEPS_PER_EPOCH = 2000 
VAL_STEPS = 200

def get_lr_scheduler(optimizer, warmup_steps, total_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_loss_plot(losses, tokens, epoch, path):
    plt.figure(figsize=(10, 5))
    plt.plot(tokens, losses, label='Training Loss')
    plt.title(f'Training Loss (Tokens Seen: {tokens[-1]/1e6:.2f}M)')
    plt.xlabel('Tokens Seen')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    def format_func(value, tick_number):
        if value >= 1e6: return f'{value/1e6:.1f}M'
        if value >= 1e3: return f'{value/1e3:.0f}K'
        return f'{value:.0f}'
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.legend()
    plt.savefig(path)
    plt.close()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    config = HSKMConfig(d_model=args.d_model, n_layers=args.n_layers, max_seq_len=args.seq_len, use_gradient_checkpointing=args.checkpoint)
    train_loader, val_loader, tokenizer = build_dataloaders(seq_len=args.seq_len, batch_size=args.batch_size)
    config.vocab_size = tokenizer.vocab_size
    model = HSKM(config).to(device)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{'params': decay_params, 'weight_decay': 0.1}, {'params': nodecay_params, 'weight_decay': 0.0}]
    optimizer = AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler(enabled=use_amp)
    total_steps = args.epochs * STEPS_PER_EPOCH
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    best_val = float('inf')
    global_losses = []
    global_tokens = []
    total_tokens_seen = 0
    train_iter = iter(train_loader)
    for epoch in range(args.epochs):
        model.train()
        epoch_step_logs = []
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}")
        for step in pbar:
            try: x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            batch_tokens = x.numel()
            total_tokens_seen += batch_tokens
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): loss, _ = model(x, labels=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            l_val = loss.item()
            global_losses.append(l_val)
            global_tokens.append(total_tokens_seen)
            epoch_step_logs.append({"step": step, "tokens_seen": total_tokens_seen, "loss": round(l_val, 4)})
            tokens_str = f"{total_tokens_seen/1e6:.2f}M" if total_tokens_seen >= 1e6 else f"{total_tokens_seen/1e3:.1f}K"
            pbar.set_postfix(loss=f"{l_val:.4f}", tokens=tokens_str, lr=f"{scheduler.get_last_lr()[0]:.2e}")
        with open(f"artifacts/epoch_{epoch+1}_steps.json", "w") as f: json.dump(epoch_step_logs, f, indent=2)
        save_loss_plot(global_losses, global_tokens, epoch + 1, f"artifacts/loss_curve.png")
        model.eval()
        val_loss = 0
        val_iter = iter(val_loader)
        with torch.no_grad():
            for _ in range(VAL_STEPS):
                try:
                    vx, vy = next(val_iter)
                    vx, vy = vx.to(device), vy.to(device)
                    vloss, _ = model(vx, labels=vy)
                    val_loss += vloss.item()
                except StopIteration: break
        val_loss /= VAL_STEPS
        print(f"Val Loss: {val_loss:.4f} | PPL: {math.exp(min(val_loss, 20)):.2f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'config': config.__dict__}, "checkpoints/best.pt")
            print("★ New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--checkpoint", action="store_true")
    train(parser.parse_args())

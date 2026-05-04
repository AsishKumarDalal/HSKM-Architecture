"""
HSKM Training Script
====================
• Trains on WikiText-2 (≈2M tokens) — fits in ~20-30 min on a single GPU.
• Mixed-precision (bf16 / fp16) via torch.amp.
• tqdm progress bars for batch + epoch level.
• Saves best checkpoint (by val loss) and final checkpoint.
• Prints a short generation sample after every epoch.

Usage:
    python train.py                      # auto-detects GPU
    python train.py --epochs 5 --lr 3e-4
"""

import os
import math
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from model   import HSKM, HSKMConfig
from dataset import build_dataloaders
from tokenizer import WordTokenizer


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU: {name}  ({mem:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("[Device] No GPU found — running on CPU (will be slow).")
    return dev


def count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M"


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))          # clip to avoid overflow


def save_checkpoint(
    model:     HSKM,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    config:    HSKMConfig,
    path:      str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config":     config.__dict__,
        },
        path,
    )
    print(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(
    path:      str,
    model:     HSKM,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint; returns the epoch number to resume from."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[Checkpoint] Resumed from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
    return ckpt["epoch"]


# ─────────────────────────────────────────────
#  Validation loop
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: HSKM, val_loader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(val_loader, desc="  Validating", leave=False, unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}")
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=use_amp, dtype=torch.float16):
            loss, _ = model(x, labels=x)        # pass x as labels (shift inside model)
        total_loss    += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_batches, 1)
    model.train()
    return avg_loss


# ─────────────────────────────────────────────
#  Quick generation sample
# ─────────────────────────────────────────────

def generate_sample(
    model:     HSKM,
    tokenizer: WordTokenizer,
    device:    torch.device,
    prompt:    str = "the cat sat on",
    max_new:   int = 50,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(
        input_ids,
        max_new_tokens = max_new,
        temperature    = 0.85,
        top_p          = 0.92,
        top_k          = 40,
    )
    generated = out_ids[0, len(ids):].tolist()
    text = tokenizer.decode(generated, skip_special=True)
    model.train()
    return text


# ─────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────

def train(args) -> None:
    # ── Device ──────────────────────────────────────────────────────────
    device = get_device()
    use_amp = device.type == "cuda"

    # ── Config ──────────────────────────────────────────────────────────
    config = HSKMConfig(
        vocab_size  = args.vocab_size,
        d_model     = args.d_model,
        d_medium    = 64,
        n_kernels   = 16,
        top_k       = 8,
        window      = args.seq_len,
        n_patterns  = 2048,
        mtm_decay   = 0.90,
        max_seq_len = args.seq_len,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, tokenizer = build_dataloaders(
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
        max_vocab  = args.vocab_size,
        tokenizer_save_path = os.path.join(args.ckpt_dir, "vocab.json"),
    )

    # Sync real vocab size (may be smaller than requested if corpus is small)
    config.vocab_size = tokenizer.vocab_size

    # ── Model ────────────────────────────────────────────────────────────
    model = HSKM(config).to(device)
    print(f"[Model] Parameters — {count_params(model)}")

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 0.1,
        betas        = (0.9, 0.95),
        eps          = 1e-8,
    )
    scaler    = GradScaler(enabled=use_amp)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = args.epochs * len(train_loader),
        eta_min = args.lr * 0.1,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    last_ckpt_path = os.path.join(args.ckpt_dir, "last.pt")

    if args.resume and os.path.exists(last_ckpt_path):
        start_epoch = load_checkpoint(last_ckpt_path, model, optimizer)

    # ── Training ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  HSKM Training  |  {args.epochs} epochs  |  device={device}")
    print("═" * 65 + "\n")

    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        # ── Batch loop with tqdm ─────────────────────────────────────────
        pbar = tqdm(
            train_loader,
            desc       = f"Epoch {epoch+1:02d}/{args.epochs}",
            unit       = "batch",
            leave      = True,
            bar_format = "{l_bar}{bar:35}{r_bar}",
        )

        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss
            with autocast(enabled=use_amp, dtype=torch.float16):
                loss, _ = model(x, labels=x)

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Accumulate
            epoch_loss += loss.item()
            n_batches  += 1

            # Live tqdm stats
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                ppl  = f"{perplexity(loss.item()):6.1f}",
                lr   = f"{lr_now:.2e}",
            )

        # ── Epoch summary ────────────────────────────────────────────────
        avg_train_loss = epoch_loss / max(n_batches, 1)
        val_loss       = evaluate(model, val_loader, device, use_amp)
        elapsed        = time.time() - t0

        print(
            f"\n  ▶ Epoch {epoch+1:02d} | "
            f"Train loss: {avg_train_loss:.4f}  PPL: {perplexity(avg_train_loss):.1f}  | "
            f"Val loss:   {val_loss:.4f}  PPL: {perplexity(val_loss):.1f}  | "
            f"Time: {elapsed:.0f}s"
        )

        # ── Save best ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config, best_ckpt_path)
            print(f"  ★ New best val loss: {best_val_loss:.4f}")

        # ── Save last ────────────────────────────────────────────────────
        save_checkpoint(model, optimizer, epoch + 1, val_loss, config, last_ckpt_path)

        # ── Generation sample ─────────────────────────────────────────────
        sample_prompts = [
            "the cat sat on",
            "in the beginning",
            "scientists discovered that",
        ]
        print("\n  ── Generation samples ──────────────────────────────")
        for prompt in sample_prompts:
            sample = generate_sample(model, tokenizer, device, prompt=prompt, max_new=40)
            print(f"  [{prompt}]  →  {sample}")
        print()

        # ── History ──────────────────────────────────────────────────────
        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(avg_train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "train_ppl":  round(perplexity(avg_train_loss), 2),
            "val_ppl":    round(perplexity(val_loss), 2),
        })

    # ── Save training history ────────────────────────────────────────────
    history_path = os.path.join(args.ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Training] History saved → {history_path}")

    # ── Final generation ──────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  TRAINING COMPLETE — Final generation (loading best checkpoint)")
    print("═" * 65)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    long_prompts = [
        "the quick brown fox",
        "once upon a time",
        "the president announced that",
        "deep learning models have",
    ]
    for prompt in long_prompts:
        sample = generate_sample(model, tokenizer, device, prompt=prompt, max_new=80)
        print(f"\n  Prompt : {prompt}")
        print(f"  Output : {sample}")

    print(f"\n  Best checkpoint : {best_ckpt_path}")
    print(f"  Vocab file      : {os.path.join(args.ckpt_dir, 'vocab.json')}")
    print("  Done! ✓\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the HSKM language model")
    p.add_argument("--epochs",     type=int,   default=5,      help="Number of training epochs")
    p.add_argument("--batch_size", type=int,   default=64,     help="Batch size per step")
    p.add_argument("--lr",         type=float, default=3e-4,   help="Peak learning rate")
    p.add_argument("--seq_len",    type=int,   default=128,    help="Sequence length")
    p.add_argument("--d_model",    type=int,   default=256,    help="Hidden dimension")
    p.add_argument("--vocab_size", type=int,   default=10000,  help="Max vocabulary size")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints", help="Checkpoint directory")
    p.add_argument("--resume",     action="store_true",        help="Resume from last checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

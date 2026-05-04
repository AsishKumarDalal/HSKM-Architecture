"""
HSKM-V3 vs. Standard Transformer Comparison Benchmark
------------------------------------------------------
Proves the linear scaling advantage of HSKM over quadratic attention.
Compatible with HSKM-V3 forward(input_ids, labels) -> (loss, logits).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HSKM, HSKMConfig
from tqdm import tqdm
import json
import os


# --- Baseline: Standard Transformer (O(n^2) attention) ---
class StandardTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layers)
        ])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


def run_performance_test(model, seq_len, is_hskm=False, batch_size=8, steps=20):
    device = next(model.parameters()).device
    model.train()
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    def run_step():
        if is_hskm:
            loss, logits = model(x, labels=x)
            loss.backward()
        else:
            out = model(x)
            F.cross_entropy(out.view(-1, out.size(-1)), x.view(-1)).backward()

    for _ in range(3):
        run_step()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(steps):
        run_step()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    ms_per_step = (elapsed / steps) * 1000
    tokens_per_sec = (batch_size * seq_len * steps) / elapsed
    return ms_per_step, tokens_per_sec


def run_vram_test(model_class, config, seq_len, is_hskm=False, batch_size=4):
    if not torch.cuda.is_available():
        return float('nan')

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = model_class(config).to(device)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    try:
        if is_hskm:
            loss, _ = model(x, labels=x)
            loss.backward()
        else:
            out = model(x)
            F.cross_entropy(out.view(-1, out.size(-1)), x.view(-1)).backward()
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    except RuntimeError:
        peak_vram = float('nan')

    del model, x
    torch.cuda.empty_cache()
    return peak_vram


if __name__ == "__main__":
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    config = HSKMConfig(d_model=512, n_layers=4, n_heads=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hskm_model = HSKM(config).to(device)
    trans_model = StandardTransformer(config).to(device)

    lengths = [128, 256, 512, 1024]
    results = []

    print("=" * 75)
    print(f"  HSKM-V3.1 vs Standard Transformer Benchmark")
    print(f"  Device: {device_name}")
    print("=" * 75)
    print(f"{'SeqLen':<8} | {'Model':<14} | {'ms/step':<10} | {'tok/sec':<12} | {'VRAM (MB)':<10}")
    print("-" * 75)

    for seq_len in lengths:
        t_vram = run_vram_test(StandardTransformer, config, seq_len, is_hskm=False)
        t_ms, t_tps = run_performance_test(trans_model, seq_len, is_hskm=False) if t_vram == t_vram else (float('nan'), float('nan'))
        print(f"{seq_len:<8} | {'Transformer':<14} | {t_ms:>9.1f} | {t_tps:>11,.0f} | {t_vram:>9.1f}")

        h_vram = run_vram_test(HSKM, config, seq_len, is_hskm=True)
        h_ms, h_tps = run_performance_test(hskm_model, seq_len, is_hskm=True) if h_vram == h_vram else (float('nan'), float('nan'))
        print(f"{'':<8} | {'HSKM-V3.1':<14} | {h_ms:>9.1f} | {h_tps:>11,.0f} | {h_vram:>9.1f}")
        print("-" * 75)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

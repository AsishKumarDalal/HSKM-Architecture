"""
HSKM Benchmarking Suite
-----------------------
Measures:
1. Training Throughput (Tokens/sec)
2. Inference Latency (ms/token)
3. Peak VRAM Consumption
4. Scaling over Sequence Lengths
"""

import time
import torch
import torch.nn as nn
from model import HSKM, HSKMConfig
from tqdm import tqdm
import numpy as np

def benchmark_training(model, seq_len=256, batch_size=12, steps=50):
    device = next(model.parameters()).device
    model.train()
    
    # Dummy data
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    print(f"Benchmarking Training (BS={batch_size}, Seq={seq_len})...")
    
    # Warmup
    for _ in range(5):
        loss, _ = model(x, labels=y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in tqdm(range(steps)):
        loss, _ = model(x, labels=y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    total_tokens = steps * batch_size * seq_len
    tokens_per_sec = total_tokens / total_time
    
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    return tokens_per_sec

def benchmark_inference(model, seq_len=256, num_tokens=100):
    device = next(model.parameters()).device
    model.eval()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    
    print(f"Benchmarking Inference (Context={seq_len}, Generating={num_tokens})...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=num_tokens)
        
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    ms_per_token = (total_time / num_tokens) * 1000
    print(f"Latency: {ms_per_token:.2f} ms/token")
    return ms_per_token

def measure_vram(model, seq_len=512, batch_size=16):
    if not torch.cuda.is_available():
        print("VRAM measurement requires CUDA.")
        return
        
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    loss, _ = model(x, labels=x)
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM (BS={batch_size}, Seq={seq_len}): {peak_mem:.2f} MB")
    return peak_mem

if __name__ == "__main__":
    config = HSKMConfig(d_model=512, n_layers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSKM(config).to(device)
    
    print("-" * 30)
    print("      HSKM BENCHMARK")
    print("-" * 30)
    
    # 1. Training Throughput
    benchmark_training(model)
    
    # 2. Inference Latency
    benchmark_inference(model)
    
    # 3. VRAM Usage
    if torch.cuda.is_available():
        measure_vram(model)
        
    # 4. Stress Test (Long Context)
    print("\n--- Long Context Stress Test ---")
    for length in [512, 1024, 2048]:
        try:
            measure_vram(model, seq_len=length, batch_size=4)
        except Exception as e:
            print(f"Failed at {length}: {e}")
            break

# Hierarchical Sparse Kernel Memory (HSKM) V3.1

A production-ready language model combining **learned sparse kernel attention** with **hierarchical memory banks**.

## V3.1 Fixes
- RoPE applied to queries (kernels stay position-invariant)
- Vectorized EMA (no Python for-loop)
- Truly sparse top-k before softmax
- LTM with gated read/write operations
- Gradient checkpointing support
- Residual scaling by layer depth
- Causal kernel gating (optional, off by default)

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --epochs 10 --batch_size 12 --seq_len 256
```

## Benchmarking
```bash
python benchmark.py
```

## Generation
```bash
python generate.py --prompt "Once upon a time" --tokens 100
```

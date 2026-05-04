# Hierarchical Sparse Kernel Memory (HSKM) - TinyStories Edition

A hybrid language model combining **learned sparse kernel attention** with **hierarchical memory banks**, now optimized for the **TinyStories** dataset.

## Features
- **Infinite Streaming**: Model sees fresh data from the 2.1M TinyStories corpus in every batch.
- **BPE Tokenization**: GPT-2 compatible byte-pair encoding.
- **Advanced Architecture**: RMSNorm, SwiGLU, and Multi-Head Kernel Attention.
- **Visual Analytics**: Step-wise JSON logging and real-time loss plots in `artifacts/`.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --epochs 10 --batch_size 12 --seq_len 256
```

## Benchmarking
To measure performance (throughput, latency, and VRAM):
```bash
python benchmark.py
```

## Generation
```bash
python generate.py --prompt "Once upon a time, there was a girl named Lily" --tokens 100
```

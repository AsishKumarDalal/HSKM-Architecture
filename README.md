# Hierarchical Sparse Kernel Memory (HSKM) - TinyStories Edition

A hybrid language model combining **learned sparse kernel attention** with **hierarchical memory banks**, now optimized for the **TinyStories** dataset.

## Major Updates
- **TinyStories Integration**: Switched from WikiText-2 to `TinyStories`. This dataset is much more suitable for training small, coherent language models.
- **BPE Tokenization**: Using `tiktoken` with GPT-2 vocabulary.
- **Advanced Architecture**: RMSNorm, Multi-Head Kernel Attention, and SwiGLU MLPs for superior convergence.
- **Visual Progress**: Real-time loss curve generation in `artifacts/loss_curve.png`.

## File Layout
- `model.py`: Multi-layer HSKM blocks.
- `tokenizer.py`: BPE encoding via `tiktoken`.
- `dataset.py`: TinyStories data pipeline.
- `train.py`: Training loop with stability features & plotting.
- `generate.py`: Text generation utility.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --epochs 5 --batch_size 12 --seq_len 256
```

## Generation
```bash
python generate.py --prompt "Once upon a time, there was a girl named Lily" --tokens 100
```

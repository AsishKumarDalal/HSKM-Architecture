# Hierarchical Sparse Kernel Memory (HSKM)

A hybrid language model combining **learned sparse kernel attention** with **hierarchical memory banks**.

## File Layout

```
statistical model/
├── model.py          ← Full HSKM architecture (all nn.Module classes)
├── tokenizer.py      ← Word-level tokenizer (build / save / load vocab)
├── dataset.py        ← WikiText-2 loader + PyTorch Dataset / DataLoader
├── train.py          ← Training loop (tqdm, mixed-precision, checkpointing)
├── generate.py       ← Interactive REPL + single-prompt generation
├── requirements.txt  ← pip dependencies
└── checkpoints/      ← Created automatically during training
    ├── vocab.json
    ├── best.pt
    ├── last.pt
    └── history.json
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train  (~20-30 min on a single GPU, ~2-3 hours on CPU)
python train.py

# 3. Generate (interactive REPL)
python generate.py

# 4. Single prompt
python generate.py --prompt "the quick brown fox" --max_tokens 80
```

## Training CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 5 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--seq_len` | 128 | Sequence length |
| `--d_model` | 256 | Hidden dimension |
| `--vocab_size` | 10000 | Max vocabulary size |
| `--ckpt_dir` | checkpoints | Save directory |
| `--resume` | False | Resume from last.pt |

## Generation CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | None | Single prompt (omit for REPL) |
| `--max_tokens` | 100 | Tokens to generate |
| `--temperature` | 0.85 | Sampling temperature |
| `--top_p` | 0.92 | Nucleus sampling p |
| `--top_k` | 40 | Top-k filter |
| `--ckpt` | checkpoints/best.pt | Checkpoint to load |

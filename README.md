# Hierarchical Sparse Kernel Memory (HSKM) - Scaled BPE Edition

A hybrid language model combining **learned sparse kernel attention** with **hierarchical memory banks**, now scaled up and using **Byte-Pair Encoding (BPE)** like GPT-2.

## Major Updates
- **BPE Tokenization**: Switched to `tiktoken` with GPT-2 vocabulary for better generalization.
- **Multi-Layer Architecture**: Added support for stacked HSKM blocks with residual connections and MLP layers.
- **Increased Size**: Default configuration scaled to `d_model=512` and `n_layers=6` (adjustable).

## File Layout
- `model.py`: Multi-layer HSKM blocks.
- `tokenizer.py`: BPE encoding via `tiktoken`.
- `dataset.py`: WikiText-2 loading with BPE tokens.
- `train.py`: Training loop for the scaled model.
- `generate.py`: Text generation utility.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --epochs 3 --batch_size 8 --seq_len 128
```

## Generation
```bash
python generate.py --prompt "The future of AI is" --tokens 50
```

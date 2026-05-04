"""
HSKM Interactive Generation Script
====================================
Loads a trained checkpoint and lets you generate text interactively
(REPL mode) or from command-line prompts.

Usage:
    # Interactive REPL
    python generate.py

    # One-shot prompt
    python generate.py --prompt "the universe began" --max_tokens 120

    # Custom checkpoint
    python generate.py --ckpt checkpoints/best.pt --prompt "science shows"
"""

import os
import sys
import argparse
import torch

from model     import HSKM, HSKMConfig
from tokenizer import WordTokenizer


# ─────────────────────────────────────────────
#  Load model + tokenizer from checkpoint
# ─────────────────────────────────────────────

def load_model(ckpt_path: str, vocab_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run `python train.py` first to train the model.")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"[ERROR] Vocab file not found: {vocab_path}")
        sys.exit(1)

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = WordTokenizer()
    tokenizer.load(vocab_path)

    # ── Config from checkpoint ─────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    config = HSKMConfig(**cfg_dict)

    # ── Model ─────────────────────────────────────────────────────────
    model = HSKM(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"[Generate] Loaded checkpoint (epoch={epoch}, val_loss={val_loss:.4f})")
    return model, tokenizer, config


# ─────────────────────────────────────────────
#  Single generation call
# ─────────────────────────────────────────────

def generate_text(
    model:         HSKM,
    tokenizer:     WordTokenizer,
    prompt:        str,
    device:        torch.device,
    max_new_tokens: int  = 100,
    temperature:   float = 0.85,
    top_p:         float = 0.92,
    top_k:         int   = 40,
) -> str:
    """Encode prompt → run model.generate() → decode output."""
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not ids:
        return "(empty prompt)"

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            top_k          = top_k,
        )

    new_ids = out_ids[0, len(ids):].tolist()
    return tokenizer.decode(new_ids, skip_special=True)


# ─────────────────────────────────────────────
#  Interactive REPL
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         HSKM Language Model — Interactive Shell         ║
╠══════════════════════════════════════════════════════════╣
║  Type a prompt and press Enter to generate text.        ║
║  Commands:                                               ║
║    /temp <float>   — set temperature  (default 0.85)    ║
║    /top_p <float>  — set top-p        (default 0.92)    ║
║    /top_k <int>    — set top-k        (default 40)      ║
║    /len <int>      — set max tokens   (default 100)     ║
║    /quit           — exit                               ║
╚══════════════════════════════════════════════════════════╝
"""

def interactive_repl(model, tokenizer, device, args):
    print(BANNER)
    temperature = args.temperature
    top_p       = args.top_p
    top_k       = args.top_k
    max_tokens  = args.max_tokens

    while True:
        try:
            prompt = input("\n  Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not prompt:
            continue

        # ── Commands ─────────────────────────────────────────────────────
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd   = parts[0]
            try:
                if cmd == "/quit":
                    print("  Goodbye!")
                    break
                elif cmd == "/temp" and len(parts) == 2:
                    temperature = float(parts[1])
                    print(f"  temperature = {temperature}")
                elif cmd == "/top_p" and len(parts) == 2:
                    top_p = float(parts[1])
                    print(f"  top_p = {top_p}")
                elif cmd == "/top_k" and len(parts) == 2:
                    top_k = int(parts[1])
                    print(f"  top_k = {top_k}")
                elif cmd == "/len" and len(parts) == 2:
                    max_tokens = int(parts[1])
                    print(f"  max_tokens = {max_tokens}")
                else:
                    print(f"  Unknown command: {cmd}")
            except ValueError:
                print("  Invalid value.")
            continue

        # ── Generate ─────────────────────────────────────────────────────
        output = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens = max_tokens,
            temperature    = temperature,
            top_p          = top_p,
            top_k          = top_k,
        )
        print(f"\n  ── Output ──────────────────────────────────────────────")
        print(f"  {prompt} {output}")
        print(f"  ────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate text with a trained HSKM model")
    p.add_argument("--ckpt",        type=str,   default="checkpoints/best.pt")
    p.add_argument("--vocab",       type=str,   default="checkpoints/vocab.json")
    p.add_argument("--prompt",      type=str,   default=None,
                   help="Single prompt (non-interactive). Omit for REPL mode.")
    p.add_argument("--max_tokens",  type=int,   default=100)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_p",       type=float, default=0.92)
    p.add_argument("--top_k",       type=int,   default=40)
    p.add_argument("--device",      type=str,   default="auto",
                   help="'auto', 'cuda', or 'cpu'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Generate] Using device: {device}")

    # Load
    model, tokenizer, config = load_model(args.ckpt, args.vocab, device)

    if args.prompt:
        # Non-interactive single generation
        output = generate_text(
            model, tokenizer, args.prompt, device,
            max_new_tokens = args.max_tokens,
            temperature    = args.temperature,
            top_p          = args.top_p,
            top_k          = args.top_k,
        )
        print(f"\nPrompt : {args.prompt}")
        print(f"Output : {args.prompt} {output}\n")
    else:
        # Interactive REPL
        interactive_repl(model, tokenizer, device, args)

"""
HSKM Generation Script (Production Edition)
- Added Repetition Penalty support
- Added Temperature and Top-P controls
"""

import torch
import argparse
from model import HSKM, HSKMConfig
from tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity (0.1 - 1.5)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--penalty", type=float, default=1.2, help="Repetition penalty (>1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        config = HSKMConfig(**checkpoint['config'])
        model = HSKM(config).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print(f"★ Loaded model from {args.ckpt}")
    except FileNotFoundError:
        print(f"⚠ Checkpoint {args.ckpt} not found. Ensure you have trained the model first.")
        return

    tokenizer = BPETokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_eos=False)], device=device)
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)
    
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.penalty
    )
    
    result = tokenizer.decode(output_ids[0].tolist())
    print(f"Generated:\n{result}")
    print("-" * 40)

if __name__ == "__main__":
    main()

"""
HSKM Generation Script (BPE Edition)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device)
    config = HSKMConfig(**checkpoint['config'])
    model = HSKM(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tokenizer = BPETokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_eos=False)], device=device)
    
    print(f"Prompt: {args.prompt}")
    output_ids = model.generate(input_ids, max_new_tokens=args.tokens)
    print(f"Generated: {tokenizer.decode(output_ids[0].tolist())}")

if __name__ == "__main__":
    main()

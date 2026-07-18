"""
Script load và chạy model Nexus từ file .pt đã train.

Chạy:
  python3 run_model.py                     # interactive, random seed
  python3 run_model.py --prompt "Hello"    # single prompt
  python3 run_model.py --steps 50          # sinh 50 tokens
"""

import argparse
import os
import sys
import time

import torch

from model import Transformer, ModelArgs


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: list,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    eos_id: int = -1,
) -> list:
    """Sinh token từ prompt."""
    device = next(model.parameters()).device
    total_len = len(prompt_tokens) + max_new_tokens
    tokens = torch.full((1, total_len), -1, dtype=torch.long, device=device)
    tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    generated = []
    prev_pos = 0
    for cur_pos in range(len(prompt_tokens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        else:
            next_token = logits.argmax(dim=-1)
        tokens[:, cur_pos] = next_token
        generated.append(next_token.item())
        prev_pos = cur_pos
        if next_token.item() == eos_id:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(description="Run Nexus model inference")
    parser.add_argument("--model", type=str, default="nexus_model.pt", help="Đường dẫn file .pt")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (nếu không có thì interactive)")
    parser.add_argument("--steps", type=int, default=50, help="Số token sinh tối đa")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load checkpoint
    print(f"Loading model từ {args.model}...")
    start = time.time()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    print(f"  dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")
    elapsed = time.time() - start

    # Khởi tạo model trên CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    model = Transformer(model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"  ✅ Loaded in {elapsed:.2f}s ({sum(p.numel() for p in model.parameters()):,} params)")

    vocab_size = checkpoint.get("vocab_size", model_args.vocab_size)
    print(f"  vocab_size={vocab_size}")

    # Batch mode
    if args.prompt is not None:
        # Chuyển prompt string thành token IDs ngẫu nhiên (vì train trên random data)
        # Trong thực tế, dùng tokenizer.encode()
        prompt_ids = [hash(c) % (vocab_size - 1) + 1 for c in args.prompt[:8]]
        print(f"  Prompt tokens: {prompt_ids}")
        start = time.time()
        generated = generate(model, prompt_ids, max_new_tokens=args.steps, temperature=args.temperature)
        elapsed = time.time() - start
        print(f"  Generated {len(generated)} tokens in {elapsed:.2f}s")
        print(f"  Output IDs: {generated}")
        return

    # Interactive mode
    print(f"\nInteractive mode. Type prompts (tokens are random). Commands: /exit, /stats")
    import sys
    start_time = time.time()
    total_tokens = 0
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break
        if prompt == "/exit":
            break
        if prompt == "/stats":
            if total_tokens > 0:
                elapsed = time.time() - start_time
                print(f"[Stats] {total_tokens} tokens, {elapsed:.1f}s ({total_tokens/elapsed:.1f} tok/s)")
            continue
        # Tạo random prompt tokens
        prompt_ids = [hash(c + str(i)) % (vocab_size - 1) + 1 for i, c in enumerate(prompt[:8])]
        start = time.time()
        generated = generate(model, prompt_ids, max_new_tokens=args.steps, temperature=args.temperature)
        elapsed = time.time() - start
        total_tokens += len(generated)
        print(f"Input IDs:  {prompt_ids}")
        print(f"Output IDs: {generated}")
        print(f"({len(generated)} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    main()

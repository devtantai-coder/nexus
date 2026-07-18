"""
Script Chat với Nexus — text generation với model đã train.
Hỗ trợ cả random data lẫn text data.

Chạy:
  python3 chat.py                                    # dùng nexus_model.pt (random)
  python3 chat.py --model chat_model.pt               # dùng chat_model.pt (text)
  python3 chat.py --model chat_model.pt --prompt "xin chào"  # 1 câu
  python3 chat.py --interactive                      # chat tay
"""

import argparse
import time
import torch

from model import Transformer, ModelArgs


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: list,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    eos_id: int = 0,
) -> list:
    """Sinh token autoregressive."""
    device = next(model.parameters()).device
    total_len = len(prompt_tokens) + max_new_tokens
    tokens = torch.full((1, total_len), -1, dtype=torch.long, device=device)
    tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    generated = []
    prev_pos = 0
    for cur_pos in range(len(prompt_tokens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        else:
            next_token = logits.argmax(dim=-1)
        token_id = next_token.item()
        tokens[:, cur_pos] = next_token
        generated.append(token_id)
        prev_pos = cur_pos
        if token_id == eos_id:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(description="Chat với Nexus model")
    parser.add_argument("--model", type=str, default="chat_model.pt",
                        help="File .pt model (mặc định: chat_model.pt)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt một lần (không dùng interactive)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Số token sinh tối đa")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature (0=greedy, >0=creative)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--interactive", action="store_true",
                        help="Chế độ chat tương tác")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load checkpoint
    print(f"[*] Loading {args.model}...", end=" ", flush=True)
    start = time.time()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    elapsed = time.time() - start
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ {elapsed:.2f}s ({n_params:,} params, device={device})")

    # Lấy vocab
    vocab = ckpt.get("vocab", None)
    if vocab is not None and isinstance(vocab, dict):
        id2word = {int(v): k for k, v in vocab.items()}
        has_vocab = True
        print(f"[*] Vocabulary: {len(vocab)} words")
    else:
        has_vocab = False

    # Hàm chuyển text → token IDs
    def text_to_ids(text: str) -> list:
        if has_vocab:
            return [vocab.get(w, 0) for w in text.strip().lower().split()]
        else:
            # Random hash fallback
            vs = ckpt.get("vocab_size", model_args.vocab_size)
            return [hash(c) % (vs - 1) + 1 for c in text[:8]]

    # Hàm chuyển IDs → text
    def ids_to_text(ids: list) -> str:
        if has_vocab:
            words = [id2word.get(i, "<?>") for i in ids if i != 0 and i != -1]
            return " ".join(words)
        else:
            return str(ids)

    # Single prompt
    if args.prompt is not None:
        prompt_ids = text_to_ids(args.prompt)
        start = time.time()
        gen_ids = generate(model, prompt_ids, args.steps, args.temperature)
        elapsed = time.time() - start
        output = ids_to_text(gen_ids)
        print(f"\nBạn: {args.prompt}")
        print(f"Nexus: {output}")
        print(f"[{len(gen_ids)} tokens in {elapsed:.2f}s]")
        return

    # Interactive
    if args.interactive or (vocab is not None):
        print(f"\n{'='*50}")
        print(f" Chat với Nexus — gõ /help để xem commands")
        print(f"{'='*50}")
        start_time = time.time()
        total_tokens = 0
        while True:
            try:
                prompt = input("\nBạn: ")
            except (EOFError, KeyboardInterrupt):
                break

            if prompt == "/exit":
                break
            elif prompt == "/stats":
                if total_tokens > 0:
                    elapsed = time.time() - start_time
                    print(f"[Stats] {total_tokens} tokens, {elapsed:.1f}s "
                          f"({total_tokens/elapsed:.1f} tok/s)")
                continue
            elif prompt == "/help":
                print("Commands: /exit, /stats, /reset, /temperature <value>")
                continue
            elif prompt.startswith("/temperature"):
                try:
                    args.temperature = float(prompt.split()[1])
                    print(f"Temperature = {args.temperature}")
                except:
                    print(f"Temperature = {args.temperature}")
                continue

            prompt_ids = text_to_ids(prompt)
            start = time.time()
            gen_ids = generate(model, prompt_ids, args.steps, args.temperature)
            elapsed = time.time() - start
            total_tokens += len(gen_ids)
            output = ids_to_text(gen_ids)
            print(f"Nexus: {output}")
            print(f"       [{len(gen_ids)} tok in {elapsed:.2f}s]")


if __name__ == "__main__":
    main()

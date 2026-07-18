"""
Script huấn luyện mô hình Transformer (Nexus) — next-token prediction.

Chạy:
  python train.py                          # CPU với cấu hình tiny
  python train.py --device cuda            # GPU
  python train.py --device cuda --steps 100 --lr 3e-4
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import Transformer, ModelArgs


# === Tiny config cho training nhanh ===
@dataclass
class TrainArgs:
    # Model
    dim: int = 256
    inter_dim: int = 1024
    moe_inter_dim: int = 512
    n_layers: int = 4
    n_dense_layers: int = 1
    n_heads: int = 4
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    vocab_size: int = 4096
    max_seq_len: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    # Training
    batch_size: int = 8
    seq_len: int = 64
    lr: float = 3e-4
    steps: int = 100
    warmup: int = 10


class RandomTextDataset(Dataset):
    """Dataset ngẫu nhiên cho next-token prediction."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:].clone()


def train():
    parser = argparse.ArgumentParser(description="Train Nexus Transformer")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=100, help="Số bước huấn luyện")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10, help="Số bước warmup LR")
    parser.add_argument("--dim", type=int, default=256, help="Kích thước ẩn mô hình")
    parser.add_argument("--n-layers", type=int, default=4, help="Số tầng transformer")
    parser.add_argument("--n-heads", type=int, default=4, help="Số đầu attention")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--compile", action="store_true", help="Bật torch.compile")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, fallback về CPU")

    # === Global precision — float32 cho training stability (CPU) ===
    torch.set_float32_matmul_precision('high')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # === Model config ===
    model_args = ModelArgs(
        max_batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        dim=args.dim,
        inter_dim=args.dim * 4,
        moe_inter_dim=args.dim,
        n_layers=args.n_layers,
        n_dense_layers=1,
        n_heads=args.n_heads,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        kv_lora_rank=max(64, args.dim // 4),
        qk_nope_head_dim=max(32, args.dim // args.n_heads // 2),
        qk_rope_head_dim=max(16, args.dim // args.n_heads // 4),
        v_head_dim=max(32, args.dim // args.n_heads),
        dtype="float32",
    )

    print(f"\n{'='*50}")
    print(f"Khởi tạo mô hình...")
    print(f"{'='*50}")
    print(f"  dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")
    print(f"  vocab_size={model_args.vocab_size}, seq_len={args.seq_len}")
    print(f"  device={device}")

    model = Transformer(model_args).to(device)

    # === Khởi tạo trọng số — tránh NaN từ torch.empty ===
    import torch.nn.init as _init
    for _name, _p in model.named_parameters():
        if _p.ndim >= 2:
            _init.normal_(_p, mean=0.0, std=0.02)
        else:
            _init.zeros_(_p)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Tham số: {total_params:,}")

    if args.compile and device.type == "cuda":
        print("  Bật torch.compile...")
        model.compile_if_enabled()

    # === Optimizer + Dataset ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup) if args.warmup > 0 else 1.0,
    )
    dataset = RandomTextDataset(args.vocab_size, args.seq_len, num_samples=1000)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # === Training loop ===
    print(f"\n{'='*50}")
    print(f"Bắt đầu huấn luyện — {args.steps} steps")
    print(f"{'='*50}")
    print(f"  lr={args.lr}, batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"  device={device}")
    print()

    model.train()
    step = 0
    epoch = 0
    total_tokens = 0
    start_time = time.time()

    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)

            logits = model.forward_train(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tokens = x.numel()
            total_tokens += tokens

            if step % 10 == 0 or step == args.steps - 1:
                elapsed = time.time() - start_time
                tok_s = total_tokens / elapsed if elapsed > 0 else 0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {step:>4d}/{args.steps} | loss {loss.item():.4f} | "
                      f"lr {lr_now:.2e} | {tok_s:.0f} tok/s")

            step += 1

        epoch += 1

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✅ Huấn luyện hoàn thành!")
    print(f"   {args.steps} steps trong {elapsed:.2f}s ({total_tokens/elapsed:.0f} tok/s)")
    print(f"   Loss cuối: {loss.item():.4f}")

    # === Save model ===
    save_path = "nexus_model.pt"
    print(f"\n{'='*50}")
    print(f"Lưu model ra {save_path}...")
    print(f"{'='*50}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_args': model_args,
        'vocab_size': args.vocab_size,
    }, save_path)
    print(f"   ✅ Đã lưu {save_path}")

    # === Test generation ===
    print(f"\n{'='*50}")
    print(f"Test sinh văn bản...")
    print(f"{'='*50}")
    model.eval()
    prompt = torch.randint(1, args.vocab_size, (1, 8), device=device)
    with torch.inference_mode():
        for _ in range(20):
            logits = model(prompt)
            next_token = logits.argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=-1)
    generated = prompt[0].tolist()
    print(f"   Prompt: [1, 8] -> {prompt.size(1)} tokens")
    print(f"   First 12 tokens: {generated[:12]}")


if __name__ == "__main__":
    train()

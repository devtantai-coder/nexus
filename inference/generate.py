import os
import json
import time
from argparse import ArgumentParser
from typing import List, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Lấy mẫu một token từ logits sử dụng scaling nhiệt độ (Gumbel-max trick).

    Args:
        logits (torch.Tensor): Tensor logits cho dự đoán token.
        temperature (float, optional): Nhiệt độ để scaling logits. Mặc định 1.0.

    Returns:
        torch.Tensor: Token đã được lấy mẫu.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Sinh các token mới dựa trên các token prompt đầu vào sử dụng mô hình chỉ định.

    Args:
        model (Transformer): Mô hình transformer dùng để sinh token.
        prompt_tokens (List[List[int]]): Danh sách các list chứa token prompt cho mỗi chuỗi.
        max_new_tokens (int): Số token mới tối đa cần sinh.
        eos_id (int): ID token kết thúc chuỗi (end-of-sequence).
        temperature (float, optional): Giá trị nhiệt độ cho việc lấy mẫu. Mặc định 1.0.

    Returns:
        List[List[int]]: Danh sách các list chứa token đã sinh cho mỗi chuỗi.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    device = next(model.parameters()).device
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=device)
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    compile_model: bool = False,
) -> None:
    """
    Hàm chính để tải mô hình và thực hiện sinh văn bản tương tác hoặc theo batch.

    Args:
        ckpt_path (str): Đường dẫn đến thư mục checkpoint mô hình.
        config (str): Đường dẫn đến file cấu hình mô hình.
        input_file (str, optional): Đường dẫn đến file chứa prompt đầu vào. Mặc định "".
        interactive (bool, optional): Có chạy ở chế độ tương tác không. Mặc định True.
        max_new_tokens (int, optional): Số token mới tối đa cần sinh. Mặc định 100.
        temperature (float, optional): Nhiệt độ cho việc lấy mẫu. Mặc định 1.0.
        compile_model (bool, optional): Bật torch.compile để tối ưu JIT. Mặc định False.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    # === Global precision & performance flags ===
    # TF32 cho matmul (~1.5-2x nhanh hơn FP32 trên GPU Ampere+)
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("Nexus")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    # Bật torch.compile nếu được yêu cầu
    if compile_model:
        print("Enabling torch.compile (mode=reduce-overhead)...")
        model.compile_if_enabled()

    if interactive:
        import sys
        messages = []
        start_time = time.time()
        token_count = 0
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                if token_count > 0:
                    elapsed = time.time() - start_time
                    print(f"[Stats] {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} tok/s)")
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            elif prompt == "/stats":
                if token_count > 0:
                    elapsed = time.time() - start_time
                    print(f"[Stats] {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} tok/s)")
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
            token_count += len(completion_tokens[0])
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        start_time = time.time()
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        elapsed = time.time() - start_time
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        total_tokens = sum(len(t) for t in completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()
        if total_tokens > 0:
            print(f"[Stats] {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s)")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Giao diện dòng lệnh cho sinh văn bản phân tán.

    Arguments:
        --ckpt-path (str): Đường dẫn đến thư mục checkpoint mô hình.
        --config (str): Đường dẫn đến file cấu hình mô hình.
        --input-file (str, optional): File chứa prompt cho xử lý theo batch.
        --interactive (bool, optional): Bật chế độ tương tác để sinh văn bản.
        --max-new-tokens (int, optional): Số token mới tối đa cần sinh. Mặc định 200.
        --temperature (float, optional): Nhiệt độ cho việc lấy mẫu. Mặc định 0.2.
        --compile (bool, optional): Bật torch.compile để tối ưu JIT.

    Raises:
        AssertionError: Nếu không chỉ định input-file hoặc chế độ tương tác.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.compile)

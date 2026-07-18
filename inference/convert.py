import os
import shutil
from argparse import ArgumentParser
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# Ánh xạ tên tham số từ định dạng HuggingFace sang định dạng nội bộ
# Mỗi mục: (tên_nội_bộ, chiều_phân_mảnh)
#   - chiều_phân_mảnh = 0: chia theo chiều đầu ra (column)
#   - chiều_phân_mảnh = 1: chia theo chiều đầu vào (row)
#   - chiều_phân_mảnh = None: không chia, giữ nguyên
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def _rename_key(name: str) -> str:
    """Đổi tên key từ định dạng HF sang nội bộ."""
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    return name


def _process_shard(args):
    """
    Xử lý một shard cho model parallelism: áp dụng narrow và lưu file kết quả.

    Trả về (idx, state_dict) để giảm peak memory nếu cần lưu tuần tự.
    """
    file_path, hf_ckpt_path, n_local_experts, mp, i, mapping = args
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if "model.layers.61" in name:
                continue
            param: torch.Tensor = f.get_tensor(name)
            orig_name = name
            name = _rename_key(name)
            key = name.split(".")[-2]
            assert key in mapping, f"Key {key} not found in mapping"
            new_key, dim = mapping[key]
            name = name.replace(key, new_key)

            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                    continue
            elif dim is not None:
                assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                shard_size = param.size(dim) // mp
                param = param.narrow(dim, i * shard_size, shard_size).contiguous()
            state_dict[name] = param
    return state_dict


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Chuyển đổi và lưu các file checkpoint của mô hình sang định dạng chỉ định.

    Args:
        hf_ckpt_path (str): Đường dẫn đến thư mục chứa các file checkpoint đầu vào.
        save_path (str): Đường dẫn đến thư mục nơi các file checkpoint đã chuyển đổi sẽ được lưu.
        n_experts (int): Tổng số expert trong mô hình.
        mp (int): Hệ số song song hóa mô hình (model parallelism).

    Returns:
        None
    """
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    n_local_experts = n_experts // mp
    os.makedirs(save_path, exist_ok=True)

    safetensor_files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))

    # Tối ưu 1: Với mỗi file, xử lý song song các shard mp bằng ThreadPoolExecutor
    # Giảm số lần đọc file xuống còn 1 lần/file thay vì mp lần/file
    for file_path in tqdm(safetensor_files, desc="Processing shards"):
        # Xử lý tất cả mp shard từ 1 lần đọc file duy nhất
        state_dicts = [{} for _ in range(mp)]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                orig_name = name
                name = _rename_key(name)
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)

                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

        # Lưu song song các shard bằng ThreadPoolExecutor
        def _save_shard(i, sd):
            save_file(sd, os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

        with ThreadPoolExecutor(max_workers=min(mp, 4)) as executor:
            futures = []
            for i in range(mp):
                if state_dicts[i]:
                    save_path_i = os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
                    futures.append(executor.submit(save_file, state_dicts[i], save_path_i))
            for f in as_completed(futures):
                f.result()  # propagate exceptions

    # Copy tokenizer files
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    parser.add_argument("--fast", action="store_true", help="Enable parallel file processing")
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)

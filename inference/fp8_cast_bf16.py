import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant


@torch.no_grad()
def _dequant_single_file(safetensor_file, fp8_path, weight_map, bf16_path):
    """
    Xử lý một file safetensor: load → dequant FP8 weights → save.

    Args:
        safetensor_file (str): Đường dẫn file safetensor.
        fp8_path (str): Thư mục gốc FP8.
        weight_map (dict): Ánh xạ tensor_name → file_name.
        bf16_path (str): Thư mục đầu ra BF16.

    Returns:
        Tuple[list, list]: (danh sách tên weight đã dequant, danh sách tensor cần kiểm tra)
    """
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cuda")

    new_state_dict = {}
    fp8_in_file = []
    scale_inv_cache = {}

    for weight_name, weight in current_state_dict.items():
        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1:  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            # Cache scale_inv để tránh nhiều lần load file
            if scale_inv_name not in scale_inv_cache:
                file_path = os.path.join(fp8_path, weight_map.get(scale_inv_name, ""))
                if os.path.exists(file_path):
                    scale_inv_cache[scale_inv_name] = load_file(file_path, device="cuda").get(scale_inv_name)
                else:
                    scale_inv_cache[scale_inv_name] = None
            scale_inv = scale_inv_cache[scale_inv_name]
            if scale_inv is not None:
                fp8_in_file.append(weight_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            else:
                print(f"Cảnh báo: Thiếu tensor scale_inv cho {weight_name}, bỏ qua chuyển đổi")
                new_state_dict[weight_name] = weight
        else:
            new_state_dict[weight_name] = weight

    new_safetensor_file = os.path.join(bf16_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    del current_state_dict, new_state_dict
    torch.cuda.empty_cache()

    return fp8_in_file


def main(fp8_path, bf16_path):
    """
    Chuyển đổi trọng số FP8 sang BF16 và lưu các trọng số đã chuyển đổi.

    Hàm này đọc trọng số FP8 từ thư mục chỉ định, chuyển đổi chúng sang BF16,
    và lưu các trọng số đã chuyển đổi vào thư mục khác. Đồng thời cập nhật
    file index của mô hình để phản ánh các thay đổi.

    Tối ưu: xử lý từng file một — load → dequant → save → giải phóng GPU ngay lập tức,
    giúp giảm peak memory xuống mức thấp nhất.

    Args:
    fp8_path (str): Đường dẫn đến thư mục chứa trọng số FP8 và file index mô hình.
    bf16_path (str): Đường dẫn đến thư mục nơi các trọng số BF16 đã chuyển đổi sẽ được lưu.

    Raises:
    KeyError: Nếu thiếu tensor scale_inv cần thiết cho một trọng số.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    torch.set_float32_matmul_precision('high')

    # Load weight_map từ index file
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    all_fp8_weights = []

    # Xử lý từng file — không cache cross-file để giảm peak memory
    for safetensor_file in tqdm(safetensor_files):
        fp8_weights = _dequant_single_file(safetensor_file, fp8_path, weight_map, bf16_path)
        all_fp8_weights.extend(fp8_weights)

    # Cập nhật index mô hình
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in all_fp8_weights:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)

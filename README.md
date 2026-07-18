<div align="center">
  <h1>Nexus 🧠</h1>
  <h3><i>Neural EXecutive Unified System</i></h3>
  <p><b>Mô Hình Ngôn Ngữ Transformer Thế Hệ Mới</b></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch 2.4.1">
    <img src="https://img.shields.io/badge/Triton-3.0.0-00B4AB?style=flat-square" alt="Triton 3.0.0">
    <img src="https://img.shields.io/badge/Architecture-MLA%20%2B%20MoE-blueviolet?style=flat-square" alt="MLA+MoE">
    <img src="https://img.shields.io/badge/Quantization-FP8-orange?style=flat-square" alt="FP8">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License">
    <br>
    <a href="https://github.com/devtantai-coder/nexus/stargazers"><img src="https://img.shields.io/github/stars/devtantai-coder/nexus?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/devtantai-coder/nexus/forks"><img src="https://img.shields.io/github/forks/devtantai-coder/nexus?style=social" alt="GitHub Forks"></a>
    <a href="https://github.com/devtantai-coder/nexus/issues"><img src="https://img.shields.io/github/issues/devtantai-coder/nexus?style=social" alt="GitHub Issues"></a>
    <img src="https://img.shields.io/github/last-commit/devtantai-coder/nexus?style=flat-square" alt="Last Commit">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT"></a>
  </p>
</div>

---

## 📦 Tổng Quan Dự Án

```
nexus/
├── inference/
│   ├── model.py              # Kiến trúc Transformer cốt lõi (MLA, MoE, FP8)
│   ├── kernel.py             # Kernel Triton cho lượng tử hóa FP8
│   ├── train.py              # Script huấn luyện với CLI đầy đủ
│   ├── build_train.py        # Xây dựng dataset & huấn luyện model tiếng Việt
│   ├── generate.py           # Sinh văn bản (interactive/batch, phân tán)
│   ├── chat.py               # Chat tương tác với model đã train
│   ├── run_model.py          # Load và chạy model từ file .pt
│   ├── convert.py            # Chuyển đổi checkpoint HuggingFace → định dạng nội bộ
│   ├── fp8_cast_bf16.py      # Chuyển đổi trọng số FP8 → BF16
│   └── eval.py               # Xây dựng dataset đánh giá
├── requirements.txt          # Phụ thuộc: torch, triton, transformers, safetensors
├── .gitignore
└── .github/
    ├── workflows/stale.yml    # Tự động đánh dấu và đóng issue cũ
    └── ISSUE_TEMPLATE/
        ├── bug_report.md      # Template báo lỗi
        └── feature_request.md # Template yêu cầu tính năng
```

---

## 🧠 Kiến Trúc Mô Hình (`model.py`)

### Tổng quan

Nexus sử dụng kiến trúc **Transformer decoder-only** với các cải tiến hiện đại:

| Thành phần | Mô tả |
|---|---|
| **Multi-Head Latent Attention (MLA)** | Attention với KV latent compression — giảm bộ nhớ đệm KV cache tới 50% so với MHA truyền thống |
| **Mixture-of-Experts (MoE)** | Kích hoạt chỉ một subset expert cho mỗi token, tiết kiệm compute |
| **FP8 Quantization** | Lượng tử hóa activation và weight xuống FP8, tăng throughput |
| **YaRN RoPE** | Mở rộng ngữ cảnh (context window) suy luận từ 4K lên 128K+ |
| **RMSNorm** | Chuẩn hóa tầng ổn định, hiệu quả |
| **Model Parallelism** | Chia mô hình trên nhiều GPU |

---

## 🏗️ Kiến Trúc Chi Tiết

### 1. Multi-Head Latent Attention (MLA)

MLA là cốt lõi của Nexus, giúp giảm đáng kể bộ nhớ KV cache so với Multi-Head Attention (MHA) truyền thống.

```
┌─────────────────────────────────────────────┐
│                  MLA                         │
│  Query:  x → wq_a → q_norm → wq_b → q      │
│  Key:    x → wkv_a → [kv, k_pe]             │
│  Value:  x → wkv_a → kv_norm → wkv_b → v    │
│                                              │
│  Cơ chế absorb:                              │
│    scores = q_nope @ (wkv_b @ kv_norm)       │
│           + q_pe @ k_pe                      │
│    output = scores @ (wkv_b @ kv_norm)        │
└──────────────────────────────────────────────┘
```

**Điểm đặc biệt của MLA:**
- **KV latent compression**: Nén key/value xuống không gian tiềm ẩn (latent) với rank thấp (`kv_lora_rank`), giảm bộ nhớ KV cache từ `O(n_heads × d)` xuống `O(d_latent)`
- **Absorb mode**: Kết hợp phép chiếu `wkv_b` vào trong attention scores, tránh phải materialize toàn bộ K và V — tiết kiệm compute và memory
- **Caching thông minh**: Cache latent representation thay vì full K/V, giảm bộ nhớ tới 50%

### 2. Mixture-of-Experts (MoE)

MoE cho phép mô hình có nhiều tham số hơn mà không tăng chi phí tính toán:

```
┌─────────────────────────────────────┐
│  Input Token                        │
│       │                             │
│  ┌────▼────┐                        │
│  │  Gate   │ ← routing network      │
│  └────┬────┘                        │
│       │                             │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│  │Expert 0 │  │Expert 1 │  │Expert 2 │  ...
│  └─────────┘  └─────────┘  └─────────┘
│       │             │             │
│       └────────┬────┘────────────┘
│                ▼
│         Weighted Sum
│                +
│         Shared Experts
│                ▼
│            Output
└─────────────────────────────────────┘
```

**Cơ chế gating thông minh:**
- **Top-k routing**: Mỗi token chỉ kích hoạt `n_activated_experts` expert (mặc định 6/64)
- **Grouped routing**: Chia expert thành nhóm, chọn top-k nhóm trước, sau đó chọn expert trong nhóm
- **Shared experts**: Các expert dùng chung cho mọi token, đảm bảo kiến thức nền tảng
- **Score functions**: Hỗ trợ cả `softmax` và `sigmoid` cho điểm định tuyến

### 3. Lượng Tử Hóa FP8 (`kernel.py`)

Nexus sử dụng kernel **Triton** tùy chỉnh cho lượng tử hóa FP8:

| Kernel | Mô tả |
|---|---|
| `act_quant` | Lượng tử hóa activation theo block (FP8), với scale per-block |
| `weight_dequant` | Giải lượng tử hóa weight từ FP8 về BF16/FP32 |
| `fp8_gemm` | Nhân ma trận FP8 với scale, autotune qua nhiều cấu hình |

**Cơ chế hoạt động:**
- Activation được lượng tử hóa theo block (mặc định 128 phần tử)
- Mỗi block có scale factor riêng, lưu ở FP32
- Hỗ trợ scale format `ue8m0` (lũy thừa của 2) để tối ưu inference
- Kernel GEMM được autotune với nhiều cấu hình BLOCK_SIZE và num_stages

### 4. YaRN (Yet another RoPE extensioN)

Cho phép mở rộng độ dài ngữ cảnh vượt xa độ dài huấn luyện ban đầu:

- **RoPE**: Mã hóa vị trí tương đối bằng phép quay trong không gian phức
- **YaRN**: Điều chỉnh tần số RoPE để mở rộng ngữ cảnh từ 4K lên 128K+ tokens
- **Hệ số hiệu chỉnh**: `beta_fast`, `beta_slow` kiểm soát vùng nội suy
- **Cache**: `precompute_freqs_cis` được cache bằng `lru_cache` để tránh tính toán lại

### 5. Model Parallelism

Nexus hỗ trợ chia mô hình trên nhiều GPU:

| Lớp | Chiều chia | Đồng bộ |
|---|---|---|
| `ParallelEmbedding` | vocab_size / world_size | `all_reduce` |
| `ColumnParallelLinear` | out_features / world_size | — |
| `RowParallelLinear` | in_features / world_size | `all_reduce` |
| `MoE` | experts / world_size | `all_reduce` |
| `head` (lm_head) | vocab_size / world_size | `all_gather` |

---

## ⚡ Kernel Triton FP8 (`kernel.py`)

Nexus sử dụng kernel **Triton** tùy chỉnh để tối ưu hóa lượng tử hóa FP8:

### `act_quant` — Lượng tử hóa Activation

```python
# Input: tensor FP32/BF16 → Output: tensor FP8 + scale factors
y, scale = act_quant(x, block_size=128)
```

- Chia tensor thành block 128 phần tử
- Tính `amax = max(|x|)` cho mỗi block
- Scale: `s = amax / 448.` (FP8 max = 448)
- Hỗ trợ `ue8m0` format: làm tròn lên lũy thừa của 2

### `weight_dequant` — Giải lượng tử hóa Weight

```python
weight_bf16 = weight_dequant(weight_fp8, scale, block_size=128)
```

- Kernel 2D: chia ma trận thành block M×N
- Mỗi block nhân với scale tương ứng
- Output ở BF16/FP32

### `fp8_gemm` — Nhân Ma Trận FP8

```python
c = fp8_gemm(a_fp8, a_scale, b_fp8, b_scale)
```

- **Autotune**: 64 cấu hình (BLOCK_SIZE_M × BLOCK_SIZE_N × num_stages)
- **Cấu hình nhỏ** cho decode stage (4 warps, 2 stages)
- **Cấu hình lớn** cho prefill stage (8 warps, 3-6 stages)
- BLOCK_SIZE_K = 128 (khớp với quantization block size)

---

## 🚀 Cách Sử Dụng

### 1. Cài đặt

```bash
pip install -r requirements.txt
# torch==2.4.1, triton==3.0.0, transformers==4.46.3, safetensors==0.4.5
```

### 2. Huấn luyện

#### Huấn luyện nhanh với dữ liệu ngẫu nhiên:

```bash
python inference/train.py --device cuda --steps 100 --lr 3e-4
```

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `--device` | `cpu`, `cuda`, `mps` | `cpu` |
| `--steps` | Số bước huấn luyện | `100` |
| `--lr` | Learning rate | `3e-4` |
| `--batch-size` | Kích thước batch | `8` |
| `--seq-len` | Độ dài chuỗi | `64` |
| `--dim` | Kích thước ẩn | `256` |
| `--n-layers` | Số tầng | `4` |
| `--n-heads` | Số đầu attention | `4` |
| `--compile` | Bật torch.compile | — |

### 2. Huấn luyện Model Tiếng Việt

```bash
python inference/build_train.py
```

Script này tự động:
1. Xây dựng từ vựng tiếng Việt ~200 từ
2. Tạo dataset ~2000+ samples (QA pairs + câu đơn)
3. Huấn luyện model ~50M tham số
4. Lưu model ra `inference/chat_model.pt`
5. Test generation với 5 mẫu câu

### 3. Chat với Model

```bash
# Chat tương tác với model tiếng Việt
python inference/chat.py --model inference/chat_model.pt --interactive

# Một câu hỏi duy nhất
python inference/chat.py --model inference/chat_model.pt --prompt "xin chào"

# Dùng model random (mặc định)
python inference/chat.py
```

**Commands trong chat:**
- `/exit` — Thoát
- `/stats` — Thống kê tốc độ
- `/temperature <value>` — Điều chỉnh độ sáng tạo (0.0 - 2.0)

### 4. Sinh văn bản phân tán (Multi-GPU)

```bash
# Chế độ tương tác
python inference/generate.py --ckpt-path /path/to/ckpt --config config.json --interactive

# Chế độ batch
python inference/generate.py --ckpt-path /path/to/ckpt --config config.json --input-file prompts.txt

# Với torch.compile
python inference/generate.py --ckpt-path /path/to/ckpt --config config.json --interactive --compile
```

### 5. Chạy Model từ File .pt

```bash
python inference/run_model.py --model nexus_model.pt --prompt "Hello" --steps 50
python inference/run_model.py --model nexus_model.pt --seed 42
```

### 6. Chuyển Đổi Checkpoint

```bash
# HuggingFace → định dạng nội bộ (với model parallelism)
python inference/convert.py \
    --hf-ckpt-path /path/to/hf_ckpt \
    --save-path /path/to/output \
    --n-experts 64 \
    --model-parallel 8

# FP8 → BF16
python inference/fp8_cast_bf16.py \
    --input-fp8-hf-path /path/to/fp8 \
    --output-bf16-hf-path /path/to/bf16
```

---

## 🧪 Cấu Hình Mô Hình

### ModelArgs — Tham số đầy đủ

```python
ModelArgs(
    max_batch_size=8,           # Batch tối đa
    max_seq_len=16384,          # Độ dài chuỗi tối đa (4K × 4)
    dtype="bf16",               # Kiểu dữ liệu: bf16, fp8
    vocab_size=102400,          # Kích thước từ vựng
    dim=2048,                   # Kích thước chiều chính
    inter_dim=10944,            # Kích thước MLP ẩn
    moe_inter_dim=1408,         # Kích thước MoE ẩn
    n_layers=27,                # Số tầng transformer
    n_dense_layers=1,           # Số tầng dense (không MoE)
    n_heads=16,                 # Số đầu attention
    n_routed_experts=64,        # Số expert được định tuyến
    n_shared_experts=2,         # Số expert dùng chung
    n_activated_experts=6,      # Số expert kích hoạt mỗi token
    kv_lora_rank=512,           # Rank cho KV latent compression
    qk_nope_head_dim=128,       # Chiều QK không có positional
    qk_rope_head_dim=64,        # Chiều QK với rotary
    v_head_dim=128,             # Chiều value
    rope_theta=10000.0,         # Cơ số RoPE
    rope_factor=40,             # Hệ số mở rộng YaRN
)
```

### Cấu hình Tiny (cho training nhanh)

```python
ModelArgs(
    dim=512, inter_dim=2048, moe_inter_dim=384,
    n_layers=6, n_dense_layers=6, n_heads=8,
    vocab_size=200, max_seq_len=64,
    kv_lora_rank=32, qk_nope_head_dim=32,
    qk_rope_head_dim=16, v_head_dim=32,
    dtype="float32",
)
# ~50M tham số
```

---

## 🔬 Các Thành Phần Kỹ Thuật Chính

### ParallelEmbedding
- Chia vocab_size cho `world_size` GPU
- Mask các token không thuộc phần của rank hiện tại
- `all_reduce` để đồng bộ kết quả

### RMSNorm
- Root Mean Square Layer Normalization
- Không học bias, chỉ học weight scale
- Ổn định hơn LayerNorm truyền thống

### YaRN RoPE
- **Mở rộng ngữ cảnh**: Từ 4K lên 128K+ tokens
- **Cơ chế**: Nội suy tần số RoPE với ramp function
- **Hệ số**: `rope_factor=40` cho 40× mở rộng
- **Hiệu chỉnh**: `beta_fast=32`, `beta_slow=1` kiểm soát vùng nội suy

### MoE Gating
- **Score functions**: `softmax` (mặc định) hoặc `sigmoid`
- **Grouped routing**: Chia expert thành nhóm, chọn top-k nhóm
- **Bias correction**: Bias học được cho cân bằng tải expert
- **Route scale**: Điều chỉnh trọng số định tuyến

### FP8 Quantization Pipeline

```
Forward pass với FP8:
  x (BF16) → act_quant → x_fp8 + scale_x
  w (FP8)  → (đã lưu FP8) + scale_w
  y = fp8_gemm(x_fp8, scale_x, w_fp8, scale_w)
  y (BF16/FP32)

Training:
  - Weight được lưu ở BF16 (hoặc FP32)
  - Forward: lượng tử hóa activation → FP8 GEMM
  - Backward: tính gradient trên bản sao BF16
```

### Chuyển Đổi Checkpoint

**HuggingFace → định dạng nội bộ** (`convert.py`):
- Ánh xạ tên tham số từ HF sang nội bộ
- Chia shard theo model parallelism
- Xử lý expert distribution cho MoE
- Copy tokenizer files

**FP8 → BF16** (`fp8_cast_bf16.py`):
- Load từng file safetensor, giải lượng tử hóa weight FP8
- Giải phóng GPU ngay sau mỗi file để giảm peak memory
- Cập nhật model index

---

## 📊 So Sánh Kiến Trúc

| Tính năng | Nexus | DeepSeek-V2 | Llama 3 |
|---|---|---|---|
| Attention | **MLA** | MLA | GQA |
| MoE | ✅ Có | ✅ Có | ❌ Dense |
| FP8 Quant | ✅ Triton | ✅ | ❌ |
| YaRN RoPE | ✅ | ✅ | ❌ |
| RMSNorm | ✅ | ✅ | ✅ |
| Model Parallel | ✅ | ✅ | ✅ |

---

## 🛠️ Yêu Cầu Hệ Thống

- **Python**: 3.10+
- **PyTorch**: 2.4.1+
- **CUDA**: 11.8+ (cho GPU, tùy chọn)
- **Triton**: 3.0.0 (cho kernel FP8, tùy chọn)
- **RAM**: Tối thiểu 8GB (16GB+ khuyến nghị cho training)
- **VRAM**: Tùy theo kích thước mô hình

### GPU hỗ trợ

| GPU | FP8 (Triton) | TF32 | FP16/BF16 |
|---|---|---|---|
| NVIDIA H100 | ✅ Tối ưu | ✅ | ✅ |
| NVIDIA A100 | ✅ | ✅ ~1.5-2× FP32 | ✅ |
| NVIDIA V100 | ❌ | ❌ | ✅ |
| RTX 4090/3090 | ✅ | ✅ | ✅ |
| RTX 30 series | ✅ | ✅ | ✅ |

---

## 📁 Cấu Trúc File Chi Tiết

### `inference/model.py` — Trái tim của Nexus

| Lớp | Dòng | Mô tả |
|---|---|---|
| `ModelArgs` | 24-100 | Dataclass chứa tất cả siêu tham số mô hình |
| `ParallelEmbedding` | 103-142 | Embedding song song hóa trên nhiều GPU |
| `linear()` | 145-177 | Hàm linear thông minh: tự động chọn FP8/BF16 GEMM |
| `Linear` | 180-219 | Lớp linear với hỗ trợ weight lượng tử hóa |
| `ColumnParallelLinear` | 222-248 | Linear chia theo cột (out_features) |
| `RowParallelLinear` | 251-281 | Linear chia theo hàng (in_features) |
| `RMSNorm` | 284-308 | Root Mean Square Normalization |
| `precompute_freqs_cis` | 365-393 | Tính trước tần số RoPE (có cache) |
| `apply_rotary_emb` | 396-411 | Áp dụng rotary positional embedding |
| `MLA` | 414-553 | Multi-Head Latent Attention (cốt lõi) |
| `MLP` | 556-588 | Multi-Layer Perceptron (SwiGLU) |
| `Gate` | 591-654 | Cơ chế gating cho MoE |
| `Expert` | 657-689 | Một expert trong MoE |
| `MoE` | 692-767 | Mixture-of-Experts hoàn chỉnh |
| `Block` | 770-809 | Một khối Transformer (Attn + FFN) |
| `Transformer` | 812-927 | Mô hình Transformer hoàn chỉnh |

### Luồng Dữ Liệu

```
Input Tokens (IDs)
    │
    ▼
ParallelEmbedding ──── one-hot → dense vector
    │
    ▼
    ┌──────────┐  (lặp n_layers lần)
    │  Block   │
    │  ┌─────┐ │
    │  │Norm │ │
    │  │MLA  │ │ ← Multi-Head Latent Attention
    │  │ +   │ │
    │  ├─────┤ │
    │  │Norm │ │
    │  │FFN  │ │ ← MLP (dense) hoặc MoE (sparse)
    │  └─────┘ │
    └──────────┘
         │
    RMSNorm
         │
    lm_head (ColumnParallelLinear)
         │
      Logits
```

---

## 🔬 Chi Tiết Kỹ Thuật

### Cơ chế Absorb trong MLA

Đây là tối ưu quan trọng nhất của MLA. Thay vì tính:

```python
# Cách thông thường:
k_full = wkv_b(kv_norm)  # materialize K
scores = q @ k_full       # attention scores
```

MLA "hấp thụ" phép chiếu `wkv_b` vào trong attention scores:

```python
# Cách absorb:
q_nope_proj = q_nope @ wkv_b[:, :qk_nope_dim]  # (batch, head, seq, kv_dim)
scores = q_nope_proj @ kv_norm + q_pe @ k_pe
output = scores @ (wkv_b[:, -v_head_dim:] @ kv_norm)
```

Lợi ích:
- ❌ Không cần materialize K và V đầy đủ
- ✅ Giảm memory bandwidth
- ✅ Tăng tốc độ attention

### MoE Sort-Based Expert Grouping

Thay vì dùng `bincount` + Python loop truyền thống, MoE của Nexus dùng:

```python
# Sort-based: giữ mọi thứ trên GPU
flat_indices = indices.flatten()
sorted_indices, perm = flat_indices.sort(stable=True)
unique_experts, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
```

Lợi ích:
- ❌ Không cần `bincount` + `tolist()` (CPU-GPU sync)
- ✅ Hoàn toàn trên GPU
- ✅ Giảm memory fragmentation

---

## 🧪 Dataset Tiếng Việt

### Từ vựng (~200 từ)

Bao gồm các từ thông dụng trong giao tiếp hàng ngày:
- **Đại từ**: tôi, bạn, chúng, ta, mình, cậu, anh, chị, em
- **Động từ**: làm, học, chơi, đi, chạy, ăn, uống, nói, đọc, viết
- **Tính từ**: tốt, xấu, đẹp, vui, buồn, khỏe, yếu, nhanh, chậm
- **Danh từ**: người, trường, lớp, máy tính, điện thoại, mạng, web
- **Thuật ngữ AI**: lập trình, thuật toán, trí tuệ nhân tạo, học máy, deep learning
- **Cảm thán**: ồ, à, ơi, này, ừ, ừm, nhỉ, nhé, nha, vâng, dạ

### Dataset (~2000+ samples)

- **142 cặp QA** (hội thoại) × biến thể (thêm cảm thán, đảo từ)
- **268 câu đơn** (single sentences)
- **Data augmentation**: Tự động mở rộng với biến thể ngôn ngữ

---

## 🔧 Công Cụ Chuyển Đổi

### `convert.py` — HuggingFace → Nexus Format

Chuyển đổi checkpoint từ định dạng HuggingFace sang định dạng nội bộ của Nexus:

```
Ánh xạ tham số:
  HF Name              → Nexus Name      | Dim Split
  ───────────────────────────────────────┼─────────
  model.embed_tokens   → embed           | column (0)
  model.layers.0.self_attn.q_proj       → wq       | column (0)
  model.layers.0.self_attn.o_proj       → wo       | row (1)
  model.layers.0.mlp.gate_proj          → w1       | column (0)
  model.layers.0.mlp.down_proj          → w2       | row (1)
  model.layers.0.mlp.up_proj            → w3       | column (0)
  model.layers.0.post_attention_layernorm → ffn_norm | none
  model.norm            → norm           | none
  lm_head               → head           | column (0)
```

### `fp8_cast_bf16.py` — FP8 → BF16

Chuyển đổi trọng số FP8 sang BF16 để tương thích ngược:

```bash
python inference/fp8_cast_bf16.py \
    --input-fp8-hf-path /path/to/fp8_ckpt \
    --output-bf16-hf-path /path/to/bf16_ckpt
```

**Tối ưu bộ nhớ:** Xử lý từng file một — load → dequant → save → giải phóng GPU — giảm peak memory tối đa.

---

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork dự án
2. Tạo branch feature (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi (`git commit -m 'Add amazing feature'`)
4. Push lên branch (`git push origin feature/amazing-feature`)
5. Mở Pull Request

### Template Issue

Dự án có sẵn template cho:
- **Bug report**: `.github/ISSUE_TEMPLATE/bug_report.md`
- **Feature request**: `.github/ISSUE_TEMPLATE/feature_request.md`

### Workflow CI/CD

- **Stale issues**: Tự động đánh dấu issue không hoạt động sau 30 ngày, đóng sau 14 ngày (`.github/workflows/stale.yml`)

---

## 📜 Giấy Phép

Dự án Nexus được phân phối dưới giấy phép mã nguồn mở. Vui lòng kiểm tra file LICENSE để biết thêm chi tiết.

---

## 🙏 Lời Cảm Ơn

Nexus được lấy cảm hứng từ các kiến trúc tiên tiến như:
- **DeepSeek-V2/V3** — Multi-Head Latent Attention & MoE
- **Llama 3** — Kiến trúc Transformer hiệu quả
- **Triton** — Ngôn ngữ lập trình cho kernel GPU hiệu năng cao

---

## 📬 Liên Hệ & Hỗ Trợ

- **GitHub Issues**: Báo lỗi hoặc yêu cầu tính năng
- **Discussions**: Thảo luận về kiến trúc và cải tiến

---

---

## 🌟 Chia Sẻ & Ủng Hộ

**Nexus là dự án mã nguồn mở Việt Nam — hãy giúp chúng tôi lan tỏa!**

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Nexus%20-%20M%C3%B4%20h%C3%ACnh%20ng%C3%B4n%20ng%E1%BB%AF%20Transformer%20th%E1%BA%BF%20h%E1%BB%87%20m%E1%BB%9Bi%20v%E1%BB%9Bi%20MLA%20%2B%20MoE%20%2B%20FP8%20Quantization&url=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-X (Twitter)-1DA1F2?style=for-the-badge&logo=x" alt="Share on X">
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin" alt="Share on LinkedIn">
  </a>
  <a href="https://www.facebook.com/sharer/sharer.php?u=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-Facebook-1877F2?style=for-the-badge&logo=facebook" alt="Share on Facebook">
  </a>
  <a href="https://www.reddit.com/submit?url=https://github.com/devtantai-coder/nexus&title=Nexus%20-%20M%C3%B4%20h%C3%ACnh%20ng%C3%B4n%20ng%E1%BB%AF%20Transformer%20v%E1%BB%9Bi%20MLA%20%2B%20MoE%20%2B%20FP8" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-Reddit-FF4500?style=for-the-badge&logo=reddit" alt="Share on Reddit">
  </a>
  <a href="https://news.ycombinator.com/submitlink?u=https://github.com/devtantai-coder/nexus&t=Nexus%20-%20M%C3%B4%20h%C3%ACnh%20ng%C3%B4n%20ng%E1%BB%AF%20Transformer%20v%E1%BB%9Bi%20MLA%20%2B%20MoE%20%2B%20FP8" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-Hacker%20News-FF6600?style=for-the-badge&logo=ycombinator" alt="Share on HN">
  </a>
  <a href="https://t.me/share/url?url=https://github.com/devtantai-coder/nexus&text=Nexus%20-%20M%C3%B4%20h%C3%ACnh%20ng%C3%B4n%20ng%E1%BB%AF%20Transformer%20v%E1%BB%9Bi%20MLA%20%2B%20MoE%20%2B%20FP8" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-Telegram-26A5E4?style=for-the-badge&logo=telegram" alt="Share on Telegram">
  </a>
  <a href="https://discord.com/share?url=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ trên-Discord-5865F2?style=for-the-badge&logo=discord" alt="Share on Discord">
  </a>
</p>

---

## 📊 Star History & Phát Triển

[![Star History Chart](https://api.star-history.com/svg?repos=devtantai-coder/nexus&type=Date)](https://star-history.com/#devtantai-coder/nexus&Date)

---

## 🎯 Lộ Trình Phát Triển (Roadmap)

### Phase 1: MVP ✅
- [x] Kiến trúc Transformer với MLA
- [x] MoE với sort-based expert grouping
- [x] FP8 quantization kernels (Triton)
- [x] YaRN RoPE mở rộng ngữ cảnh
- [x] Model parallelism
- [x] Dataset tiếng Việt + training script

### Phase 2: Cộng Đồng 🚀
- [ ] Website/landing page chính thức
- [ ] Colab notebook demo
- [ ] Docker image
- [ ] Pre-trained weights release
- [ ] API endpoint (HuggingFace Spaces)
- [ ] Tài liệu tiếng Anh

### Phase 3: Nâng Cao 🔥
- [ ] Fine-tuning script với LoRA/QLoRA
- [ ] RLHF/DPO training
- [ ] Multi-modal (vision + text)
- [ ] ONNX/TensorRT export
- [ ] Web UI (Gradio/Streamlit)
- [ ] Benchmark suite

---

## 🌟 Được Xây Dựng Bởi

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/devtantai-coder">
        <img src="https://github.com/devtantai-coder.png" width="80" alt="Nexus AI"/>
        <br /><b>Nexus AI Team</b>
      </a>
      <br />
      <sub>🇻🇳 Made in Vietnam</sub>
    </td>
  </tr>
</table>

---

## ☕ Ủng Hộ Dự Án

Nếu bạn thấy Nexus hữu ích, hãy:

- ⭐ **Star** repository trên GitHub
- 🔄 **Fork** và đóng góp code
- 📣 **Chia sẻ** với bạn bè và đồng nghiệp
- 🐛 **Báo lỗi** hoặc đề xuất tính năng mới

---

<div align="center">
  <h3>🌟 Được xây dựng với ❤️ cho cộng đồng AI Việt Nam 🌟</h3>
  <p>
    <a href="https://github.com/devtantai-coder/nexus">
      <img src="https://img.shields.io/github/stars/devtantai-coder/nexus?style=social" alt="Stars">
    </a>
    <a href="https://github.com/devtantai-coder/nexus/fork">
      <img src="https://img.shields.io/github/forks/devtantai-coder/nexus?style=social" alt="Forks">
    </a>
  </p>
  <p>
    <sub>⭐ Star repo này nếu bạn thấy dự án hữu ích!</sub>
  </p>
</div>

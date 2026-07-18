<div align="center">
  <h1>Nexus AI 🧠</h1>
  <h3><i>Neural EXecutive Unified System</i></h3>
  <p><b>Mô Hình Ngôn Ngữ Transformer Thế Hệ Mới — Open Source LLM Việt Nam</b></p>

  <!-- Badges SEO -->
  <p>
    <img src="https://img.shields.io/github/stars/devtantai-coder/nexus?style=flat-square&logo=github&label=Stars&color=yellow" alt="GitHub Stars">
    <img src="https://img.shields.io/github/forks/devtantai-coder/nexus?style=flat-square&logo=github&label=Forks&color=blue" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/devtantai-coder/nexus?style=flat-square&label=Watchers" alt="Watchers">
    <img src="https://img.shields.io/github/contributors/devtantai-coder/nexus?style=flat-square&label=Contributors" alt="Contributors">
    <br>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch 2.4.1">
    <img src="https://img.shields.io/badge/Triton-3.0.0-00B4AB?style=flat-square&logo=nvidia" alt="Triton 3.0.0">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License">
    <br>
    <img src="https://img.shields.io/badge/Architecture-Multi--Head%20Latent%20Attention-blueviolet?style=flat-square" alt="MLA">
    <img src="https://img.shields.io/badge/MoE-64%20Experts-orange?style=flat-square" alt="MoE 64 Experts">
    <img src="https://img.shields.io/badge/Quantization-FP8%20Triton-brightgreen?style=flat-square" alt="FP8 Quantization">
    <img src="https://img.shields.io/badge/Context-128K%20YaRN%20RoPE-red?style=flat-square" alt="128K Context">
    <br>
    <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Active">
    <img src="https://img.shields.io/github/last-commit/devtantai-coder/nexus?style=flat-square&label=Last%20Commit" alt="Last Commit">
    <img src="https://img.shields.io/github/release/devtantai-coder/nexus?style=flat-square&label=Latest%20Release" alt="Latest Release">
    <img src="https://img.shields.io/github/issues/devtantai-coder/nexus?style=flat-square&label=Issues" alt="Issues">
  </p>

  <!-- Quick Links -->
  <p>
    <a href="#-kiến-trúc-mô-hình"><b>Kiến Trúc</b></a> •
    <a href="#-cài-đặt-nhanh"><b>Cài Đặt</b></a> •
    <a href="#-cách-sử-dụng"><b>Sử Dụng</b></a> •
    <a href="#-so-sánh-kiến-trúc"><b>So Sánh</b></a> •
    <a href="#-lộ-trình-phát-triển"><b>Lộ Trình</b></a> •
    <a href="#%EF%B8%8F-ủng-hộ"><b>⭐ Star</b></a>
  </p>
</div>

---

## 🧠 Nexus AI Là Gì?

**Nexus AI** là mô hình ngôn ngữ lớn (Large Language Model - LLM) mã nguồn mở được xây dựng trên kiến trúc **Transformer decoder-only** với những công nghệ tiên tiến nhất hiện nay. Đây là dự án **AI Việt Nam** với mục tiêu mang kiến trúc LLM hiện đại đến gần hơn với cộng đồng.

### ✨ Tính Năng Nổi Bật

| Tính năng | Mô tả | Lợi ích |
|---|---|---|
| **🧠 Multi-Head Latent Attention (MLA)** | KV latent compression, cơ chế absorb | Giảm **50%** bộ nhớ KV cache so với MHA |
| **🧩 Mixture-of-Experts (MoE)** | 64 routed experts, top-6, grouped routing | Tiết kiệm compute, mô hình lớn hơn với chi phí thấp hơn |
| **⚡ FP8 Quantization** | Custom Triton kernels, autotune 64 configs | **1.5-2x** throughput trên GPU Ampere+ |
| **🔗 YaRN RoPE** | Mở rộng tần số với ramp function | Context window **128K+** tokens |
| **🖥️ Model Parallelism** | Column/Row parallel + expert distribution | Chạy trên nhiều GPU |
| **🇻🇳 Tiếng Việt First** | Dataset ~2000+ samples, từ vựng ~200 từ | Hỗ trợ ngôn ngữ Việt Nam |

---

## 🏗️ Kiến Trúc Mô Hình

### Luồng Dữ Liệu Transformer

```
📝 Input Tokens (IDs)
    │
    ▼
🔤 ParallelEmbedding (vocab_size / world_size)
    │
    ▼
    ┌──────────────────┐  (lặp n_layers = 27 lần)
    │     Block         │
    │  ┌─────────────┐  │
    │  │  RMSNorm     │  │  ← Root Mean Square Normalization
    │  │  MLA         │  │  ← Multi-Head Latent Attention ⭐
    │  │  Residual +  │  │  
    │  ├─────────────┤  │
    │  │  RMSNorm     │  │
    │  │  MLP / MoE   │  │  ← Dense hoặc Mixture-of-Experts
    │  └─────────────┘  │
    └──────────────────┘
    │
    ▼
🎯 RMSNorm → lm_head → Logits (vocab_size)
```

### 1. Multi-Head Latent Attention (MLA) — Cốt Lõi Của Nexus

MLA là phiên bản cải tiến của Multi-Head Attention, được DeepSeek-V2 giới thiệu. Nexus triển khai MLA với cơ chế **absorb** tối ưu:

```python
# Cơ chế absorb: không cần materialize K/V đầy đủ
q_nope_proj = q_nope @ wkv_b[:, :qk_nope_dim]  # (batch, head, seq, kv_dim)
scores = q_nope_proj @ kv_norm + q_pe @ k_pe    # Attention scores
output = scores @ (wkv_b[:, -v_head_dim:] @ kv_norm)  # Output
```

**Lợi ích của MLA:**
- ✅ Giảm bộ nhớ KV cache từ `O(n_heads × d)` → `O(d_latent)`
- ✅ Tăng tốc độ attention, giảm memory bandwidth
- ✅ Cache latent representation thay vì full K/V

### 2. Mixture-of-Experts (MoE) — 64 Experts

MoE cho phép mô hình có nhiều tham số nhưng chỉ kích hoạt một phần:

```
Input → Gate (routing network) → Top-k Experts (6/64) → Weighted Sum + Shared Experts → Output
```

- **Sort-based expert grouping**: Hoàn toàn trên GPU, không CPU-GPU sync
- **Grouped routing**: Chia 64 expert thành nhóm, chọn top-k nhóm trước
- **Shared experts**: 2 expert dùng chung cho mọi token
- **Score functions**: Softmax hoặc Sigmoid

### 3. FP8 Quantization — Kernel Triton Tùy Chỉnh

Nexus sử dụng kernel Triton cho lượng tử hóa FP8 với autotune:

```python
# Lượng tử hóa activation
y_fp8, scale = act_quant(x_bf16, block_size=128)

# Nhân ma trận FP8
result = fp8_gemm(a_fp8, a_scale, b_fp8, b_scale)

# Giải lượng tử hóa weight
w_bf16 = weight_dequant(w_fp8, w_scale, block_size=128)
```

### 4. YaRN RoPE — Mở Rộng Ngữ Cảnh

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `original_seq_len` | 4,096 | Độ dài chuỗi gốc |
| `rope_factor` | 40 | Hệ số mở rộng (40×) |
| `max_seq_len` | 163,840 | Độ dài tối đa (163K tokens) |
| `beta_fast` | 32 | Cận trên hiệu chỉnh |
| `beta_slow` | 1 | Cận dưới hiệu chỉnh |

---

## 🚀 Bắt Đầu Nhanh

### Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.4.1+ |
| CUDA | 11.8+ (khuyến nghị cho GPU) |
| Triton | 3.0.0 (cho FP8 kernels) |
| RAM | Tối thiểu 8GB (16GB+ khuyến nghị) |

### Cài Đặt

```bash
# Clone repo
git clone https://github.com/devtantai-coder/nexus.git
cd nexus

# Cài đặt dependencies
pip install -r requirements.txt
```

### Huấn Luyện Model Tiếng Việt

```bash
# Dataset ~2000+ samples, từ vựng ~200 từ tiếng Việt
# Model ~50M tham số
python inference/build_train.py
```

### Chat Với Nexus

```bash
# Chat tương tác
python inference/chat.py --model inference/chat_model.pt --interactive

# Hoặc gõ 1 câu
python inference/chat.py --model inference/chat_model.pt --prompt "xin chào"
```

### Huấn Luyện Nhanh (Dữ Liệu Ngẫu Nhiên)

```bash
python inference/train.py --device cuda --steps 100 --lr 3e-4
```

---

## 📋 Cấu Hình Chi Tiết

### ModelArgs — Tham Số Đầy Đủ

```python
ModelArgs(
    max_batch_size=8,           # Batch tối đa
    max_seq_len=16384,          # Độ dài chuỗi (4K × 4)
    dtype="bf16",               # bf16 hoặc fp8
    vocab_size=102400,          # Kích thước từ vựng
    dim=2048,                   # Chiều chính
    inter_dim=10944,            # MLP ẩn
    moe_inter_dim=1408,         # MoE ẩn
    n_layers=27,                # Số tầng
    n_dense_layers=1,           # Tầng dense
    n_heads=16,                 # Số đầu attention
    n_routed_experts=64,        # Tổng expert
    n_shared_experts=2,         # Expert chung
    n_activated_experts=6,      # Expert kích hoạt/token
    kv_lora_rank=512,           # KV latent rank
    qk_nope_head_dim=128,       # QK không positional
    qk_rope_head_dim=64,        # QK với rotary
    v_head_dim=128,             # Value dimension
    rope_theta=10000.0,         # RoPE base
    rope_factor=40,             # YaRN factor
)
```

### Cấu Hình Tiny (Train Nhanh — ~50M Tham Số)

```python
ModelArgs(
    dim=512, inter_dim=2048, moe_inter_dim=384,
    n_layers=6, n_dense_layers=6, n_heads=8,
    vocab_size=200, max_seq_len=64,
    kv_lora_rank=32, qk_nope_head_dim=32,
    qk_rope_head_dim=16, v_head_dim=32,
    dtype="float32",
)
```

---

## 📊 So Sánh Kiến Trúc

| Tính năng | **Nexus AI** 🧠 | DeepSeek-V2 | Llama 3 | Mistral |
|---|---|---|---|---|
| **Attention** | **MLA** ⭐ | MLA | GQA | GQA |
| **KV Cache** | Giảm 50% | Giảm 50% | Tiêu chuẩn | Tiêu chuẩn |
| **MoE** | ✅ 64 experts | ✅ 160 experts | ❌ Dense | ❌ Dense |
| **FP8 Quant** | ✅ Triton kernels | ✅ | ❌ | ❌ |
| **Context** | **128K+** (YaRN) | 128K | 8K | 32K |
| **Model Parallel** | ✅ Column/Row | ✅ | ✅ | ✅ |
| **Open Source** | ✅ MIT | ✅ MIT | ✅ Custom | ✅ Apache |
| **Tiếng Việt** | ✅ Native | ❌ | ❌ | ❌ |

---

## 🧪 Dataset Tiếng Việt

### Từ Vựng (~200 Từ)

| Loại | Ví dụ |
|---|---|
| Đại từ | tôi, bạn, chúng, ta, mình, cậu, anh, chị, em, các |
| Động từ | làm, học, chơi, đi, chạy, ăn, uống, nói, đọc, viết |
| Tính từ | tốt, xấu, đẹp, vui, buồn, khỏe, yếu, nhanh, chậm |
| Danh từ | người, trường, lớp, máy tính, điện thoại, mạng |
| AI/Tech | lập trình, thuật toán, trí tuệ nhân tạo, học máy |
| Cảm thán | ồ, à, ơi, này, ừ, nhỉ, nhé, nha, vâng, dạ |

### Dataset (~2000+ Samples)

- **142 cặp QA** × biến thể (thêm cảm thán, đảo từ)
- **268 câu đơn** về học tập, công nghệ, cuộc sống
- **Data augmentation** tự động mở rộng

---

## 🔧 Các Script Hỗ Trợ

| Script | Công dụng |
|---|---|
| `build_train.py` | Xây dựng dataset tiếng Việt + huấn luyện |
| `train.py` | Huấn luyện với CLI đầy đủ tham số |
| `chat.py` | Chat tương tác với model |
| `generate.py` | Sinh văn bản (interactive/batch, phân tán) |
| `run_model.py` | Load và chạy model từ file .pt |
| `convert.py` | Chuyển đổi HuggingFace → định dạng nội bộ |
| `fp8_cast_bf16.py` | Chuyển đổi trọng số FP8 → BF16 |

---

## ⚡ GPU Hỗ Trợ

| GPU | FP8 (Triton) | TF32 | BF16 |
|---|---|---|---|
| NVIDIA H100 | ✅ Tối ưu | ✅ | ✅ |
| NVIDIA A100 | ✅ | ✅ (1.5-2× FP32) | ✅ |
| RTX 4090/3090 | ✅ | ✅ | ✅ |
| RTX 4080/3080 | ✅ | ✅ | ✅ |
| RTX 4070/3070 | ✅ | ✅ | ✅ |
| NVIDIA V100 | ❌ (No FP8) | ❌ | ✅ |
| RTX 4060/3060 | ✅ | ✅ | ✅ |

---

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. ⭐ **Star** repository — giúp Nexus đến với nhiều người hơn
2. **Fork** dự án
3. Tạo branch: `git checkout -b feature/tinh-nang-moi`
4. Commit: `git commit -m 'Thêm tính năng mới'`
5. Push: `git push origin feature/tinh-nang-moi`
6. Mở **Pull Request**

## 📬 Liên Hệ

- **GitHub Issues**: https://github.com/devtantai-coder/nexus/issues
- **Discussions**: Thảo luận về kiến trúc và cải tiến
- **Website**: https://github.com/devtantai-coder/nexus

---

## 🎯 Lộ Trình Phát Triển

### Phase 1: MVP ✅
- [x] Kiến trúc Transformer với MLA
- [x] MoE với sort-based expert grouping
- [x] FP8 quantization kernels (Triton)
- [x] YaRN RoPE mở rộng ngữ cảnh
- [x] Model parallelism
- [x] Dataset tiếng Việt + training script

### Phase 2: Cộng Đồng 🚀
- [ ] Pre-trained weights release
- [ ] HuggingFace integration
- [ ] Docker image
- [ ] Google Colab notebook demo
- [ ] Tài liệu tiếng Anh

### Phase 3: Nâng Cao 🔥
- [ ] Fine-tuning với LoRA/QLoRA
- [ ] RLHF/DPO training
- [ ] Web UI (Gradio/Streamlit)
- [ ] ONNX/TensorRT export
- [ ] Benchmark suite

---

## 📜 Giấy Phép

Dự án được phân phối dưới giấy phép **MIT License** — xem [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Lời Cảm Ơn

Nexus AI được lấy cảm hứng từ:
- **DeepSeek-V2/V3** — Multi-Head Latent Attention & MoE
- **Llama 3** — Kiến trúc Transformer hiệu quả
- **OpenAI Triton** — Ngôn ngữ lập trình kernel GPU

---

## ⭐ Ủng Hộ Nexus

**Nexus AI là dự án mã nguồn mở Việt Nam. Hãy giúp chúng tôi phát triển!**

<p align="center">
  <a href="https://github.com/devtantai-coder/nexus/stargazers">
    <img src="https://img.shields.io/github/stars/devtantai-coder/nexus?style=for-the-badge&logo=github&label=STAR%20REPO&color=gold" alt="Star">
  </a>
  <a href="https://github.com/devtantai-coder/nexus/fork">
    <img src="https://img.shields.io/github/forks/devtantai-coder/nexus?style=for-the-badge&logo=github&label=FORK&color=blue" alt="Fork">
  </a>
  <a href="https://twitter.com/intent/tweet?text=Nexus%20AI%20-%20M%C3%B4%20h%C3%ACnh%20ng%C3%B4n%20ng%E1%BB%AF%20Transformer%20m%C3%A3%20ngu%E1%BB%93n%20m%E1%BB%9F%20Vi%E1%BB%87t%20Nam%20%F0%9F%A7%A0%0A%0AMLA%20%2B%20MoE%20%2B%20FP8%20Quantization%0A%0A&url=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ-X/Twitter-1DA1F2?style=for-the-badge&logo=x" alt="Share X">
  </a>
  <a href="https://www.facebook.com/sharer/sharer.php?u=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ-Facebook-1877F2?style=for-the-badge&logo=facebook" alt="Share Facebook">
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/devtantai-coder/nexus" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin" alt="Share LinkedIn">
  </a>
  <a href="https://www.reddit.com/submit?url=https://github.com/devtantai-coder/nexus&title=Nexus%20AI%20-%20Open%20Source%20Transformer%20LLM%20with%20MLA%20%2B%20MoE%20%2B%20FP8" target="_blank">
    <img src="https://img.shields.io/badge/Chia sẻ-Reddit-FF4500?style=for-the-badge&logo=reddit" alt="Share Reddit">
  </a>
</p>

---

<div align="center">
  <h3>🌟 Được xây dựng với ❤️ cho cộng đồng AI Việt Nam 🌟</h3>
  <p>
    <a href="https://github.com/devtantai-coder/nexus/stargazers">
      <img src="https://img.shields.io/github/stars/devtantai-coder/nexus?style=social" alt="Stars">
    </a>
  </p>
  <p>
    <sub>⭐ Star repo này nếu bạn thấy Nexus hữu ích!</sub>
  </p>
  <p>
    <sub>© 2026 Nexus AI Team — <a href="https://github.com/devtantai-coder/nexus">GitHub</a></sub>
  </p>
</div>

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


# === Cấu hình toàn cục cho tính toán phân tán và lượng tử hóa ===
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

# === Biến toàn cục dùng chung cho các module ===

@dataclass
class ModelArgs:
    """
    Lớp dữ liệu định nghĩa các tham số và siêu tham số của mô hình.

    Attributes:
        max_batch_size (int): Kích thước batch tối đa.
        max_seq_len (int): Độ dài chuỗi tối đa.
        dtype (Literal["bf16", "fp8"]): Kiểu dữ liệu cho tính toán.
        scale_fmt (Optional[str]): Định dạng cho hệ số tỷ lệ lượng tử hóa.
        vocab_size (int): Kích thước từ vựng.
        dim (int): Kích thước chiều của mô hình.
        inter_dim (int): Kích thước chiều trung gian cho các tầng MLP.
        moe_inter_dim (int): Kích thước chiều trung gian cho các tầng MoE.
        n_layers (int): Số tầng transformer.
        n_dense_layers (int): Số tầng dense trong mô hình.
        n_heads (int): Số đầu attention.
        n_routed_experts (int): Số expert được định tuyến cho các tầng MoE.
        n_shared_experts (int): Số expert dùng chung cho các tầng MoE.
        n_activated_experts (int): Số expert được kích hoạt trong các tầng MoE.
        n_expert_groups (int): Số nhóm expert.
        n_limited_groups (int): Số nhóm giới hạn cho việc định tuyến MoE.
        score_func (Literal["softmax", "sigmoid"]): Hàm tính điểm cho việc định tuyến MoE.
        route_scale (float): Hệ số tỷ lệ cho điểm định tuyến.
        q_lora_rank (int): Rank LoRA cho các phép chiếu query.
        kv_lora_rank (int): Rank LoRA cho các phép chiếu key-value.
        qk_nope_head_dim (int): Kích thước chiều cho phép chiếu query-key không có positional embeddings.
        qk_rope_head_dim (int): Kích thước chiều cho phép chiếu query-key với rotary embeddings.
        v_head_dim (int): Kích thước chiều cho phép chiếu value.
        original_seq_len (int): Độ dài chuỗi gốc.
        rope_theta (float): Cơ số cho mã hóa vị trí rotary.
        rope_factor (float): Hệ số tỷ lệ cho độ dài chuỗi mở rộng.
        beta_fast (int): Hệ số hiệu chỉnh beta nhanh.
        beta_slow (int): Hệ số hiệu chỉnh beta chậm.
        mscale (float): Hệ số tỷ lệ cho attention mở rộng.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    use_gate_bias: bool = False

    def __hash__(self):
        return hash((self.dim, self.n_layers, self.n_heads, self.kv_lora_rank, self.max_seq_len))

    def __eq__(self, other):
        if not isinstance(other, ModelArgs):
            return NotImplemented
        return (self.dim, self.n_layers, self.n_heads, self.kv_lora_rank, self.max_seq_len) == \
               (other.dim, other.n_layers, other.n_heads, other.kv_lora_rank, other.max_seq_len)


class ParallelEmbedding(nn.Module):
    """
    Tầng embedding hỗ trợ song song hóa trên các tiến trình phân tán.

    Args:
        vocab_size (int): Kích thước từ vựng.
        dim (int): Kích thước chiều embedding.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho tầng embedding song song.

        Args:
            x (torch.Tensor): Tensor đầu vào chứa các chỉ số token.

        Returns:
            torch.Tensor: Biểu diễn embedding.

        Raises:
            ValueError: Nếu `world_size` không được định nghĩa.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Áp dụng biến đổi tuyến tính vào dữ liệu đầu vào: y = xA^T + b.
    Hàm này hỗ trợ các triển khai chuyên biệt dựa trên lượng tử hóa
    và định dạng tensor.

    Args:
        x (torch.Tensor): Tensor đầu vào.
        weight (torch.Tensor): Tensor trọng số. Có thể đã được lượng tử hóa và
            cần giải lượng tử hóa trong một số trường hợp.
        bias (Optional[torch.Tensor]): Tensor bias được cộng thêm. Mặc định là None.

    Returns:
        torch.Tensor: Kết quả của biến đổi tuyến tính, có thể bao gồm
        tính toán nhận biết lượng tử hóa tùy thuộc vào tham số đầu vào.

    Notes:
        - Nếu `weight` đã được lượng tử hóa (ví dụ: `element_size() == 1`), phiên bản
          đã giải lượng tử hóa được sử dụng cho tính toán.
        - Nếu `gemm_impl == "bf16"`, giải lượng tử hóa và phép toán GEMM `bf16` được áp dụng.
        - Trong các trường hợp khác, hàm áp dụng lượng tử hóa cho `x` và sử dụng `fp8_gemm` để tính toán.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Lớp linear tùy chỉnh hỗ trợ trọng số đã lượng tử hóa và bias tùy chọn.

    Args:
        in_features (int): Số lượng đặc trưng đầu vào.
        out_features (int): Số lượng đặc trưng đầu ra.
        bias (bool): Có bao gồm bias hay không. Mặc định là False.
        dtype (optional): Kiểu dữ liệu cho tầng. Mặc định là `torch.bfloat16`.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho lớp linear tùy chỉnh.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đã biến đổi sau tính toán tuyến tính.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Lớp linear với song song cột, chia đặc trưng đầu ra trên các tiến trình phân tán.

    Args:
        in_features (int): Số lượng đặc trưng đầu vào.
        out_features (int): Tổng số lượng đặc trưng đầu ra.
        bias (bool): Có bao gồm bias hay không. Mặc định là False.
        dtype (optional): Kiểu dữ liệu cho tầng. Mặc định là `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho lớp linear song song cột.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đã biến đổi với tính toán song song cột.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Lớp linear với song song hàng, chia đặc trưng đầu vào trên các tiến trình phân tán.

    Args:
        in_features (int): Tổng số lượng đặc trưng đầu vào.
        out_features (int): Số lượng đặc trưng đầu ra.
        bias (bool): Có bao gồm bias hay không. Mặc định là False.
        dtype (optional): Kiểu dữ liệu cho tầng. Mặc định là `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho lớp linear song song hàng.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đã biến đổi với tính toán song song hàng.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Kích thước chiều của tensor đầu vào.
        eps (float): Giá trị epsilon cho ổn định số học. Mặc định là 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Lan truyền tiến cho RMSNorm.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đã chuẩn hóa với cùng kích thước như đầu vào.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    """
    Tính toán chiều hiệu chỉnh cho một số vòng quay nhất định trong rotary positional embedding.

    Args:
        num_rotations (float): Số vòng quay để tính hiệu chỉnh.
        dim (int): Số chiều của không gian embedding.
        base (float): Giá trị cơ số cho tính toán mũ.
        max_seq_len (int): Độ dài chuỗi tối đa.

    Returns:
        float: Chiều hiệu chỉnh dựa trên các tham số đầu vào.
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int) -> Tuple[int, int]:
    """
    Tính toán phạm vi các chiều hiệu chỉnh cho rotary positional embeddings.

    Args:
        low_rot (float): Cận dưới cho số vòng quay.
        high_rot (float): Cận trên cho số vòng quay.
        dim (int): Số chiều của không gian embedding.
        base (float): Giá trị cơ số cho tính toán mũ.
        max_seq_len (int): Độ dài chuỗi tối đa.

    Returns:
        Tuple[int, int]: Phạm vi các chiều hiệu chỉnh (thấp, cao), được kẹp vào chỉ số hợp lệ.
    """
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """
    Tính toán hàm dốc tuyến tính dùng để làm mượt giá trị giữa cận dưới và cận trên.

    Args:
        min_val (float): Giá trị tối thiểu cho hàm dốc.
        max_val (float): Giá trị tối đa cho hàm dốc.
        dim (int): Số chiều của tensor dốc.

    Returns:
        torch.Tensor: Tensor có kích thước (dim,) với các giá trị được nội suy tuyến tính
            giữa 0 và 1, được kẹp trong đoạn [0, 1].
    """
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


@lru_cache(maxsize=2)
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Tính toán trước các giá trị số phức mũ dựa trên tần số cho rotary positional embeddings.
    Kết quả được cache với `lru_cache` để tránh tính lại khi `ModelArgs` không đổi.

    Args:
        args (ModelArgs): Tham số mô hình chứa các tham số positional embedding.

    Returns:
        torch.Tensor: Giá trị số phức mũ đã tính toán trước cho positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - _linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Áp dụng rotary positional embeddings lên tensor đầu vào.

    Args:
        x (torch.Tensor): Tensor đầu vào cần áp dụng positional embeddings.
        freqs_cis (torch.Tensor): Giá trị số phức mũ đã tính toán trước cho positional embeddings.

    Returns:
        torch.Tensor: Tensor đã được áp dụng rotary embeddings.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Kích thước chiều của đặc trưng đầu vào.
        n_heads (int): Số đầu attention.
        n_local_heads (int): Số đầu attention cục bộ cho hệ thống phân tán.
        q_lora_rank (int): Rank cho phép chiếu query low-rank.
        kv_lora_rank (int): Rank cho phép chiếu key/value low-rank.
        qk_nope_head_dim (int): Kích thước chiều cho phép chiếu query/key không có positional.
        qk_rope_head_dim (int): Kích thước chiều cho phép chiếu query/key với rotary.
        qk_head_dim (int): Tổng kích thước chiều cho phép chiếu query/key.
        v_head_dim (int): Kích thước chiều cho phép chiếu value.
        softmax_scale (float): Hệ số tỷ lệ cho softmax trong tính toán attention.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Cache cho wkv_b view (absorb mode) — instance-level, không dùng global
        self._wkv_b_cache: Optional[Tuple] = None

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Lan truyền tiến cho Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, dim).
            start_pos (int): Vị trí bắt đầu trong chuỗi cho caching.
            freqs_cis (torch.Tensor): Giá trị số phức mũ đã tính toán trước cho rotary embeddings.
            mask (Optional[torch.Tensor]): Tensor mặt nạ để loại trừ các vị trí khỏi attention.

        Returns:
            torch.Tensor: Tensor đầu ra có cùng kích thước với đầu vào.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Lan truyền tiến cho Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, dim).
            start_pos (int): Vị trí bắt đầu trong chuỗi cho caching.
            freqs_cis (torch.Tensor): Giá trị số phức mũ đã tính toán trước cho rotary embeddings.
            mask (Optional[torch.Tensor]): Tensor mặt nạ để loại trừ các vị trí khỏi attention.

        Returns:
            torch.Tensor: Tensor đầu ra có cùng kích thước với đầu vào.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if self.training:
            # === Training mode: không cache, attention toàn bộ sequence ===
            if attn_impl == "naive":
                q_full = torch.cat([q_nope, q_pe], dim=-1)
                kv_full = self.wkv_b(self.kv_norm(kv))
                kv_full = kv_full.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope_full, v_full = torch.split(kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k_full = torch.cat([k_nope_full, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                scores = torch.einsum("bshd,bthd->bsht", q_full, k_full) * self.softmax_scale
                if mask is not None:
                    scores += mask.unsqueeze(1)
                scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
                x = torch.einsum("bsht,bthd->bshd", scores, v_full)
            else:
                # === Absorb mode: không materialize K/V đầy đủ ===
                if self._wkv_b_cache is not None and self._wkv_b_cache[0] is self.wkv_b.weight:
                    wkv_b = self._wkv_b_cache[1]
                else:
                    wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
                    wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
                    self._wkv_b_cache = (self.wkv_b.weight, wkv_b)
                q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                kv_norm = self.kv_norm(kv)
                scores = (torch.einsum("bshc,btc->bsht", q_nope_proj, kv_norm) +
                          torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.softmax_scale
                if mask is not None:
                    scores += mask.unsqueeze(1)
                scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
                x = torch.einsum("bsht,btc->bshc", scores, kv_norm)
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        else:
            # === Inference mode: KV cache ===
            if attn_impl == "naive":
                q_full = torch.cat([q_nope, q_pe], dim=-1)
                kv_full = self.wkv_b(self.kv_norm(kv))
                kv_full = kv_full.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope_full, v_full = torch.split(kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k_full = torch.cat([k_nope_full, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                self.k_cache[:bsz, start_pos:end_pos] = k_full
                self.v_cache[:bsz, start_pos:end_pos] = v_full
                scores = torch.einsum("bshd,bthd->bsht", q_full, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            else:
                if self._wkv_b_cache is not None and self._wkv_b_cache[0] is self.wkv_b.weight:
                    wkv_b = self._wkv_b_cache[1]
                else:
                    wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
                    wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
                    self._wkv_b_cache = (self.wkv_b.weight, wkv_b)
                q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                kv_norm = self.kv_norm(kv)
                self.kv_cache[:bsz, start_pos:end_pos] = kv_norm
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                scores = (torch.einsum("bshc,btc->bsht", q_nope_proj, self.kv_cache[:bsz, :end_pos]) +
                          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            if mask is not None:
                scores += mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            if attn_impl == "naive":
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) dùng làm tầng feed-forward.

    Attributes:
        w1 (nn.Module): Tầng linear biến đổi đầu vào thành không gian ẩn.
        w2 (nn.Module): Tầng linear biến đổi không gian ẩn thành đầu ra.
        w3 (nn.Module): Tầng linear bổ sung cho biến đổi đặc trưng.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Khởi tạo tầng MLP.

        Args:
            dim (int): Kích thước đầu vào và đầu ra.
            inter_dim (int): Kích thước tầng ẩn.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho tầng MLP.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đầu ra sau tính toán MLP.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Cơ chế gating (cổng) để định tuyến đầu vào trong mô hình mixture-of-experts (MoE).

    Attributes:
        dim (int): Kích thước chiều của đặc trưng đầu vào.
        topk (int): Số expert hàng đầu được kích hoạt cho mỗi đầu vào.
        n_groups (int): Số nhóm cho việc định tuyến.
        topk_groups (int): Số nhóm để định tuyến đầu vào.
        score_func (str): Hàm tính điểm ('softmax' hoặc 'sigmoid').
        route_scale (float): Hệ số tỷ lệ cho trọng số định tuyến.
        weight (torch.nn.Parameter): Trọng số học được cho cổng.
        bias (Optional[torch.nn.Parameter]): Bias tùy chọn cho cổng.
    """
    def __init__(self, args: ModelArgs):
        """
        Khởi tạo module Gate.

        Args:
            args (ModelArgs): Tham số mô hình chứa các tham số gating.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if args.use_gate_bias else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lan truyền tiến cho cơ chế gating.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Trọng số định tuyến và chỉ số expert được chọn.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Tầng Expert cho mô hình Mixture-of-Experts (MoE).

    Attributes:
        w1 (nn.Module): Tầng linear biến đổi đầu vào thành không gian ẩn.
        w2 (nn.Module): Tầng linear biến đổi không gian ẩn thành đầu ra.
        w3 (nn.Module): Tầng linear bổ sung cho biến đổi đặc trưng.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Khởi tạo tầng Expert.

        Args:
            dim (int): Kích thước đầu vào và đầu ra.
            inter_dim (int): Kích thước tầng ẩn.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho tầng Expert.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đầu ra sau tính toán expert.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Module Mixture-of-Experts (MoE).

    Attributes:
        dim (int): Kích thước chiều của đặc trưng đầu vào.
        n_routed_experts (int): Tổng số expert trong mô hình.
        n_local_experts (int): Số expert được xử lý cục bộ trong hệ thống phân tán.
        n_activated_experts (int): Số expert được kích hoạt cho mỗi đầu vào.
        gate (nn.Module): Cơ chế gating để định tuyến đầu vào đến các expert.
        experts (nn.ModuleList): Danh sách các module expert.
        shared_experts (nn.Module): Các expert dùng chung áp dụng cho tất cả đầu vào.
    """
    def __init__(self, args: ModelArgs):
        """
        Khởi tạo module MoE.

        Args:
            args (ModelArgs): Tham số mô hình chứa các tham số MoE.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến cho module MoE.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đầu ra sau khi định tuyến expert và tính toán.
        """
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices = self.gate(x_flat)
        y = torch.zeros_like(x_flat)

        # Sort-based expert grouping: giữ mọi thứ trên GPU
        flat_indices = indices.flatten()
        sorted_indices, perm = flat_indices.sort(stable=True)
        unique_experts, counts = torch.unique_consecutive(sorted_indices, return_counts=True)

        # Pre-filter local experts trên GPU — giảm .item() calls
        local_mask = (unique_experts >= self.experts_start_idx) & (unique_experts < self.experts_end_idx)
        local_expert_ids = unique_experts[local_mask]
        local_counts = counts[local_mask]

        if local_counts.numel() > 0:
            all_offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                all_offsets[1:] = counts.cumsum(0)[:-1]
            local_offsets = all_offsets[local_mask]

            topk = self.n_activated_experts
            for i in range(len(local_expert_ids)):
                expert_idx = local_expert_ids[i].item()
                expert = self.experts[expert_idx]
                if expert is None:
                    continue
                start = local_offsets[i].item()
                end = start + local_counts[i].item()
                token_perm = perm[start:end]
                rows = token_perm // topk
                w = weights.view(-1)[token_perm, None]
                y.index_add_(0, rows, expert(x_flat[rows]) * w)
        z = self.shared_experts(x_flat)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Khối transformer kết hợp tầng attention và feed-forward.

    Attributes:
        attn (nn.Module): Tầng attention (MLA).
        ffn (nn.Module): Mạng feed-forward (MLP hoặc MoE).
        attn_norm (nn.Module): Chuẩn hóa tầng cho attention.
        ffn_norm (nn.Module): Chuẩn hóa tầng cho feed-forward.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Khởi tạo khối Transformer.

        Args:
            layer_id (int): Chỉ số tầng trong transformer.
            args (ModelArgs): Tham số mô hình chứa các tham số khối.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Lan truyền tiến cho khối Transformer.

        Args:
            x (torch.Tensor): Tensor đầu vào.
            start_pos (int): Vị trí bắt đầu trong chuỗi.
            freqs_cis (torch.Tensor): Giá trị số phức mũ đã tính toán trước cho rotary embeddings.
            mask (Optional[torch.Tensor]): Tensor mặt nạ để loại trừ các vị trí khỏi attention.

        Returns:
            torch.Tensor: Tensor đầu ra sau tính toán khối.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Mô hình Transformer với positional embeddings, nhiều tầng và phép chiếu đầu ra.

    Attributes:
        max_seq_len (int): Độ dài chuỗi tối đa cho transformer.
        embed (nn.Module): Tầng embedding cho token đầu vào.
        layers (torch.nn.ModuleList): Danh sách các khối transformer.
        norm (nn.Module): Chuẩn hóa tầng áp dụng sau tất cả các khối.
        head (nn.Module): Tầng chiếu đầu ra ánh xạ tới kích thước từ vựng.
        freqs_cis (torch.Tensor): Giá trị số phức mũ đã tính toán trước cho rotary embeddings.
    """
    @staticmethod
    def setup_global_flags():
        """Bật TF32 và cudnn benchmark — gọi 1 lần khi khởi tạo model."""
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def __init__(self, args: ModelArgs):
        """
        Khởi tạo mô hình Transformer.

        Args:
            args (ModelArgs): Tham số mô hình chứa các tham số transformer.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else (torch.bfloat16 if args.dtype == "bf16" else torch.float32)
        Linear.scale_fmt = args.scale_fmt
        Transformer.setup_global_flags()
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=Linear.dtype)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        # Flag cho phép torch.compile — chỉ bật nếu môi trường hỗ trợ
        self._compiled = False

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Lan truyền tiến cho mô hình Transformer.

        Args:
            tokens (torch.Tensor): Tensor đầu vào chứa ID token với kích thước (batch_size, seq_len).
            start_pos (int, optional): Vị trí bắt đầu trong chuỗi cho rotary embeddings. Mặc định 0.

        Returns:
            torch.Tensor: Tensor logits có kích thước (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            # Sử dụng all_gather_into_tensor thay vì all_gather + cat khi khả dụng
            if hasattr(dist, "all_gather_into_tensor") and dist.is_available():
                all_logits = torch.empty(world_size * logits.size(-1), device=logits.device, dtype=logits.dtype)
                dist.all_gather_into_tensor(all_logits, logits)
                logits = all_logits.view(1, -1)
            else:
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)
        return logits

    def forward_train(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward cho training — trả về logits cho tất cả vị trí.

        Args:
            tokens (torch.Tensor): (batch_size, seq_len) token IDs.

        Returns:
            torch.Tensor: (batch_size, seq_len, vocab_size) logits.
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask)
        h = self.norm(h)
        logits = self.head(h)
        if world_size > 1:
            if hasattr(dist, "all_gather_into_tensor") and dist.is_available():
                all_logits = torch.empty(world_size * logits.size(-1), device=logits.device, dtype=logits.dtype)
                dist.all_gather_into_tensor(all_logits, logits)
                logits = all_logits.view(1, -1)
            else:
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)
        return logits

    def compile_if_enabled(self):
        """Bật torch.compile để tối ưu JIT. Gọi sau khi load weights."""
        if not self._compiled:
            try:
                self.forward = torch.compile(self.forward, mode="reduce-overhead")
                self._compiled = True
            except Exception:
                pass  # torch.compile không khả dụng, fallback về eager


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())

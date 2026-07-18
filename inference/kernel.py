from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    Lượng tử hóa tensor đầu vào `x_ptr` và lưu kết quả vào `y_ptr` cùng hệ số tỷ lệ vào `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Con trỏ đến tensor đầu vào.
        y_ptr (triton.Pointer): Con trỏ đến tensor đầu ra nơi lưu giá trị đã lượng tử hóa.
        s_ptr (triton.Pointer): Con trỏ đến tensor đầu ra nơi lưu hệ số tỷ lệ.
        BLOCK_SIZE (tl.constexpr): Kích thước block được xử lý bởi mỗi instance chương trình.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))  # reduction
    amax = tl.maximum(amax, 1e-4)  # clamp để tránh chia cho 0
    if scale_fmt == "ue8m0":
        # Làm tròn lên lũy thừa của 2 cho định dạng ue8m0
        exp = tl.math.ceil(tl.math.log2(amax / 448.))
        s = tl.math.exp2(exp)
    else:
        s = amax / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lượng tử hóa tensor đầu vào `x` sử dụng lượng tử hóa theo block.

    Args:
        x (torch.Tensor): Tensor đầu vào cần lượng tử hóa. Phải liên tục (contiguous) và kích thước chiều cuối phải chia hết cho `block_size`.
        block_size (int, optional): Kích thước block dùng cho lượng tử hóa. Mặc định 128.
        scale_fmt (Optional[str], optional): Định dạng của hệ số tỷ lệ. Mặc định None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Một tuple chứa:
            - Tensor đã lượng tử hóa với dtype `torch.float8_e4m3fn`.
            - Tensor hệ số tỷ lệ với dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, scale_fmt=scale_fmt)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Giải lượng tử hóa trọng số sử dụng hệ số tỷ lệ đã cho và lưu kết quả.

    Args:
        x_ptr (tl.pointer): Con trỏ đến trọng số đã lượng tử hóa.
        s_ptr (tl.pointer): Con trỏ đến hệ số tỷ lệ.
        y_ptr (tl.pointer): Con trỏ đến bộ đệm đầu ra cho trọng số đã giải lượng tử hóa.
        M (int): Số hàng trong ma trận trọng số.
        N (int): Số cột trong ma trận trọng số.
        BLOCK_SIZE (tl.constexpr): Kích thước block cho việc chia nhỏ (tiling).

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Giải lượng tử hóa tensor trọng số đã cho sử dụng tensor tỷ lệ đã cho.

    Args:
        x (torch.Tensor): Tensor trọng số đã lượng tử hóa với kích thước (M, N).
        s (torch.Tensor): Tensor tỷ lệ với kích thước (M//block_size, N//block_size).
        block_size (int, optional): Kích thước block dùng cho giải lượng tử hóa. Mặc định 128.

    Returns:
        torch.Tensor: Tensor trọng số đã giải lượng tử hóa với cùng kích thước như `x`.

    Raises:
        AssertionError: Nếu `x` hoặc `s` không liên tục (contiguous) hoặc nếu số chiều của chúng không phải 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Các cấu hình autotune cho kernel FP8 GEMM
# BLOCK_SIZE_M và BLOCK_SIZE_N được tối ưu hóa cho các kích thước ma trận khác nhau
# Lưu ý: BLOCK_SIZE_K phải bằng 128 (khớp với quantization block size) để scale pointer arithmetic đúng
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64, 128] for block_n in [32, 64, 128, 256] for num_stages in [3, 4, 5, 6]
]

# Cấu hình GEMM nhanh cho ma trận nhỏ (decode stage) — 4 warps, ít stages
# BLOCK_SIZE_K=128 để khớp với quantization block size cho scale pointer arithmetic
fp8_gemm_small_configs = [
    Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=4),
    Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=4),
    Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Thực hiện phép nhân ma trận trên các ma trận FP8 với hệ số tỷ lệ.

    Args:
        a_ptr (tl.tensor): Con trỏ đến ma trận đầu vào A.
        b_ptr (tl.tensor): Con trỏ đến ma trận đầu vào B.
        c_ptr (tl.tensor): Con trỏ đến ma trận đầu ra C.
        a_s_ptr (tl.tensor): Con trỏ đến hệ số tỷ lệ cho ma trận A.
        b_s_ptr (tl.tensor): Con trỏ đến hệ số tỷ lệ cho ma trận B.
        M (int): Số hàng trong ma trận A và C.
        N (tl.constexpr): Số cột trong ma trận B và C.
        K (tl.constexpr): Số cột trong ma trận A và số hàng trong ma trận B.
        BLOCK_SIZE_M (tl.constexpr): Kích thước block cho chiều M.
        BLOCK_SIZE_N (tl.constexpr): Kích thước block cho chiều N.
        BLOCK_SIZE_K (tl.constexpr): Kích thước block cho chiều K.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k_blocks
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        # Giảm mask overhead: load có mask mặc định 0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Thực hiện phép nhân ma trận sử dụng độ chính xác FP8.

    Args:
        a (torch.Tensor): Ma trận đầu vào thứ nhất, phải liên tục (contiguous).
        a_s (torch.Tensor): Hệ số tỷ lệ cho ma trận đầu vào thứ nhất, phải liên tục.
        b (torch.Tensor): Ma trận đầu vào thứ hai, phải liên tục.
        b_s (torch.Tensor): Hệ số tỷ lệ cho ma trận đầu vào thứ hai, phải liên tục.

    Returns:
        torch.Tensor: Kết quả của phép nhân ma trận.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

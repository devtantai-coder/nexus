"""Train model với dataset mở rộng ~2000 samples"""
import json, random, torch, sys, os, math
sys.path.insert(0, '/home/thinkpad/Downloads/nexus/inference')
from model import Transformer, ModelArgs
import torch.nn.init as _init

random.seed(42)
torch.manual_seed(42)

print("=" * 50)
print("XÂY DỰNG DATASET MỞ RỘNG")
print("=" * 50)

words = [
    "tôi", "bạn", "chúng", "ta", "mình", "cậu", "anh", "chị", "em", "các",
    "là", "và", "của", "với", "cho", "trong", "trên", "dưới", "tại", "ở",
    "có", "không", "sẽ", "đã", "đang", "rất", "thật", "quá", "lắm", "nhé",
    "nào", "gì", "đâu", "sao", "thế", "vậy", "nhỉ", "ạ", "nha", "à",
    "thì", "mà", "cũng", "nếu", "nhưng", "vì", "nên", "để", "hay", "hoặc",
    "vào", "ra", "lên", "xuống", "qua", "lại", "cùng", "vừa", "mới", "được",
    "làm", "học", "chơi", "đi", "chạy", "ăn", "uống", "ngủ", "đọc", "viết",
    "nói", "nghe", "nhìn", "biết", "hiểu", "nghĩ", "thích", "yêu", "ghét", "muốn",
    "cần", "phải", "giúp", "cho", "tặng", "mua", "bán", "gọi", "hỏi", "trả",
    "lời", "tập", "luyện", "dạy", "bảo", "kể", "xem", "thấy",
    "người", "thầy", "trò", "trường", "lớp", "bài", "bút",
    "máy", "tính", "điện", "thoại", "mạng", "web", "phần", "mềm",
    "thời", "gian", "tiền", "công", "việc", "cuộc", "sống", "thế",
    "giới", "tương", "lai", "quá", "khứ", "nhà", "cửa", "xe", "đường", "phố",
    "tốt", "xấu", "đẹp", "xinh", "vui", "buồn", "khỏe", "yếu", "nhanh", "chậm",
    "giỏi", "dốt", "chăm", "chỉ", "lười", "dễ", "khó", "mới",
    "cũ", "lớn", "nhỏ", "xa", "gần", "nhiều", "ít", "đủ", "thiếu", "giàu",
    "lập", "trình", "thuật", "toán", "trí", "tuệ", "nhân", "tạo", "học", "máy",
    "sâu", "mạng", "nơ", "ron", "xử", "lý", "ngôn", "ngữ", "thị", "giác",
    "lớn", "python", "javascript", "react", "nodejs", "ứng", "dụng",
    "hệ", "thống", "mô", "hình", "kiến", "thức", "kinh", "nghiệm", "kỹ", "năng",
    "ồ", "à", "ơi", "này", "ừ", "ừm", "nhỉ", "nhé", "nha", "vâng", "dạ",
    "khoan", "thôi", "chà", "ôi", "trời", "chết", "đùa", "thật", "tuyệt", "ngon",
    "xin", "chào", "cảm", "ơn", "tạm", "biệt", "hẹn", "gặp", "lại", "chúc",
    "khỏe", "vui", "buồn", "mệt", "đói", "khát", "ngủ", "dậy", "sớm", "muộn",
]

vocab = {w: i+1 for i, w in enumerate(sorted(set(words)))}
vocab_size = len(vocab) + 1
print(f'Vocab: {len(vocab)} words, vocab_size={vocab_size}')

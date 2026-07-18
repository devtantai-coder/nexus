# Nexus AI 🧠

Open-source Vietnamese Transformer LLM with Multi-Head Latent Attention, MoE 64 experts, and FP8 Quantization.

## Features
- 🧠 Multi-Head Latent Attention (MLA) — KV latent compression
- 🧩 Mixture-of-Experts — 64 routed experts, top-6 routing
- ⚡ FP8 Quantization — Custom Triton kernels with autotune
- 🔗 YaRN RoPE — 128K+ context window
- 🖥️ Model Parallelism — Column/Row parallel + expert distribution
- 🇻🇳 Vietnamese dataset — ~2000+ samples

## Quick Start
```bash
git clone https://github.com/devtantai-coder/nexus.git
pip install -r requirements.txt
python inference/chat.py --interactive
```

## License
MIT

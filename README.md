# Drisyon Sadhana · LLM Internals & Inference Engine (LLM-INT40)

[![Sadhana CI Gate](https://github.com/tnmurthy/drisyon-sadhana-llmint40/actions/workflows/gate.yml/badge.svg)](https://github.com/tnmurthy/drisyon-sadhana-llmint40/actions)

Welcome to the official **LLM Internals: Architecture, Math & GPU Inference (40H)** starter lab repository.

---

## 🚀 Quickstart

```bash
# 1. Clone repository
git clone https://github.com/tnmurthy/drisyon-sadhana-llmint40.git
cd drisyon-sadhana-llmint40

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Gate 1 Verification Suite
pytest tests/test_gate1.py -v
```

---

## 📂 Repository Structure

- `core/bpe.py`: Byte Pair Encoding (BPE) tokenizer with vocabulary merge learning.
- `core/attention.py`: Scaled dot-product attention, causal masking, RoPE, and Grouped-Query Attention (GQA).
- `core/memory.py`: KV Cache VRAM footprint and MHA vs GQA savings calculators.
- `core/inference.py`: PagedAttention virtual block manager and Speculative Decoding verification.
- `tests/test_gate1.py`: Automated Pytest gate assertions.

---

## 🏆 100-Point Evaluation Standards

- **Gate 1 CI (25 Pts):** All automated tests pass without patch overrides.
- **Gate 2 Mid-Term (25 Pts):** RoPE 2D rotation invariance + KV Cache state persistence.
- **Gate 3 Capstone Defense (50 Pts):** Live jury audit of custom inference serving engine under stress.

"""
Memory & VRAM calculations for LLM inference, KV-cache, and attention architectures.
"""


def calculate_kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Computes exact byte size of the Key-Value Cache.
    Formula: 2 (K and V) * dtype_bytes * n_layers * n_kv_heads * d_head * seq_len * batch_size
    """
    return 2 * dtype_bytes * n_layers * n_kv_heads * d_head * seq_len * batch_size


def calculate_model_weights_bytes(n_params: int, precision: str = "fp16") -> int:
    """
    Computes VRAM required to load model weights.
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }.get(precision.lower(), 2)
    return int(n_params * bytes_per_param)


def compare_mha_vs_gqa_memory(
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    Compares Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA) KV Cache requirements.
    """
    mha_bytes = calculate_kv_cache_bytes(n_layers, n_heads, d_head, seq_len, batch_size, dtype_bytes)
    gqa_bytes = calculate_kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch_size, dtype_bytes)
    
    savings_bytes = mha_bytes - gqa_bytes
    savings_ratio = mha_bytes / max(gqa_bytes, 1)
    
    return {
        "mha_bytes": mha_bytes,
        "gqa_bytes": gqa_bytes,
        "savings_bytes": savings_bytes,
        "savings_ratio": round(savings_ratio, 2),
        "mha_mb": round(mha_bytes / (1024 * 1024), 2),
        "gqa_mb": round(gqa_bytes / (1024 * 1024), 2),
    }

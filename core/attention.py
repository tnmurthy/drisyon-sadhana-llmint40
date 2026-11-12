"""
Attention Mechanics: Scaled Dot-Product, RoPE & Grouped-Query Attention (GQA).
"""

import math
import numpy as np


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: str = None) -> np.ndarray:
    """
    Scaled dot-product attention: softmax(Q @ K.T / sqrt(d_k) + M) @ V
    Q: [..., seq_len_q, d_k]
    K: [..., seq_len_k, d_k]
    V: [..., seq_len_v, d_v]
    """
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    
    # Compute dot product
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    
    # Apply causal mask if requested
    if mask == "causal":
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]
        causal_mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=bool), k=1)
        scores = np.where(causal_mask, -1e9, scores)
    elif mask is not None:
        scores = np.where(mask, -1e9, scores)
        
    # Softmax along last dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted sum with Values
    output = np.matmul(attn_weights, V)
    return output, attn_weights


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute cosine and sine frequencies for Rotary Position Embeddings (RoPE).
    dim must be even.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rotary_pos_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply RoPE rotation to 2D coordinate pairs.
    x: [..., seq_len, dim]
    """
    # Split into pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Rotate
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    # Interleave back
    out = np.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA) where multiple Query heads share single Key/Value heads.
    Example: 8 Query heads, 2 KV heads (group size 4).
    """
    def __init__(self, n_heads: int, n_kv_heads: int, d_head: int):
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.d_head = d_head
        self.kv_cache = {"k": None, "v": None}

    def reset_cache(self):
        self.kv_cache = {"k": None, "v": None}

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, use_cache: bool = True):
        """
        q: [batch, n_heads, seq_len, d_head]
        k: [batch, n_kv_heads, seq_len, d_head]
        v: [batch, n_kv_heads, seq_len, d_head]
        """
        batch_size, _, seq_len, _ = q.shape

        if use_cache:
            if self.kv_cache["k"] is None:
                self.kv_cache["k"] = k
                self.kv_cache["v"] = v
            else:
                self.kv_cache["k"] = np.concatenate([self.kv_cache["k"], k], axis=2)
                self.kv_cache["v"] = np.concatenate([self.kv_cache["v"], v], axis=2)
            k = self.kv_cache["k"]
            v = self.kv_cache["v"]

        # Repeat KV heads to match Q heads
        k_rep = np.repeat(k, self.group_size, axis=1)
        v_rep = np.repeat(v, self.group_size, axis=1)

        # Compute attention per head
        out, weights = scaled_dot_product_attention(q, k_rep, v_rep, mask="causal" if seq_len > 1 else None)
        return out, weights

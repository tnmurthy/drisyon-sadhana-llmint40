"""
Automated CI Verification Suite: Gate 1 (LLM Internals & Inference Engine)
Asserts correctness of BPE Tokenization, Causal Attention, RoPE, GQA, Memory Calculations,
PagedAttention Block Tables, and Speculative Decoding.
"""

import numpy as np
import pytest

from core.bpe import BPETokenizer
from core.attention import (
    scaled_dot_product_attention,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    GroupedQueryAttention,
)
from core.memory import calculate_kv_cache_bytes, compare_mha_vs_gqa_memory
from core.inference import PagedAttentionSimulator, SpeculativeDecodingVerifier


def test_bpe_training_and_encoding():
    corpus = [
        "transformer attention architecture",
        "attention is all you need",
        "transformer models scale with compute",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, max_merges=10)
    
    assert len(tokenizer.merges) > 0, "BPE should learn merge rules from corpus"
    
    sample_text = "transformer attention"
    encoded = tokenizer.encode(sample_text)
    assert len(encoded) > 0, "Encoded tokens should not be empty"
    
    decoded = tokenizer.decode(encoded)
    assert "transformer" in decoded and "attention" in decoded, f"Roundtrip decode failed: {decoded}"


def test_scaled_dot_product_attention_causal_mask():
    seq_len = 4
    d_k = 8
    np.random.seed(42)
    Q = np.random.randn(1, 1, seq_len, d_k)
    K = np.random.randn(1, 1, seq_len, d_k)
    V = np.random.randn(1, 1, seq_len, d_k)
    
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask="causal")
    
    assert output.shape == (1, 1, seq_len, d_k)
    assert attn_weights.shape == (1, 1, seq_len, seq_len)
    
    # Check causal mask: upper triangle (future tokens) must have 0 attention probability
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_weights[0, 0, i, j] < 1e-6, f"Future position ({i}, {j}) has non-zero attention: {attn_weights[0, 0, i, j]}"


def test_rope_rotary_position_embedding():
    dim = 16
    seq_len = 8
    cos, sin = precompute_freqs_cis(dim=dim, end=seq_len)
    
    np.random.seed(42)
    x = np.random.randn(1, seq_len, dim)
    rotated = apply_rotary_pos_emb(x, cos, sin)
    
    assert rotated.shape == x.shape, "RoPE output shape must match input shape"
    assert not np.allclose(rotated, x), "Rotated vectors must differ from unrotated vectors"


def test_grouped_query_attention_and_kv_cache():
    n_heads = 8
    n_kv_heads = 2
    d_head = 16
    gqa = GroupedQueryAttention(n_heads=n_heads, n_kv_heads=n_kv_heads, d_head=d_head)
    
    # Prefill step (seq_len = 3)
    np.random.seed(42)
    q1 = np.random.randn(1, n_heads, 3, d_head)
    k1 = np.random.randn(1, n_kv_heads, 3, d_head)
    v1 = np.random.randn(1, n_kv_heads, 3, d_head)
    
    out1, weights1 = gqa.forward(q1, k1, v1, use_cache=True)
    assert out1.shape == (1, n_heads, 3, d_head)
    assert gqa.kv_cache["k"].shape == (1, n_kv_heads, 3, d_head)
    
    # Decode step: single token (seq_len = 1)
    q2 = np.random.randn(1, n_heads, 1, d_head)
    k2 = np.random.randn(1, n_kv_heads, 1, d_head)
    v2 = np.random.randn(1, n_kv_heads, 1, d_head)
    
    out2, weights2 = gqa.forward(q2, k2, v2, use_cache=True)
    assert out2.shape == (1, n_heads, 1, d_head)
    assert gqa.kv_cache["k"].shape == (1, n_kv_heads, 4, d_head), "KV cache must append new token"


def test_kv_cache_memory_and_gqa_reduction():
    n_layers = 32
    n_heads = 32
    n_kv_heads = 8  # 4:1 ratio
    d_head = 128
    seq_len = 4096
    batch_size = 8
    
    res = compare_mha_vs_gqa_memory(n_layers, n_heads, n_kv_heads, d_head, seq_len, batch_size)
    
    assert res["savings_ratio"] == 4.0, f"Expected 4x savings ratio, got {res['savings_ratio']}"
    assert res["mha_bytes"] > res["gqa_bytes"]
    assert res["savings_bytes"] > 0


def test_paged_attention_block_manager():
    simulator = PagedAttentionSimulator(num_total_blocks=64, block_size=16)
    
    # Allocate request 1 with 25 tokens (requires 2 blocks: 16 + 9)
    req1_blocks = simulator.allocate("req-001", num_tokens=25)
    assert len(req1_blocks) == 2
    assert len(simulator.free_blocks) == 62
    
    # Append 7 tokens -> fills up second block exactly (25 + 7 = 32)
    for _ in range(7):
        simulator.append_token("req-001")
    assert len(simulator.block_tables["req-001"]) == 2
    
    # Append 1 more token -> crosses boundary, allocates 3rd block
    simulator.append_token("req-001")
    assert len(simulator.block_tables["req-001"]) == 3
    assert len(simulator.free_blocks) == 61
    
    # Free request 1 -> returns all 3 blocks
    simulator.free("req-001")
    assert len(simulator.free_blocks) == 64
    assert "req-001" not in simulator.block_tables


def test_speculative_decoding_verification():
    verifier = SpeculativeDecodingVerifier(gamma=4)
    
    draft_tokens = [101, 204, 305, 408]
    draft_probs = [0.85, 0.80, 0.75, 0.40]
    target_probs = [0.88, 0.82, 0.70, 0.10]
    
    res = verifier.verify_tokens(draft_tokens, draft_probs, target_probs)
    
    assert res["num_proposed"] == 4
    # First 3 should be accepted since accept_prob >= 0.70; token 4 has 0.10 / 0.40 = 0.25 < 0.70
    assert res["num_accepted"] == 3
    assert res["accepted_tokens"] == [101, 204, 305]
    assert res["acceptance_rate_alpha"] == 0.75
    assert res["theoretical_speedup"] >= 1.5, f"Expected speedup >= 1.5x, got {res['theoretical_speedup']}"

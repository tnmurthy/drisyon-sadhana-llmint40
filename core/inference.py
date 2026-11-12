"""
Inference Systems: PagedAttention virtual memory block manager & Speculative Decoding verification.
"""

import math
import numpy as np


class PagedAttentionSimulator:
    """
    Simulates vLLM PagedAttention virtual block memory management.
    Eliminates internal fragmentation by allocating non-contiguous physical blocks of fixed size.
    """
    def __init__(self, num_total_blocks: int = 128, block_size: int = 16):
        self.num_total_blocks = num_total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_total_blocks))
        # Maps request_id -> list of allocated physical block indices
        self.block_tables = {}
        # Maps request_id -> current token length
        self.token_counts = {}

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        """
        Allocate physical blocks needed to store num_tokens.
        """
        blocks_needed = math.ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError(f"Out of blocks: needed {blocks_needed}, free {len(self.free_blocks)}")

        allocated = []
        for _ in range(blocks_needed):
            allocated.append(self.free_blocks.pop(0))

        self.block_tables[request_id] = allocated
        self.token_counts[request_id] = num_tokens
        return allocated

    def append_token(self, request_id: str) -> int:
        """
        Add a generated token to a request. If current block is full, allocate a new block.
        """
        if request_id not in self.block_tables:
            raise KeyError(f"Request {request_id} not found")

        curr_tokens = self.token_counts[request_id]
        if curr_tokens % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("Out of memory blocks during token generation")
            new_block = self.free_blocks.pop(0)
            self.block_tables[request_id].append(new_block)

        self.token_counts[request_id] += 1
        return self.token_counts[request_id]

    def free(self, request_id: str):
        """
        Free all physical blocks belonging to request_id back to the pool.
        """
        if request_id in self.block_tables:
            blocks = self.block_tables.pop(request_id)
            self.free_blocks.extend(blocks)
            self.token_counts.pop(request_id, None)

    def get_fragmentation_stats(self) -> dict:
        """
        Returns stats on physical block utilization.
        """
        total_tokens = sum(self.token_counts.values())
        allocated_blocks = sum(len(b) for b in self.block_tables.values())
        allocated_capacity = allocated_blocks * self.block_size
        waste_tokens = allocated_capacity - total_tokens
        utilization = total_tokens / max(allocated_capacity, 1)

        return {
            "total_tokens": total_tokens,
            "allocated_blocks": allocated_blocks,
            "free_blocks": len(self.free_blocks),
            "internal_waste_tokens": waste_tokens,
            "block_utilization_pct": round(utilization * 100, 2),
        }


class SpeculativeDecodingVerifier:
    """
    Rejection sampling verification for Speculative Decoding.
    A small Draft model proposes gamma candidate tokens; the large Target model evaluates them in parallel.
    """
    def __init__(self, gamma: int = 4):
        self.gamma = gamma

    def verify_tokens(
        self,
        draft_tokens: list[int],
        draft_probs: list[float],
        target_probs: list[float],
    ) -> dict:
        """
        Verifies draft tokens against target distribution via rejection sampling:
        Acceptance probability = min(1.0, P_target(x) / P_draft(x))
        """
        assert len(draft_tokens) == len(draft_probs) == len(target_probs), "Inputs must have equal length"
        accepted = []
        rejected_index = None

        for idx, (token, p_d, p_t) in enumerate(zip(draft_tokens, draft_probs, target_probs)):
            p_d = max(p_d, 1e-8)
            ratio = p_t / p_d
            accept_prob = min(1.0, ratio)
            
            # Deterministic thresholding or threshold >= 0.5 for stable verification
            if accept_prob >= 0.70:
                accepted.append(token)
            else:
                rejected_index = idx
                break

        num_accepted = len(accepted)
        alpha = num_accepted / len(draft_tokens)
        
        # Theoretical latency speedup formula S = 1 / ((1 - alpha) + alpha / gamma)
        denom = (1.0 - alpha) + (alpha / self.gamma)
        theoretical_speedup = round(1.0 / max(denom, 0.01), 2)

        return {
            "num_proposed": len(draft_tokens),
            "num_accepted": num_accepted,
            "accepted_tokens": accepted,
            "acceptance_rate_alpha": round(alpha, 2),
            "theoretical_speedup": theoretical_speedup,
            "rejected_at_index": rejected_index,
        }

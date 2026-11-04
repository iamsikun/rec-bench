"""End-to-end Target Attention (ETA) for CTR."""

from __future__ import annotations

import torch

from recsys.algorithms.sim import TwoStageInterest


class ETA(TwoStageInterest):
    """ETA (Chen et al., DLP-KDD 2022): SimHash signatures retrieve behaviors end to end by Hamming distance.

    Behavior and target embeddings are hashed with a fixed random projection
    into ``hash_bits`` sign bits (SimHash, FuxiCTR's ``reuse_hash=True``); the
    ``top_k`` behaviors with the most matching bits feed multi-head target
    attention, and a short-term branch attends over the most recent
    ``short_len`` behaviors. Retrieval uses the current embeddings, so it
    follows them as they train. Deviation: since histories hold at most 50
    items, ``top_k=20`` and ``short_len=10`` replace FuxiCTR's 50/50 on
    thousands-long sequences, which limits the gain from retrieval.
    """

    space = {"top_k": [10, 20], "hash_bits": [16, 32], "dropout": [0.0, 0.1], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, hash_bits: int = 32, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hash_bits = hash_bits

    def network(self, data) -> None:
        super().network(data)
        self.register_buffer("rotations", torch.randn(self.dim, self.hash_bits))

    def search(self, q, keys, mask):
        with torch.no_grad():
            signature = (keys @ self.rotations) > 0
            target = (q @ self.rotations).unsqueeze(1) > 0
            return (signature == target).sum(-1).float()  # hash_bits - Hamming distance

"""LIGER (Yang et al., TMLR 2025): generative retrieval of semantic IDs re-ranked by dense retrieval.

One encoder-decoder (TIGER's) is trained with two objectives: next-item
semantic-ID cross-entropy and a dense contrastive loss between the decoder's
first output and the target item's content representation. At inference,
constrained beam search proposes candidates, cold-start items are added, and
the union is ranked by the dense score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._semantic_ids import item_content
from recsys.algorithms.tiger import TIGER


class LIGER(TIGER):
    """LIGER (Yang et al., TMLR 2025): semantic-ID beam search proposes candidates, a dense head ranks them.

    Each history item enters the encoder as its semantic-ID token embeddings
    plus a learned projection of its content vector. The decoder output at the
    BOS position is the user's dense query; it is trained with an InfoNCE loss
    over the full catalog on the cosine similarity to the projected content
    of the target item, added to TIGER's generation loss. ``recommend`` decodes
    ``max(beam, k)`` candidates, appends items never seen in training (which
    generative retrieval rarely reaches), drops the user's history, and returns
    the top ``k`` by cosine similarity.

    Deviations from the paper: (1) as in our TIGER, item content is a
    truncated SVD of the train matrix plus one-hot side features instead of
    Sentence-T5 text embeddings, so "cold-start" items are those without train
    interactions, whose content is their side features only; (2) TIGER's
    smaller Transformer, inductive history-only input, and pruning of fully
    seen prefixes (see TIGER); (3) the dense temperature ``temp`` and the loss
    weight ``dense_weight`` are our choices.
    """

    space = {**TIGER.space, "temp": [0.05, 0.1]}

    def __init__(self, temp: float = 0.07, dense_weight: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temp, self.dense_weight = temp, dense_weight

    def build(self, data) -> None:
        super().build(data)
        x = item_content(data, self.n_factors, data.seed)
        self.register_buffer("_content", F.pad(x, (0, 0, 1, 0)), persistent=False)  # row 0 = padding item
        cold = np.bincount(data.history.items, minlength=data.n_items) == 0
        self.register_buffer("_cold", torch.as_tensor(np.flatnonzero(cold)), persistent=False)
        self.item_proj = nn.Linear(x.shape[1], self.dim, bias=False)

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self._tokens[seq].flatten(1)
        pad = tok == 0
        pad[:, -1] = False  # never a fully masked row
        content = self.item_proj(self._content[seq]).repeat_interleave(self._tokens.shape[1], dim=1)
        x = self.tok_emb(tok) + content + self.enc_pos(torch.arange(tok.shape[1], device=tok.device))
        return self.encoder(F.dropout(x, self.dropout, self.training), src_key_padding_mask=pad), pad

    def hidden(self, prefix: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Decoder states ``(B, T, dim)`` for decoder inputs ``prefix``."""
        T = prefix.shape[1]
        y = self.tok_emb(prefix) + self.dec_pos(torch.arange(T, device=prefix.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=prefix.device)
        return self.decoder(y, memory, tgt_mask=mask, tgt_is_causal=True, memory_key_padding_mask=pad)

    def decode(self, prefix: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.hidden(prefix, memory, pad) @ self.tok_emb.weight.T + self._level_mask[: prefix.shape[1]]

    def dense(self, query: torch.Tensor) -> torch.Tensor:
        """Cosine similarity / ``temp`` of user queries ``(B, dim)`` to every item ``(B, n_items)``."""
        items = F.normalize(self.item_proj(self._content[1:]), dim=-1)
        return F.normalize(query, dim=-1) @ items.T / self.temp

    def loss(self, batch):
        seq, target = batch
        tgt = self._tokens[target]
        bos = torch.ones_like(tgt[:, :1])
        h = self.hidden(torch.cat([bos, tgt[:, :-1]], 1), *self.encode(seq))
        logits = h @ self.tok_emb.weight.T + self._level_mask[: h.shape[1]]
        generative = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
        return generative + self.dense_weight * F.cross_entropy(self.dense(h[:, 0]), target - 1)

    def _beam_search(self, users, history, k):
        cand = super()._beam_search(users, history, max(self.beam, k))
        if len(self._cold):
            seen = torch.as_tensor(history.matrix(users)[:, self._cold.cpu().numpy()].toarray() > 0,
                                   device=self.device)
            cold = self._cold.expand(len(users), -1)
            dup = (cand[:, :, None] == cold[:, None, :]).any(dim=1)
            cand = torch.cat([cand, cold.masked_fill(seen | dup, -1)], dim=1)
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        memory, pad = self.encode(seq)
        query = self.hidden(torch.ones(len(users), 1, dtype=torch.long, device=self.device), memory, pad)[:, 0]
        score = self.dense(query).gather(1, cand.clamp(min=0)).masked_fill(cand < 0, -torch.inf)
        score, idx = score.topk(min(k, cand.shape[1]), dim=1)
        items = cand.gather(1, idx).masked_fill(score == -torch.inf, -1)
        return F.pad(items, (0, k - items.shape[1]), value=-1)

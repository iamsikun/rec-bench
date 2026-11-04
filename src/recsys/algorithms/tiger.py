"""TIGER (Rajput et al., NeurIPS 2023): generative retrieval that decodes the next item's semantic ID.

Items are tokenized into semantic IDs by an RQ-VAE (``levels`` codebooks of
``codes`` entries plus a collision token), and an encoder-decoder
Transformer reads the history as semantic-ID tokens and generates the next
item's ID with beam search constrained to valid prefixes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._semantic_ids import RQVAE, disambiguate, item_content, prefix_tables
from recsys.algorithms.trainer import TorchRecommender, minibatches


class TIGER(TorchRecommender):
    """TIGER (Rajput et al., NeurIPS 2023): seq2seq generation of semantic item IDs with constrained beam search.

    Deviations from the paper: (1) our benchmarks carry no item text, so the
    RQ-VAE input is a truncated SVD of the train interaction matrix plus
    one-hot item side features instead of Sentence-T5 embeddings; (2) no
    hashed user-id token (the model is inductive: history only); (3) a smaller
    Transformer (2+2 layers, d=128, FF 512) trained with Adam at a constant
    learning rate and early stopping instead of 200k steps of inverse-sqrt
    decay, where an "epoch" is ``epoch_examples`` random (history, next item)
    pairs; (4) the codebooks are refit by k-means on the trained encoder's
    latents, which cuts collisions, and the collision token's vocabulary is
    sized to the largest collision group rather than fixed; (5) decoding prunes prefixes whose
    items are all in the history, so beams are spent on new items.
    """

    tasks = frozenset({"ranking"})
    requires = frozenset({"timestamps"})
    inductive = True
    space = {"dim": [64, 128], "dropout": [0.1, 0.2], "max_len": [20, 50]}

    def __init__(self, dim: int = 128, n_layers: int = 2, n_heads: int = 4, ff: int = 512, max_len: int = 20,
                 dropout: float = 0.1, levels: int = 3, codes: int = 256, beam: int = 20, rq_steps: int = 1000,
                 n_factors: int = 64, epoch_examples: int | None = 100_000, *, epochs: int = 50,
                 batch_size: int = 256, lr: float = 1e-3, patience: int | None = 5, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim, self.n_layers, self.n_heads, self.ff = dim, n_layers, n_heads, ff
        self.max_len, self.dropout, self.levels, self.codes = max_len, dropout, levels, codes
        self.beam, self.rq_steps, self.n_factors, self.epoch_examples = beam, rq_steps, n_factors, epoch_examples

    def tokenize(self, data) -> np.ndarray:
        """Unique semantic IDs ``(n_items, levels + 1)``: RQ-VAE codes over item content plus a collision token."""
        x = item_content(data, self.n_factors, data.seed).to(self.device)
        rq = RQVAE(x.shape[1], levels=self.levels, codes=self.codes).to(self.device)
        return disambiguate(rq.fit(x, self.rq_steps).cpu().numpy())

    def build(self, data) -> None:
        self.sids = self.tokenize(data)
        self.sizes = [self.codes] * self.levels + [int(self.sids[:, -1].max()) + 1]
        self.offsets = np.r_[0, np.cumsum(self.sizes)[:-1]] + 2  # token 0 = pad, 1 = BOS
        child, path = prefix_tables(self.sids, self.sizes)
        self.child = [torch.as_tensor(t, device=self.device) for t in child]
        self.path = [torch.as_tensor(p, device=self.device) for p in path]
        vocab, depth = 2 + sum(self.sizes), len(self.sizes)
        tokens = np.vstack([np.zeros((1, depth), dtype=np.int64), self.sids + self.offsets])  # row 0 = padding item
        self.register_buffer("_tokens", torch.as_tensor(tokens), persistent=False)
        level_mask = torch.full((depth, vocab), -torch.inf)
        for t, (o, s) in enumerate(zip(self.offsets, self.sizes)):
            level_mask[t, o : o + s] = 0.0
        self.register_buffer("_level_mask", level_mask, persistent=False)
        self.tok_emb = nn.Embedding(vocab, self.dim, padding_idx=0)
        self.enc_pos = nn.Embedding(self.max_len * depth, self.dim)
        self.dec_pos = nn.Embedding(depth, self.dim)
        for emb in (self.tok_emb, self.enc_pos, self.dec_pos):
            nn.init.normal_(emb.weight, std=0.02)
        kw = dict(dim_feedforward=self.ff, dropout=self.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.dim, self.n_heads, **kw), self.n_layers,
                                             norm=nn.LayerNorm(self.dim), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.dim, self.n_heads, **kw), self.n_layers,
                                             norm=nn.LayerNorm(self.dim))
        # Training examples: every position after the first of each user's train sequence.
        h = data.history
        starts = np.repeat(h.indptr[:-1], h.lengths())
        keep = np.arange(len(h.items)) > starts
        for name, arr in (("_items", h.items + 1), ("_ends", np.flatnonzero(keep)), ("_starts", starts[keep])):
            self.register_buffer(name, torch.as_tensor(arr), persistent=False)

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder memory and padding mask for shifted item sequences ``(B, max_len)`` (0 = padding)."""
        tok = self._tokens[seq].flatten(1)
        pad = tok == 0
        pad[:, -1] = False  # never a fully masked row
        x = self.tok_emb(tok) + self.enc_pos(torch.arange(tok.shape[1], device=tok.device))
        return self.encoder(F.dropout(x, self.dropout, self.training), src_key_padding_mask=pad), pad

    def decode(self, prefix: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Next-token logits ``(B, T, vocab)`` for decoder inputs ``prefix`` (``BOS`` + generated tokens)."""
        T = prefix.shape[1]
        y = self.tok_emb(prefix) + self.dec_pos(torch.arange(T, device=prefix.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=prefix.device)
        h = self.decoder(y, memory, tgt_mask=mask, tgt_is_causal=True, memory_key_padding_mask=pad)
        return h @ self.tok_emb.weight.T + self._level_mask[:T]

    def batches(self, data):
        steps = torch.arange(self.max_len, device=self.device)
        n = len(self._ends)
        sample = torch.randperm(n, device=self.device)[: self.epoch_examples or n]
        for idx in minibatches(len(sample), self.batch_size, self.device):
            idx = sample[idx]
            ends, starts = self._ends[idx], self._starts[idx]
            pos = ends[:, None] - self.max_len + steps
            yield torch.where(pos >= starts[:, None], self._items[pos.clamp(min=0)], 0), self._items[ends]

    def loss(self, batch):
        seq, target = batch
        tgt = self._tokens[target]
        bos = torch.ones_like(tgt[:, :1])
        logits = self.decode(torch.cat([bos, tgt[:, :-1]], 1), *self.encode(seq))
        return F.cross_entropy(logits.flatten(0, 1), tgt.flatten())

    @torch.no_grad()
    def recommend(self, users, history, k, chunk: int = 256):
        out = [self._beam_search(users[i : i + chunk], history, k) for i in range(0, len(users), chunk)]
        return torch.cat(out) if out else torch.full((0, k), -1, device=self.device)

    def _beam_search(self, users, history, k):
        B, W = len(users), max(self.beam, k)
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        unseen = torch.as_tensor(history.matrix(users).toarray() == 0, device=self.device).float()
        memory, pad = self.encode(seq)
        score = torch.zeros(B, 1, device=self.device)
        prefix = torch.zeros(B, 1, dtype=torch.long, device=self.device)  # prefix id per beam
        tokens = torch.ones(B, 1, 1, dtype=torch.long, device=self.device)  # BOS + generated tokens
        for t, (off, size) in enumerate(zip(self.offsets, self.sizes)):
            w = tokens.shape[1]
            logits = self.decode(tokens.flatten(0, 1), memory.repeat_interleave(w, 0), pad.repeat_interleave(w, 0))
            logp = logits[:, -1, off : off + size].log_softmax(-1).view(B, w, size)
            child = self.child[t][prefix.clamp(min=0)]  # (B, w, size)
            invalid = (child < 0) | (prefix < 0).unsqueeze(-1)
            # Prune prefixes whose items are all in the history, so beams end on new items.
            fresh = unseen.new_zeros(B, len(self.child[t + 1]) if t + 1 < len(self.child) else len(unseen[0]))
            fresh = fresh.index_add_(1, self.path[t], unseen) > 0
            invalid |= ~fresh.gather(1, child.clamp(min=0).flatten(1)).view_as(child)
            total = (score.unsqueeze(-1) + logp).masked_fill(invalid, -torch.inf).flatten(1)
            score, idx = total.topk(min(W, total.shape[1]), dim=1)
            beam, code = idx // size, idx % size
            prefix = child.flatten(1).gather(1, idx).masked_fill(score == -torch.inf, -1)
            tokens = torch.cat([tokens.gather(1, beam[..., None].expand(-1, -1, tokens.shape[2])),
                                (code + off)[..., None]], dim=2)
        items = prefix[:, :k]
        return F.pad(items, (0, k - items.shape[1]), value=-1)

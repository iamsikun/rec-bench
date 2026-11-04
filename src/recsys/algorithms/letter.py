"""LETTER (Wang et al., CIKM 2024): a learnable item tokenizer for generative recommendation.

The RQ-VAE of TIGER is trained with two extra terms: a collaborative
regularizer that aligns each item's quantized embedding with a collaborative
item embedding (in-batch contrastive), and a diversity regularizer that
contrasts code embeddings within k-means clusters of each codebook to spread
code assignments. The generative recommender is TIGER's, trained with the
ranking-guided loss (cross-entropy over temperature-scaled logits).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from recsys.algorithms._semantic_ids import RQVAE, disambiguate, item_content, item_transitions, kmeans
from recsys.algorithms.tiger import TIGER


class LETTER(TIGER):
    """LETTER (Wang et al., CIKM 2024): RQ-VAE semantic IDs regularized toward collaborative signal and code diversity.

    The tokenizer loss is ``recon + rq + alpha * L_cf + beta * L_div``.
    ``L_cf`` is an InfoNCE loss between an item's quantized latent and its
    collaborative embedding over a random batch of items; ``L_div`` samples, for
    each item and level, a positive code from the same k-means cluster of the
    codebook as the assigned code and contrasts it against all other codes.
    Generation then trains TIGER's encoder-decoder on these IDs with
    ``CE(logits / tau)``; a temperature below one sharpens the softmax and
    weights hard negatives more (the paper's ranking-guided generation loss).

    Deviations from the paper: (1) as in our TIGER, item content is a
    truncated SVD of the train matrix plus one-hot side features instead of
    LLaMA text embeddings; (2) the collaborative embedding is a truncated SVD
    of the symmetric next-item transition matrix (a sequential CF signal
    distinct from the co-consumption SVD in the content) instead of pretrained
    SASRec/LightGCN embeddings; (3) the collaborative contrast uses cosine
    similarity with temperature ``cf_temp`` rather than raw inner products,
    which is scale-free for our unnormalized latents; (4) codebooks are
    reclustered every step (5 Lloyd iterations) and are not refit after
    training, so the diversity term shapes the final assignment; no Sinkhorn
    assignment at the last level; (5) TIGER's smaller Transformer, inductive
    history-only input, and pruning of fully seen prefixes (see TIGER).
    """

    space = {**TIGER.space, "tau": [0.7, 1.0], "alpha": [0.02, 0.1]}

    def __init__(self, alpha: float = 0.02, beta: float = 1e-4, n_clusters: int = 10, tau: float = 0.7,
                 cf_temp: float = 0.1, rq_batch: int = 1024, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha, self.beta, self.n_clusters, self.tau = alpha, beta, n_clusters, tau
        self.cf_temp, self.rq_batch = cf_temp, rq_batch

    def tokenize(self, data) -> np.ndarray:
        x = item_content(data, self.n_factors, data.seed).to(self.device)
        rq = RQVAE(x.shape[1], levels=self.levels, codes=self.codes).to(self.device)
        cf = item_transitions(data, rq.codebooks.shape[-1], data.seed).to(self.device)

        def regularizer(z, zq, codes):
            idx = torch.randperm(len(z), device=z.device)[: self.rq_batch]
            sim = F.normalize(zq[idx], dim=-1) @ F.normalize(cf[idx], dim=-1).T / self.cf_temp
            loss = self.alpha * F.cross_entropy(sim, torch.arange(len(idx), device=z.device))
            return loss + self.beta * sum(self._diversity(book, codes[idx, t]) for t, book in enumerate(rq.codebooks))

        return disambiguate(rq.fit(x, self.rq_steps, regularizer=regularizer, refit=False).cpu().numpy())

    def _diversity(self, book: torch.Tensor, assigned: torch.Tensor) -> torch.Tensor:
        """Contrast each assigned code with a random code of its cluster against all other codes."""
        with torch.no_grad():
            cluster = kmeans(book, min(self.n_clusters, len(book)), iters=5)[1]
            same = cluster[assigned][:, None] == cluster[None, :]
            rows = torch.arange(len(assigned), device=book.device)
            same[rows, assigned] = False
            pos = torch.rand(same.shape, device=book.device).masked_fill(~same, -1.0).argmax(dim=1)
            ok = same.any(dim=1)
        sim = (book[assigned] @ book.T).index_put((rows, assigned), torch.tensor(-torch.inf, device=book.device))
        return F.cross_entropy(sim[ok], pos[ok]) if ok.any() else sim.new_zeros(())

    def loss(self, batch):
        seq, target = batch
        tgt = self._tokens[target]
        bos = torch.ones_like(tgt[:, :1])
        logits = self.decode(torch.cat([bos, tgt[:, :-1]], 1), *self.encode(seq))
        return F.cross_entropy(logits.flatten(0, 1) / self.tau, tgt.flatten())

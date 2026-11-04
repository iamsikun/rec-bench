"""SimGCL and XSimGCL (Yu et al., SIGIR 2022 / TKDE 2023): graph contrastive learning with noise-perturbed views."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recsys.algorithms.lightgcn import LightGCN


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE between matching rows of ``a`` and ``b`` (cosine similarity, in-batch negatives)."""
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    return F.cross_entropy(a @ b.T / tau, torch.arange(len(a), device=a.device))


class SimGCL(LightGCN):
    """SimGCL (Yu et al., SIGIR 2022): LightGCN plus InfoNCE between two views made by random embedding noise.

    Each propagation layer adds ``eps * sign(e) * normalize(U(0, 1))`` noise to
    the embeddings; two independently perturbed passes give two views, and an
    InfoNCE loss (temperature ``tau``, weight ``cl_weight``) over the batch's
    unique users and items pulls the views together, which makes the
    representation more uniform. The recommendation loss is BPR on a clean
    pass with L2 ``reg`` on the batch's propagated embeddings. As in the
    reference code (SELFRec), the final embedding is the mean of layers
    ``1..n_layers`` (the unperturbed layer 0 is left out). Scoring uses the
    clean propagation.
    """

    space = {"cl_weight": [0.1, 0.2, 0.5], "eps": [0.05, 0.1, 0.2], "lr": [1e-3, 5e-3]}

    def __init__(self, dim: int = 64, n_layers: int = 3, reg: float = 1e-4, eps: float = 0.1, tau: float = 0.2,
                 cl_weight: float = 0.2, *, batch_size: int = 2048, **kwargs) -> None:
        super().__init__(dim, n_layers, reg, batch_size=batch_size, **kwargs)
        self.eps = eps
        self.tau = tau
        self.cl_weight = cl_weight

    def layers(self, perturbed: bool) -> list[torch.Tensor]:
        """Embeddings after each of the ``n_layers`` propagations, noise-perturbed if ``perturbed``."""
        e, out = self.emb.weight, []
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self._adj, e)
            if perturbed:
                e = e + torch.sign(e) * F.normalize(torch.rand_like(e), dim=1) * self.eps
            out.append(e)
        return out

    def propagate(self, perturbed: bool = False) -> torch.Tensor:
        return torch.stack(self.layers(perturbed)).mean(0)

    def bpr(self, out: torch.Tensor, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        margin = (out[u] * (out[i] - out[j])).sum(1)
        l2 = out[u].square().sum() + out[i].square().sum() + out[j].square().sum()
        return -F.logsigmoid(margin).mean() + self.reg * l2 / len(u)

    def contrast(self, a: torch.Tensor, b: torch.Tensor, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """InfoNCE over the unique users plus over the unique items of the batch."""
        users, items = torch.unique(u), torch.unique(i)
        return info_nce(a[users], b[users], self.tau) + info_nce(a[items], b[items], self.tau)

    def loss(self, batch):
        u, i, j = batch
        cl = self.contrast(self.propagate(True), self.propagate(True), u, i)
        return self.bpr(self.propagate(), u, i, j) + self.cl_weight * cl


class XSimGCL(SimGCL):
    """XSimGCL (Yu et al., TKDE 2023): one noisy pass; contrast the final embedding with an intermediate layer.

    A single perturbed propagation serves both losses: BPR on the mean of the
    perturbed layers, and InfoNCE between that final embedding and the output
    of layer ``cl_layer`` (cross-layer contrast), which removes SimGCL's two
    extra passes. Scoring uses the clean (noise-free) propagation.
    """

    space = {"cl_weight": [0.05, 0.1, 0.2], "eps": [0.1, 0.2], "lr": [1e-3, 5e-3]}

    def __init__(self, dim: int = 64, n_layers: int = 2, reg: float = 1e-4, eps: float = 0.2, tau: float = 0.2,
                 cl_weight: float = 0.2, cl_layer: int = 1, **kwargs) -> None:
        super().__init__(dim, n_layers, reg, eps, tau, cl_weight, **kwargs)
        self.cl_layer = cl_layer

    def loss(self, batch):
        u, i, j = batch
        layers = self.layers(True)
        out = torch.stack(layers).mean(0)
        cl = self.contrast(out, layers[self.cl_layer - 1], u, i)
        return self.bpr(out, u, i, j) + self.cl_weight * cl

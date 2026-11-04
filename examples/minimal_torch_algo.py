import torch
import torch.nn.functional as F
from torch import nn

import recsys
from recsys.algorithms.trainer import minibatches


class TinyMF(recsys.TorchRecommender):
    tasks = frozenset({"ranking"})

    def __init__(self, dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, data):
        self.users = nn.Embedding(data.n_users, self.dim)
        self.items = nn.Embedding(data.n_items, self.dim)
        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)
        self.register_buffer("u", torch.tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("i", torch.tensor(data.train["item"].to_numpy()), persistent=False)

    def batches(self, data):
        for idx in minibatches(len(self.u), self.batch_size, self.device):
            yield self.u[idx], self.i[idx], torch.randint(data.n_items, (len(idx),), device=self.device)

    def loss(self, batch):
        u, i, j = batch
        return -F.logsigmoid((self.users(u) * (self.items(i) - self.items(j))).sum(1)).mean()

    def score(self, users, history):
        return self.users.weight[torch.as_tensor(users, device=self.device)] @ self.items.weight.T


if __name__ == "__main__":
    recsys.run_all(TinyMF, seeds=[0])

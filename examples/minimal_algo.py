import numpy as np
import torch

import recsys


class RecentPopularity(recsys.Recommender):
    tasks = frozenset({"ranking"})
    inductive = True

    def fit(self, data, window=10):
        self.window = window

    def score(self, users, history):
        recent = history.sequences(np.arange(history.n_users), self.window).ravel()
        counts = np.bincount(recent[recent >= 0], minlength=history.n_items)
        return torch.tensor(counts, dtype=torch.float32).repeat(len(users), 1)


if __name__ == "__main__":
    recsys.run_all(RecentPopularity, seeds=[0])

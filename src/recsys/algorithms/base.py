"""The ``Recommender`` base class every algorithm implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import polars as pl

    from recsys import Benchmark, History, TrainData


class Recommender(ABC):
    """Base class for all algorithms.

    Set :attr:`tasks` to the tasks the algorithm supports and implement:

    * ``"ranking"``: :meth:`score` (dense scores over the catalog) or
      :meth:`recommend` (top-k item ids, for retrieval and generative models
      that never materialize full score rows). The evaluator prefers ``score``.
    * ``"ctr"``: :meth:`predict`.

    :attr:`requires` lists benchmark inputs the algorithm needs
    (``user_features``, ``item_features``, ``behavior``); :attr:`inductive`
    says it can score users unseen in training from their history alone,
    which held-out-user benchmarks demand. Hyperparameters are constructor
    keyword arguments with defaults.
    """

    tasks: ClassVar[frozenset[str]] = frozenset()
    requires: ClassVar[frozenset[str]] = frozenset()
    inductive: ClassVar[bool] = False
    #: Hyperparameter grid for :func:`recsys.tune`: constructor argument -> candidate values.
    space: ClassVar[dict[str, list]] = {}

    @abstractmethod
    def fit(self, data: TrainData) -> None:
        """Fit on ``data.train``; may call ``data.validate(self)`` for model selection."""

    def score(self, users: np.ndarray, history: History) -> Any:
        """Return a new ``(len(users), n_items)`` array/tensor of scores; higher ranks first.

        ``history`` holds every interaction known at prediction time (in time
        order); the evaluator masks those items itself and breaks ties at random.
        """
        raise NotImplementedError

    def recommend(self, users: np.ndarray, history: History, k: int) -> Any:
        """Return ``(len(users), k)`` item ids, best first; ``-1`` pads short lists.

        Items already in ``history`` should not be returned (they count as misses).
        """
        raise NotImplementedError

    def predict(self, rows: pl.DataFrame) -> Any:
        """Return ``P(label = 1)`` for each row of encoded CTR fields."""
        raise NotImplementedError


def implements(algo: type[Recommender] | Recommender, method: str) -> bool:
    cls = algo if isinstance(algo, type) else type(algo)
    return getattr(cls, method) is not getattr(Recommender, method)


def incompatibility(algo: type[Recommender], benchmark: Benchmark) -> str | None:
    """Why ``algo`` cannot run on ``benchmark``, or ``None`` if it can."""
    if benchmark.task not in algo.tasks:
        return f"task {benchmark.task!r} not in {sorted(algo.tasks)}"
    missing = algo.requires - benchmark.provides
    if missing:
        return f"benchmark lacks {sorted(missing)}"
    if "cold_users" in benchmark.provides and not algo.inductive:
        return "test users are unseen in training and the algorithm is not inductive"
    return None

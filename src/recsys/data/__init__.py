"""Datasets: download the original public data and load it as :class:`Interactions`."""

from recsys.data.amazon import Amazon2023
from recsys.data.amazon2014 import Amazon2014
from recsys.data.bars import BARS
from recsys.data.base import Dataset, Interactions, data_root
from recsys.data.kuaishou import KuaiRand, KuaiRec
from recsys.data.lastfm import LastFM1K
from recsys.data.lightgcn import LightGCNData
from recsys.data.movielens import MovieLens
from recsys.data.synthetic import Synthetic

DATASETS: dict[str, type[Dataset]] = {
    "movielens": MovieLens,
    "amazon2023": Amazon2023,
    "amazon2014": Amazon2014,
    "lightgcn": LightGCNData,
    "bars": BARS,
    "lastfm1k": LastFM1K,
    "kuairec": KuaiRec,
    "kuairand": KuaiRand,
    "synthetic": Synthetic,
}

__all__ = [
    "BARS", "DATASETS", "Amazon2014", "Amazon2023", "Dataset", "Interactions", "KuaiRand", "KuaiRec", "LastFM1K",
    "LightGCNData", "MovieLens", "Synthetic", "data_root",
]

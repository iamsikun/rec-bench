"""Algorithms. Add new ones to :data:`ALGORITHMS` to include them in tests and the CLI."""

from recsys.algorithms.autoint import AutoInt
from recsys.algorithms.base import Recommender
from recsys.algorithms.bert4rec import BERT4Rec
from recsys.algorithms.bpr import BPR
from recsys.algorithms.dcn import DCN
from recsys.algorithms.dcnv2 import DCNv2
from recsys.algorithms.deepfm import DeepFM
from recsys.algorithms.dien import DIEN
from recsys.algorithms.din import DIN
from recsys.algorithms.dssm import DSSM
from recsys.algorithms.ease import EASE
from recsys.algorithms.finalmlp import FinalMLP
from recsys.algorithms.fm import FM
from recsys.algorithms.gru4rec import GRU4Rec
from recsys.algorithms.gsasrec import GSASRec
from recsys.algorithms.hstu import HSTU
from recsys.algorithms.ials import IALS
from recsys.algorithms.itemknn import ItemKNN
from recsys.algorithms.lightgcn import LightGCN
from recsys.algorithms.lr import LR
from recsys.algorithms.multvae import MultDAE, MultVAE
from recsys.algorithms.neumf import NeuMF
from recsys.algorithms.popularity import Popularity
from recsys.algorithms.puresvd import PureSVD
from recsys.algorithms.random import Random
from recsys.algorithms.rp3beta import RP3beta
from recsys.algorithms.sasrec import SASRec
from recsys.algorithms.sbcnn import TwoTowerLogQ
from recsys.algorithms.sknn import SKNN
from recsys.algorithms.slim import SLIM
from recsys.algorithms.srgnn import SRGNN
from recsys.algorithms.tiger import TIGER
from recsys.algorithms.trainer import TorchRecommender
from recsys.algorithms.userknn import UserKNN
from recsys.algorithms.widedeep import WideDeep
from recsys.algorithms.xdeepfm import XDeepFM
from recsys.algorithms.youtubednn import YouTubeDNN

ALGORITHMS: dict[str, type[Recommender]] = {
    "random": Random,
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "userknn": UserKNN,
    "slim": SLIM,
    "rp3beta": RP3beta,
    "puresvd": PureSVD,
    "ials": IALS,
    "sknn": SKNN,
    "ease": EASE,
    "bpr": BPR,
    "lightgcn": LightGCN,
    "sasrec": SASRec,
    "gsasrec": GSASRec,
    "bert4rec": BERT4Rec,
    "gru4rec": GRU4Rec,
    "srgnn": SRGNN,
    "neumf": NeuMF,
    "multvae": MultVAE,
    "multdae": MultDAE,
    "lr": LR,
    "fm": FM,
    "widedeep": WideDeep,
    "deepfm": DeepFM,
    "dcn": DCN,
    "dcnv2": DCNv2,
    "xdeepfm": XDeepFM,
    "autoint": AutoInt,
    "finalmlp": FinalMLP,
    "din": DIN,
    "dien": DIEN,
    "dssm": DSSM,
    "youtubednn": YouTubeDNN,
    "twotower_logq": TwoTowerLogQ,
    "tiger": TIGER,
    "hstu": HSTU,
}

__all__ = [
    "ALGORITHMS", "BPR", "DeepFM", "EASE", "LightGCN", "Popularity", "Recommender", "SASRec",
    "TorchRecommender",
]

"""Algorithms. Add new ones to :data:`ALGORITHMS` to include them in tests and the CLI."""

from recsys.algorithms.autoint import AutoInt
from recsys.algorithms.base import Recommender
from recsys.algorithms.bert4rec import BERT4Rec
from recsys.algorithms.bpr import BPR
from recsys.algorithms.bsarec import BSARec
from recsys.algorithms.bst import BST
from recsys.algorithms.cl4srec import CL4SRec
from recsys.algorithms.dcn import DCN
from recsys.algorithms.dcnv2 import DCNv2
from recsys.algorithms.deepfm import DeepFM
from recsys.algorithms.dien import DIEN
from recsys.algorithms.din import DIN
from recsys.algorithms.dssm import DSSM
from recsys.algorithms.duorec import DuoRec
from recsys.algorithms.ease import EASE
from recsys.algorithms.eta import ETA
from recsys.algorithms.finalmlp import FinalMLP
from recsys.algorithms.finalnet import FinalNet
from recsys.algorithms.fm import FM
from recsys.algorithms.fmlprec import FMLPRec
from recsys.algorithms.gdcn import GDCN
from recsys.algorithms.gfcf import GFCF
from recsys.algorithms.gru4rec import GRU4Rec
from recsys.algorithms.gsasrec import GSASRec
from recsys.algorithms.hstu import HSTU
from recsys.algorithms.ials import IALS
from recsys.algorithms.itemknn import ItemKNN
from recsys.algorithms.letter import LETTER
from recsys.algorithms.liger import LIGER
from recsys.algorithms.lightgcn import LightGCN
from recsys.algorithms.lr import LR
from recsys.algorithms.mamba4rec import Mamba4Rec
from recsys.algorithms.masknet import MaskNet
from recsys.algorithms.multvae import MultDAE, MultVAE
from recsys.algorithms.neumf import NeuMF
from recsys.algorithms.popularity import Popularity
from recsys.algorithms.puresvd import PureSVD
from recsys.algorithms.random import Random
from recsys.algorithms.recvae import RecVAE
from recsys.algorithms.rp3beta import RP3beta
from recsys.algorithms.sasrec import SASRec
from recsys.algorithms.sbcnn import TwoTowerLogQ
from recsys.algorithms.sim import SIM
from recsys.algorithms.simgcl import SimGCL, XSimGCL
from recsys.algorithms.sknn import SKNN
from recsys.algorithms.slim import SLIM
from recsys.algorithms.srgnn import SRGNN
from recsys.algorithms.tiger import TIGER
from recsys.algorithms.tisasrec import TiSASRec
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
    "gfcf": GFCF,
    "bpr": BPR,
    "lightgcn": LightGCN,
    "simgcl": SimGCL,
    "xsimgcl": XSimGCL,
    "sasrec": SASRec,
    "gsasrec": GSASRec,
    "bert4rec": BERT4Rec,
    "tisasrec": TiSASRec,
    "cl4srec": CL4SRec,
    "duorec": DuoRec,
    "fmlprec": FMLPRec,
    "bsarec": BSARec,
    "mamba4rec": Mamba4Rec,
    "gru4rec": GRU4Rec,
    "srgnn": SRGNN,
    "neumf": NeuMF,
    "multvae": MultVAE,
    "multdae": MultDAE,
    "recvae": RecVAE,
    "lr": LR,
    "fm": FM,
    "widedeep": WideDeep,
    "deepfm": DeepFM,
    "dcn": DCN,
    "dcnv2": DCNv2,
    "xdeepfm": XDeepFM,
    "autoint": AutoInt,
    "finalmlp": FinalMLP,
    "gdcn": GDCN,
    "finalnet": FinalNet,
    "masknet": MaskNet,
    "din": DIN,
    "dien": DIEN,
    "bst": BST,
    "sim": SIM,
    "eta": ETA,
    "dssm": DSSM,
    "youtubednn": YouTubeDNN,
    "twotower_logq": TwoTowerLogQ,
    "tiger": TIGER,
    "hstu": HSTU,
    "letter": LETTER,
    "liger": LIGER,
}

__all__ = [
    "ALGORITHMS", "BPR", "DeepFM", "EASE", "LightGCN", "Popularity", "Recommender", "SASRec",
    "TorchRecommender",
]

"""Built-in benchmarks. Never change a definition once results exist; add a new name instead."""

from __future__ import annotations

import dataclasses

from recsys.benchmarks.base import Benchmark
from recsys.data import BARS, Amazon2014, Amazon2023, KuaiRand, KuaiRec, LightGCNData, MovieLens, Synthetic
from recsys.protocol import (
    Binarize,
    DateParts,
    Dedup,
    HeldOutUsers,
    KCore,
    LeaveLastOut,
    LogSquareBucket,
    MinCount,
    MinValue,
    Predefined,
    RandomSplit,
    TemporalSplit,
)

RANKING = ("ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr@10")
CTR = ("auc", "logloss")

LOO_LEAKAGE = (
    "Dedup and 5-core filtering use all events before splitting; leave-last-out lets training contain "
    "interactions that happen after other users' test events (Ji et al., 2023). Compare with *-temporal."
)
RANDOM_CTR_LEAKAGE = (
    "Random row split: training rows can be later in time than test rows. Field vocabularies and behavior "
    "histories use train (and train + val for test rows) only."
)


def _loo(name: str, dataset, description: str, reference: str, user_fields=(), item_fields=()) -> Benchmark:
    return Benchmark(
        name, dataset, "ranking", LeaveLastOut(), RANKING, (Dedup(), KCore(5)),
        user_fields=user_fields, item_fields=item_fields,
        description=description, leakage=LOO_LEAKAGE, reference=reference,
    )


_SYNTH = "Generated data for tests; no reference."
_ALL = [
    # Generated data for tests and smoke runs: one benchmark per input/protocol type.
    Benchmark("synthetic-ranking", Synthetic(), "ranking", LeaveLastOut(), RANKING, (Dedup(), KCore(3)),
              user_fields=("segment",), item_fields=("category",),
              description="Generated sequences, leave-last-out, with side features.",
              leakage=LOO_LEAKAGE, reference=_SYNTH),
    Benchmark("synthetic-heldout", Synthetic(), "ranking", HeldOutUsers(30, 30, min_events=5), RANKING, (Dedup(),),
              description="Generated data, strong generalization (held-out users with fold-in).",
              leakage="Held-out users never appear in training; only items seen in training are scored.",
              reference=_SYNTH),
    Benchmark("synthetic-ctr", Synthetic(), "ctr", RandomSplit(), CTR, (Binarize("rating", 4),),
              fields=("user", "item", "segment", "category"), history=10,
              description="Generated ratings >= 4 as clicks, with behavior history.",
              leakage=RANDOM_CTR_LEAKAGE, reference=_SYNTH),
    # Next-item ranking: leave-last-out on 5-core data, full-catalog metrics.
    _loo("ml-100k-loo", MovieLens("100k"), "MovieLens-100K, all ratings as implicit feedback.",
         "Development benchmark; no published full-ranking reference.",
         user_fields=("gender", "age", "occupation"), item_fields=("genre",)),
    _loo("ml-1m-loo", MovieLens("1m"), "MovieLens-1M, all ratings as implicit feedback.",
         "SASRec/BERT4Rec full-ranking protocol (Klenitskiy & Vasilev, RecSys 2023; Petrov & Macdonald, "
         "RecSys 2022).",
         user_fields=("gender", "age", "occupation"), item_fields=("genre",)),
    _loo("ml-20m-loo", MovieLens("20m"), "MovieLens-20M, all ratings as implicit feedback.",
         "HSTU (Zhai et al., ICML 2024) uses ML-20M leave-one-out.", item_fields=("genre",)),
    _loo("amazon-games-loo", Amazon2023("Video_Games"), "Amazon Reviews 2023, Video_Games.",
         "No published reference on the 2023 release.", item_fields=("main_category",)),
    *[_loo(f"amazon14-{short}-loo", Amazon2014(cat), f"Amazon Reviews 2014 5-core {cat}.",
           "S3-Rec / TIGER protocol (Rajput et al., NeurIPS 2023, Table 1: full ranking, leave-one-out); "
           "files match S3-Rec's preprocessed data exactly.", item_fields=("category", "brand"))
      for short, cat in (("beauty", "Beauty"), ("sports", "Sports_and_Outdoors"), ("toys", "Toys_and_Games"))],
    # Graph CF: the train/test files released with NGCF/LightGCN.
    *[Benchmark(f"{name.replace('-', '')}-lightgcn", LightGCNData(name), "ranking", Predefined(val=0.05),
                ("recall@20", "ndcg@20"),
                description=f"{name} 10-core, LightGCN release train/test; 5% of train held out for validation.",
                leakage="Random 80/20 split per user from the original release (no time information); "
                        "the release has no validation split, so 5% of train rows are carved out at random.",
                reference="He et al. (SIGIR 2020), Table 3; Wang et al. (NGCF, SIGIR 2019), Table 2.")
      for name in ("gowalla", "yelp2018", "amazon-book")],
    # Future prediction: global temporal cutoffs, several targets per user.
    Benchmark("ml-1m-temporal", MovieLens("1m"), "ranking", TemporalSplit(0.1, 0.1), RANKING,
              (MinValue("rating", 4), Dedup(), KCore(5)),
              description="MovieLens-1M ratings >= 4, last 10% of time as test, previous 10% as val.",
              leakage="Leak-free in time: every training event precedes every val/test event. 5-core uses "
                      "all events.",
              reference="Global temporal split as recommended by Ji et al. (TOIS 2023); no fixed number."),
    # Strong generalization (Liang et al., 2018): held-out users, fold-in history.
    Benchmark("ml-20m-strong", MovieLens("20m"), "ranking", HeldOutUsers(), ("ndcg@100", "nrecall@20", "nrecall@50"),
              (MinValue("rating", 4), MinCount("user", 5)),
              description="MovieLens-20M ratings >= 4, users with >= 5 ratings, 10k val + 10k test held-out users, "
                          "80% fold-in.",
              leakage="Held-out users never appear in training. Items first seen in held-out users are dropped.",
              reference="Liang et al. (WWW 2018) protocol; EASE (Steck, WWW 2019) and iALS (Rendle et al., "
                        "RecSys 2022) report Recall@20/50 and NDCG@100 on it."),
    # Click-through rate prediction.
    Benchmark("ml-1m-ctr", MovieLens("1m"), "ctr", RandomSplit(), CTR, (Binarize("rating", 4),),
              fields=("user", "item", "gender", "age", "occupation", "genre"), history=50,
              description="MovieLens-1M, rating >= 4 as positive, random 80/10/10 split, 50-item behavior history.",
              leakage=RANDOM_CTR_LEAKAGE, reference="No published reference for this exact split."),
    Benchmark("kuairand-ctr", KuaiRand("pure"), "ctr", TemporalSplit(0.1, 0.1), CTR, (Binarize("is_click", 1),),
              fields=("user", "item", "tab", "user_active_degree", "author_id", "video_type", "upload_type",
                      "music_type"), history=50,
              description="KuaiRand-Pure standard logs, is_click label, temporal 80/10/10 split, behavior history.",
              leakage="Temporal split: training rows precede val/test rows. Vocabularies and histories use "
                      "train (and train + val for test rows) only.",
              reference="No published reference for this exact split."),
    Benchmark("avazu-x4", BARS("Avazu_x4"), "ctr", Predefined(), CTR, (DateParts(),),
              fields=("hour", "weekday", "weekend", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                      "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                      "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"),
              min_count=2,
              description="Avazu, BARS Avazu_x4_001 split (8:1:1 random), hour/weekday/weekend from the hour field.",
              leakage="Random row split published by BARS. Vocabularies use train only (min count 2).",
              reference="BARS / FuxiCTR Avazu_x4_001 (Zhu et al., CIKM 2021)."),
    Benchmark("criteo-x4", BARS("Criteo_x4"), "ctr", Predefined(), CTR,
              (LogSquareBucket(tuple(f"I{i}" for i in range(1, 14))),),
              fields=(*(f"I{i}" for i in range(1, 14)), *(f"C{i}" for i in range(1, 27))), min_count=10,
              description="Criteo, BARS Criteo_x4_001 split (8:1:1 random), numeric fields bucketed as floor(ln(x)^2).",
              leakage="Random row split published by BARS. Vocabularies use train only (min count 10).",
              reference="BARS / FuxiCTR Criteo_x4_001 (Zhu et al., CIKM 2021); DeepFM AUC 0.8143."),
    Benchmark("kuairec-ctr", KuaiRec("small"), "ctr", RandomSplit(), CTR, (Binarize("watch_ratio", 2.0),),
              fields=("user", "item", "category"),
              description="KuaiRec small matrix, watch_ratio >= 2 as positive, random 80/10/10 split.",
              leakage=RANDOM_CTR_LEAKAGE, reference="No published reference for this exact split."),
]

# Pinned (n_users, n_items, n_train, n_val_targets, n_test_targets); CTR: (user, item cardinality, rows).
# prepare() fails if a dataset or preprocessing change alters them.
STATS: dict[str, tuple[int, ...]] = {
    'synthetic-ranking': (300, 200, 6918, 300, 300),
    'synthetic-heldout': (300, 200, 6030, 134, 131),
    'synthetic-ctr': (301, 201, 7143, 920, 937),
    'ml-100k-loo': (943, 1349, 97401, 943, 943),
    'ml-1m-loo': (6040, 3416, 987531, 6040, 6040),
    'ml-1m-temporal': (6034, 3125, 459500, 57437, 57439),
    'ml-1m-ctr': (6041, 3680, 800051, 99891, 100267),
    'kuairand-ctr': (26249, 7541, 1149286, 143661, 143662),
    'kuairec-ctr': (1412, 3328, 3740507, 468370, 467693),
    'gowalla-lightgcn': (29858, 40981, 769748, 40380, 217242),
    'yelp2018-lightgcn': (31668, 38048, 1175431, 61828, 324147),
    'amazonbook-lightgcn': (52643, 91599, 2261672, 119058, 603378),
    'amazon14-beauty-loo': (22363, 12101, 153776, 22363, 22363),
    'amazon14-sports-loo': (35598, 18357, 225141, 35598, 35598),
    'amazon14-toys-loo': (19412, 11924, 128773, 19412, 19412),
    'ml-20m-loo': (138493, 18345, 19707038, 138493, 138493),
    'amazon-games-loo': (94762, 25612, 625062, 94762, 94762),
    'ml-20m-strong': (136677, 20103, 8527788, 141865, 138565),
    'avazu-x4': (0, 0, 32343172, 4042897, 4042898),
    'criteo-x4': (0, 0, 36672493, 4584062, 4584062),
}

BENCHMARKS: dict[str, Benchmark] = {b.name: dataclasses.replace(b, stats=STATS.get(b.name)) for b in _ALL}

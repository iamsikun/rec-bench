"""Parsing tests for the LightGCN, Amazon 2014 and BARS loaders on tiny fixture files (offline),
plus opt-in ``-m network`` tests that download the real data and check the published statistics."""

import gzip
import json

import polars as pl
import pytest

from recsys.data.amazon2014 import Amazon2014
from recsys.data.bars import BARS
from recsys.data.lastfm import LastFM1K
from recsys.data.lightgcn import LightGCNData


@pytest.mark.parametrize("dataset", [
    *(LightGCNData(n) for n in ("gowalla", "yelp2018", "amazon-book")),
    *(Amazon2014(c) for c in ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")),
    BARS("Criteo_x4"), BARS("Avazu_x4"), LastFM1K("artist"), LastFM1K("track"),
])
def test_pins_present(dataset):
    pins = dataset.pins()
    assert pins and all(len(p) in (40, 64) and int(p, 16) >= 0 for p in pins)


def test_unknown_names_rejected():
    for cls in (LightGCNData, Amazon2014, BARS, LastFM1K):
        with pytest.raises(ValueError):
            cls("nope")


def test_lightgcn_read(tmp_path):
    d = tmp_path / "gowalla"
    d.mkdir()
    (d / "train.txt").write_text("0 0 1 2\n1 2 3\n2\n")
    (d / "test.txt").write_text("0 3\n1 0 1\n")
    ds = LightGCNData("gowalla")
    assert ds.files_present(tmp_path)
    ev = ds.read(tmp_path).events
    assert ev.columns == ["user", "item", "split", "timestamp"]
    train = ev.filter(pl.col("split") == "train").select("user", "item").rows()
    assert train == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)]
    assert ev.filter(pl.col("split") == "test").select("user", "item").rows() == [(0, 3), (1, 0), (1, 1)]
    assert ev["timestamp"].unique().to_list() == [0] and ev["user"].dtype == pl.Int64


def _gz_lines(path, lines):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))


def test_amazon2014_read(tmp_path):
    reviews = [
        {"reviewerID": "U1", "asin": "A1", "overall": 5.0, "unixReviewTime": 1391040000, "reviewText": "x"},
        {"reviewerID": "U2", "asin": "A2", "overall": 3.0, "unixReviewTime": 1397779200, "helpful": [1, 1]},
    ]
    _gz_lines(tmp_path / "reviews_Beauty_5.json.gz", [json.dumps(r) for r in reviews])
    meta = [
        {"asin": "A1", "title": "it's", "price": 12.5, "brand": "Acme",
         "categories": [["Beauty", "Skin Care", "Face"]]},
        {"asin": "A2", "categories": [["Beauty"], ["Health"]]},
        {"asin": "A3", "price": 1.0, "categories": [["Beauty", "Hair"]]},  # not in events: dropped
    ]
    _gz_lines(tmp_path / "meta_Beauty.json.gz", [repr(m) for m in meta])  # the release uses Python literals
    data = Amazon2014("Beauty").read(tmp_path)
    assert data.events.rows() == [("U1", "A1", 5.0, 1391040000), ("U2", "A2", 3.0, 1397779200)]
    assert data.events.columns == ["user", "item", "rating", "timestamp"]
    assert data.items.sort("item").rows() == [("A1", "Face", "Acme", 12.5), ("A2", "Beauty", None, None)]
    no_meta = Amazon2014("Beauty", meta=False)
    assert no_meta.read(tmp_path).items is None and len(no_meta.pins()) == 1


_CRITEO_HEADER = "Label," + ",".join(f"I{i}" for i in range(1, 14)) + "," + ",".join(f"C{i}" for i in range(1, 27))
_AVAZU_HEADER = (
    "id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,"
    "device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21"
)


def test_bars_criteo_read(tmp_path):
    row = "1,7,-1,," + ",".join(["3"] * 10) + ",3c9d8785,ffffffff," + ",".join([""] * 24)
    for f in ("train", "valid", "test"):
        (tmp_path / "Criteo_x4").mkdir(exist_ok=True)
        (tmp_path / "Criteo_x4" / f"{f}.csv").write_text(f"{_CRITEO_HEADER}\n{row}\n")
    ds = BARS("Criteo_x4")
    assert ds.files_present(tmp_path)
    ev = ds.read(tmp_path).events
    assert ev["split"].cast(pl.String).to_list() == ["train", "val", "test"]
    assert ev.columns[:3] == ["label", "I1", "I2"] and ev.width == 1 + 13 + 26 + 1
    r = ev.row(0, named=True)
    assert (r["label"], r["I1"], r["I2"], r["I3"]) == (1, 7, -1, None)
    assert (r["C1"], r["C2"], r["C3"]) == (0x3C9D8785, 0xFFFFFFFF, None)
    assert ev["C1"].dtype == pl.UInt32 and ev["I1"].dtype == pl.Int32 and ev["label"].dtype == pl.UInt8


def test_bars_avazu_read(tmp_path):
    row = ("10002028568167339219,0,14102100,1005,0,9e8cf15d,0d3cb7be,f028772b,ecad2386,7801e8d9,07d7df22,"
           "a99f214a,58811cdf,8326c04b,1,2,20596,320,50,2161,0,35,-1,157")
    (tmp_path / "Avazu_x4").mkdir()
    for f in ("train", "valid", "test"):
        (tmp_path / "Avazu_x4" / f"{f}.csv").write_text(f"{_AVAZU_HEADER}\n{row}\n")
    ev = BARS("Avazu_x4").read(tmp_path).events
    r = ev.row(0, named=True)
    assert r["id"] == 10002028568167339219 and r["label"] == 0 and r["hour"] == 14102100
    assert r["timestamp"] == 1413849600  # 2014-10-21 00:00 UTC
    assert r["site_id"] == 0x9E8CF15D and r["C20"] == -1 and "click" not in ev.columns


def test_lastfm1k_read(tmp_path):
    d = tmp_path / "lastfm-dataset-1K"
    d.mkdir()
    (d / "userid-timestamp-artid-artname-traid-traname.tsv").write_text(
        "user_000001\t2009-05-04T23:08:57Z\tART1\tDeep Dish\t\tFuck Me (\"x\")\n"
        "user_000001\t2009-05-04T13:54:10Z\tART2\t坂本龍一\tTRK2\tComposition 0919\n"
        "user_000002\t2009-05-04T13:52:04Z\t\tUnknown\tTRK3\tSong\n"
    )
    (d / "userid-profile.tsv").write_text(
        "#id\tgender\tage\tcountry\tregistered\nuser_000001\tm\t\tJapan\tAug 13, 2006\n"
        "user_000002\tf\t22\tPeru\t\n"
    )
    artist = LastFM1K("artist").read(tmp_path)
    assert artist.events.select("user", "item", "timestamp").rows() == [
        ("user_000001", "ART1", 1241478537), ("user_000001", "ART2", 1241445250),
    ]
    assert artist.users.rows() == [("user_000001", "m", None, "Japan", "Aug 13, 2006"),
                                   ("user_000002", "f", 22, "Peru", None)]
    assert LastFM1K("track").read(tmp_path).events["item"].to_list() == ["TRK2", "TRK3"]


@pytest.mark.network
def test_lightgcn_gowalla_real():
    ev = LightGCNData("gowalla").load().events
    # LightGCN paper, Table 1: 29,858 users, 40,981 items, 1,027,370 interactions.
    assert (ev["user"].n_unique(), ev["item"].n_unique(), ev.height) == (29858, 40981, 1027370)


@pytest.mark.network
def test_amazon2014_toys_real():
    data = Amazon2014("Toys_and_Games", meta=False).load()
    ev = data.events
    # SASRec / S3-Rec / TIGER statistics for Toys: 19,412 users, 11,924 items, 167,597 reviews.
    assert (ev["user"].n_unique(), ev["item"].n_unique(), ev.height) == (19412, 11924, 167597)
    assert 9e8 < ev["timestamp"].min() and ev["timestamp"].max() < 1.5e9  # seconds, 1999-2014


@pytest.mark.network
def test_bars_avazu_real():  # downloads 1.3 GB
    ev = BARS("Avazu_x4").load().events
    counts = dict(ev.group_by(pl.col("split").cast(pl.String)).len().rows())
    assert counts == {"train": 32343172, "val": 4042897, "test": 4042898}

# Datasets

Every dataset is a frozen dataclass in `src/recsys/data/`. `load()` downloads the original files on first use and returns `Interactions(events, users, items)`. Downloads are cached under `$RECSYS_DATA` (default `~/.cache/recsys`), with one subdirectory per source. Nothing is stored in this repository.

| Class | Variants | Source | Size | `events` columns | Side tables |
|---|---|---|---|---|---|
| `MovieLens(variant)` | `100k`, `1m`, `10m`, `20m`, `25m`, `32m` | files.grouplens.org | 5 MB – 240 MB zip | user, item, rating, timestamp (s) | `items`: genre (first listed); `users` (100k, 1m): gender, age, occupation, zip |
| `Amazon2023(category)` | any category file name, e.g. `All_Beauty`, `Video_Games`, `Books` | Hugging Face `McAuley-Lab/Amazon-Reviews-2023` | 0.5 GB (All_Beauty; only 253 users after 5-core) – tens of GB | user, item (parent ASIN), rating, timestamp (ms) | `items`: store, main_category |
| `KuaiRec(variant)` | `small` (fully observed 1.4K × 3.3K), `big` | Zenodo | 432 MB zip | user, item, timestamp (s), watch_ratio, play_duration, video_duration | `items`: category |
| `KuaiRand(variant)` | `pure`, `1k`, `27k` | Zenodo | 47 MB – 40 GB | user, item, timestamp (ms), tab, is_click, is_like, long_view, play_time_ms, … | `users`: activity, follower and registration buckets, …; `items`: author_id, video_type, upload_type, music_type, tag, … |
| `Amazon2014(category)` | `Beauty`, `Sports_and_Outdoors`, `Toys_and_Games` (5-core) | SNAP mirror of McAuley et al. 2015 (`snap.stanford.edu/data/amazon/productGraph`) | 42–68 MB + 99–183 MB meta | user, item, rating, timestamp (day resolution) | `items`: category, brand, price |
| `LightGCNData(name)` | `gowalla`, `yelp2018`, `amazon-book` | the NGCF/LightGCN release (`kuandeng/LightGCN`, pinned commit) | 5–40 MB | user, item, split (train/test); **no time** (`has_time = False`, timestamp 0) | none |
| `BARS(name)` | `Criteo_x4`, `Avazu_x4` | Hugging Face `reczoo/<name>` (pinned revision + zip sha256) | 4.0 GB / 1.3 GB zip | label, raw fields, split (train/val/test); Avazu also timestamp | none |
| `LastFM1K(level)` | `artist`, `track` | MTG-UPF (`mtg.upf.edu`) | 673 MB | user, item, timestamp, artist/track ids and names | `users`: gender, age, country, registered |
| `Synthetic(...)` | sizes and seed | generated | — | user, item, timestamp, rating | `users`: segment; `items`: category |

**Notes**

- Ids are kept exactly as they appear in the source. Benchmarks encode them after filtering.
- Zenodo has throttled the KuaiRec archive at times. If the download stalls, put the extracted archive under `$RECSYS_DATA/kuairec/` by hand. The loader searches that directory for `small_matrix.csv` / `big_matrix.csv`.
- Amazon 2014's original host (jmcauley.ucsd.edu) no longer responds; the SNAP mirror serves byte-identical release files, pinned by sha256.
- Criteo_x4 is ~7 GB in memory once loaded; the CTR pipeline keeps encoded fields as int32.
- KuaiRand only loads the standard (recommender-driven) logs. The randomly exposed `log_random_*` files are left for future debiasing benchmarks.

## Adding a dataset

1. Subclass `Dataset` with `@dataclass(frozen=True)` and set `dirname`.
2. Implement:
   - `files_present(root)`
   - `download(root)`, using `recsys.data.download.fetch(url, root)` for archives
   - `read(root) -> Interactions`, renaming the id columns to `user` / `item` and the time column to `timestamp`
3. Register the class in `DATASETS` in `src/recsys/data/__init__.py`.
4. Add a row to the table above.

If the data can't be fetched automatically (for example, it requires accepting a license on Kaggle), `download` should raise an error that says where to get the files and where to put them.

Candidates, roughly in priority order:

- **Ranking:** Steam (1.3 GB, Python-literal format), Book-Crossing, Pinterest
- **Session-based:** Yoochoose (RecSys 2015; `.7z`, needs an extractor), Diginetica (Google Drive), RetailRocket
- **CTR:** AmazonElectronics_x1 (BARS, for DIN/DIEN), Taobao UserBehavior, Frappe
- **News:** MIND (the original blob currently refuses public access)

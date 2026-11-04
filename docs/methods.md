# Recommendation Methods Catalog

This catalog lists published recommendation algorithms so the library can track what exists and what to build next. It is a planning document, not a list of supported models. For what ships today, run `uv run recsys list algorithms`.

**Task types.** The **Task** column uses this library's terms:

- `ranking`: top-K item ranking from a user's interaction history, scored over the full catalog with Recall/NDCG/HR/MRR. Sequential and generative models use this task too.
- `ctr`: pointwise click probability over categorical (and dense) features, scored with AUC/LogLoss.

A few methods fit both (DIN, for example, is a CTR model whose history-aware scorer could also rank items). Methods that need multiple labels (multi-task) or a retrieval index are marked **out of scope for now**.

**Inputs.** `inter` = user-item interactions only (an unordered implicit matrix); `seq` = time-ordered interaction sequences; `feat` = categorical or dense side features (user, item, context); `kg` = knowledge graph; `text` = item or user text; `mm` = multimodal content.

**Tiers.**

- **Tier 1**: ubiquitous baselines that any credible benchmark must have. Reviewers expect them, and the "are we really making progress" literature shows they are often hard to beat.
- **Tier 2**: widely cited strong methods. They are common comparison points in recent papers.
- **Tier 3**: niche, superseded, industrial-scale-only, or very expensive (for example, it needs a 7B LLM).

**Status.** **Implemented** marks methods that already ship in the library: Popularity, EASE, BPR-MF, LightGCN, SASRec, DeepFM.

**Reference-implementation shorthand:** [RecBole](https://github.com/RUCAIBox/RecBole), [FuxiCTR](https://github.com/reczoo/FuxiCTR), [DeepCTR](https://github.com/shenweichen/DeepCTR), [Dacrema](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) (the baselines from "Are We Really Making Much Progress?", [arXiv:1907.06902](https://arxiv.org/abs/1907.06902)), [implicit](https://github.com/benfred/implicit), [Cornac](https://github.com/PreferredAI/cornac), [SELFRec](https://github.com/Coder-Yu/SELFRec), [session-rec](https://github.com/rn5l/session-rec). "Official" means the authors' repository. "—" means no well-known reference implementation.

Every arXiv ID and DOI below was checked against arXiv or Crossref (November 2026). Venues come from the arXiv comment or journal-ref field or from the DOI record. A † marks an entry whose venue or link could not be fully confirmed.

---

## 1. Non-personalized, neighborhood, and linear item-item models

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| Random | Uniformly random item scores; sanity floor | — | — | — | ranking | inter | RecBole | 1 | **Implemented** |
| Popularity (TopPop) | Rank items by global interaction count | — | — | — | ranking | inter | RecBole, Dacrema | 1 | **Implemented** |
| UserKNN | User-user similarity-weighted neighbor voting (GroupLens) | 1994 | CSCW | [10.1145/192844.192905](https://doi.org/10.1145/192844.192905) | ranking | inter | Dacrema, RecBole | 1 | **Implemented** |
| ItemKNN | Item-item cosine similarity CF | 2001 | WWW | [10.1145/371920.372071](https://doi.org/10.1145/371920.372071) | ranking | inter | Dacrema, RecBole | 1 | **Implemented** |
| ItemKNN top-N (Deshpande & Karypis) | Item-based top-N with normalized similarity | 2004 | TOIS | [10.1145/963770.963776](https://doi.org/10.1145/963770.963776) | ranking | inter | Dacrema | 2 | |
| Slope One | Average rating-difference predictor | 2005 | SDM | [arXiv:cs/0702144](https://arxiv.org/abs/cs/0702144) | ranking | inter | — | 3 | |
| SLIM | Sparse linear item-item regression with L1/L2 penalties | 2011 | ICDM | [10.1109/ICDM.2011.134](https://doi.org/10.1109/ICDM.2011.134) | ranking | inter | [Official](https://github.com/KarypisLab/SLIM), Dacrema | 1 | **Implemented** |
| P3alpha | Three-step random walk on the user-item graph | 2014 | WWW (companion) | [10.1145/2567948.2579244](https://doi.org/10.1145/2567948.2579244) | ranking | inter | Dacrema | 2 | |
| RP3beta | P3alpha with popularity-penalized re-ranking | 2016 | TiiS | [10.1145/2955101](https://doi.org/10.1145/2955101) | ranking | inter | Dacrema | 1 | **Implemented** |
| GLSLIM | Global plus user-cluster-local SLIM models | 2016 | RecSys | [10.1145/2959100.2959185](https://doi.org/10.1145/2959100.2959185) | ranking | inter | — | 3 | |
| EASE | Closed-form ridge item-item model with zero diagonal | 2019 | WWW | [arXiv:1905.03375](https://arxiv.org/abs/1905.03375) | ranking | inter | RecBole | 1 | **Implemented** |
| ADMM-SLIM | SLIM/EASE variants solved with ADMM (non-negativity, sparsity) | 2020 | WSDM | [10.1145/3336191.3371774](https://doi.org/10.1145/3336191.3371774) | ranking | inter | RecBole | 3 | |
| EDLAE | Emphasized denoising linear autoencoder (fixes EASE identity overfit) | 2020 | NeurIPS | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf) | ranking | inter | — | 2 | |
| SANSA | Sparse approximate-inverse EASE that scales to millions of items | 2023 | RecSys | [10.1145/3604915.3608827](https://doi.org/10.1145/3604915.3608827) | ranking | inter | [Official](https://github.com/glami/sansa) | 2 | |
| SKNN (session-kNN) | Nearest-neighbor sessions vote on the next item | 2017 | RecSys | [10.1145/3109859.3109872](https://doi.org/10.1145/3109859.3109872) | ranking | seq | session-rec | 1 | **Implemented** |
| V-SKNN / SR / AR | Position-weighted SKNN, sequential rules, association rules | 2018 | UMUAI | [arXiv:1803.09587](https://arxiv.org/abs/1803.09587) | ranking | seq | session-rec | 2 | |
| STAN | Session kNN with sequence- and time-aware decay | 2019 | SIGIR | [10.1145/3331184.3331322](https://doi.org/10.1145/3331184.3331322) | ranking | seq | session-rec | 2 | |

## 2. Matrix factorization and latent-factor models

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| WMF / iALS | Confidence-weighted MF for implicit feedback, solved by ALS | 2008 | ICDM | [10.1109/ICDM.2008.22](https://doi.org/10.1109/ICDM.2008.22) | ranking | inter | implicit, Cornac | 1 | **Implemented** |
| SVD++ | MF plus implicit-feedback neighborhood term | 2008 | KDD | [10.1145/1401890.1401944](https://doi.org/10.1145/1401890.1401944) | ranking | inter | Cornac | 3 | |
| BiasedMF / FunkSVD | MF with user and item biases (Netflix Prize style) | 2009 | IEEE Computer | [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263) | ranking | inter | Cornac | 2 | |
| timeSVD++ | SVD++ with temporal bias and factor drift | 2009 | KDD | [10.1145/1557019.1557072](https://doi.org/10.1145/1557019.1557072) | ranking | seq | — | 3 | |
| BPR-MF | MF trained with a pairwise Bayesian personalized ranking loss | 2009 | UAI | [arXiv:1205.2618](https://arxiv.org/abs/1205.2618) | ranking | inter | implicit, RecBole | 1 | **Implemented** |
| PureSVD | Truncated SVD of the zero-filled interaction matrix | 2010 | RecSys | [10.1145/1864708.1864721](https://doi.org/10.1145/1864708.1864721) | ranking | inter | Dacrema | 1 | **Implemented** |
| WARP | Weighted approximate-rank pairwise loss | 2011 | IJCAI | [PDF](https://www.ijcai.org/Proceedings/11/Papers/460.pdf) | ranking | inter | [LightFM](https://github.com/lyst/lightfm) | 2 | |
| FISM | Factored item similarity (item-item via low-rank factors) | 2013 | KDD | [10.1145/2487575.2487589](https://doi.org/10.1145/2487575.2487589) | ranking | inter | RecBole | 3 | |
| LightFM | Hybrid MF where entities are sums of feature embeddings | 2015 | CBRecSys@RecSys | [arXiv:1507.08439](https://arxiv.org/abs/1507.08439) | ranking | inter, feat | [Official](https://github.com/lyst/lightfm) | 2 | |
| CoFactor | MF jointly factorizing item co-occurrence (SPPMI) | 2016 | RecSys | [10.1145/2959100.2959182](https://doi.org/10.1145/2959100.2959182) | ranking | inter | Cornac | 3 | |
| Item2Vec | Skip-gram embeddings over item co-occurrence | 2016 | MLSP | [arXiv:1603.04259](https://arxiv.org/abs/1603.04259) | ranking | inter | — | 3 | |
| NeuMF / NCF | GMF plus MLP fusion over user and item embeddings | 2017 | WWW | [arXiv:1708.05031](https://arxiv.org/abs/1708.05031) | ranking | inter | RecBole, Dacrema | 1 | **Implemented** |
| DMF | Deep MF with MLP towers over interaction rows and columns | 2017 | IJCAI | [10.24963/ijcai.2017/447](https://doi.org/10.24963/ijcai.2017/447) | ranking | inter | RecBole | 3 | |
| CML | Collaborative metric learning in Euclidean space | 2017 | WWW | [10.1145/3038912.3052639](https://doi.org/10.1145/3038912.3052639) | ranking | inter | Cornac | 3 | |
| ConvNCF | CNN over the user-item outer-product map | 2018 | IJCAI | [arXiv:1808.03912](https://arxiv.org/abs/1808.03912) | ranking | inter | RecBole | 3 | |
| NAIS | Attentive item-similarity model (FISM with attention) | 2018 | TKDE | [arXiv:1809.07053](https://arxiv.org/abs/1809.07053) | ranking | inter | RecBole | 3 | |
| ENMF | Efficient non-sampling whole-data neural MF | 2020 | TOIS | [10.1145/3373807](https://doi.org/10.1145/3373807) | ranking | inter | RecBole | 3 | |
| SimpleX | MF with cosine contrastive loss and many negatives | 2021 | CIKM | [arXiv:2109.12613](https://arxiv.org/abs/2109.12613) | ranking | inter | [Official (reczoo)](https://github.com/reczoo/RecZoo), RecBole | 2 | |
| iALS++ / tuned iALS | Subspace-optimized iALS; shows tuned iALS stays competitive | 2022 | RecSys (revisit) / arXiv | [arXiv:2110.14037](https://arxiv.org/abs/2110.14037), [arXiv:2110.14044](https://arxiv.org/abs/2110.14044) | ranking | inter | [Official](https://github.com/google-research/google-research/tree/master/ials) | 2 | |
| DirectAU | Optimize alignment and uniformity of embeddings directly | 2022 | KDD | [arXiv:2206.12811](https://arxiv.org/abs/2206.12811) | ranking | inter | [Official](https://github.com/THUwangcy/DirectAU) | 2 | |

## 3. Autoencoder, VAE, and diffusion models (user-vector input)

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| AutoRec | Item- or user-based autoencoder reconstructing ratings | 2015 | WWW (companion) | [10.1145/2740908.2742726](https://doi.org/10.1145/2740908.2742726) | ranking | inter | Cornac | 3 | |
| CDAE | Denoising autoencoder with a user-specific node | 2016 | WSDM | [10.1145/2835776.2835837](https://doi.org/10.1145/2835776.2835837) | ranking | inter | RecBole | 2 | |
| CVAE (collaborative VAE) | VAE over item content plus CF | 2017 | KDD | [10.1145/3097983.3098077](https://doi.org/10.1145/3097983.3098077) | ranking | inter, text | Cornac | 3 | |
| Mult-VAE / Mult-DAE | Multinomial-likelihood VAE (and DAE) over user histories | 2018 | WWW | [arXiv:1802.05814](https://arxiv.org/abs/1802.05814) | ranking | inter | [Official](https://github.com/dawenl/vae_cf), RecBole | 1 | **Implemented** |
| MacridVAE | Disentangled macro/micro intent VAE | 2019 | NeurIPS | [arXiv:1910.14238](https://arxiv.org/abs/1910.14238) | ranking | inter | RecBole | 3 | |
| RaCT | Actor-critic fine-tuning of a VAE for ranking metrics | 2020 | ICLR | [arXiv:1906.04281](https://arxiv.org/abs/1906.04281) | ranking | inter | RecBole | 3 | |
| RecVAE | Mult-VAE with composite prior and alternating updates | 2020 | WSDM | [arXiv:1912.11160](https://arxiv.org/abs/1912.11160) | ranking | inter | [Official](https://github.com/ilya-shenbin/RecVAE), RecBole | 2 | |
| DiffRec / L-DiffRec / T-DiffRec | Denoising diffusion over user interaction vectors | 2023 | SIGIR | [arXiv:2304.04971](https://arxiv.org/abs/2304.04971) | ranking | inter | [Official](https://github.com/YiyanXu/DiffRec), RecBole | 2 | |

## 4. Graph-based collaborative filtering

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| GC-MC | Graph autoencoder for matrix completion | 2017 | arXiv (KDD DL workshop)† | [arXiv:1706.02263](https://arxiv.org/abs/1706.02263) | ranking | inter | RecBole | 3 | |
| PinSage | GraphSAGE with random-walk sampling at web scale | 2018 | KDD | [arXiv:1806.01973](https://arxiv.org/abs/1806.01973) | ranking | inter, feat | — | 3 | |
| SpectralCF | Spectral graph convolution on the bipartite graph | 2018 | RecSys | [arXiv:1808.10523](https://arxiv.org/abs/1808.10523) | ranking | inter | RecBole | 3 | |
| NGCF | Message passing with feature transforms and nonlinearities | 2019 | SIGIR | [arXiv:1905.08108](https://arxiv.org/abs/1905.08108) | ranking | inter | [Official](https://github.com/xiangwang1223/neural_graph_collaborative_filtering), RecBole | 2 | |
| LR-GCCF | Linear residual graph convolution CF | 2020 | AAAI | [arXiv:2001.10167](https://arxiv.org/abs/2001.10167) | ranking | inter | — | 3 | |
| DGCF | Disentangled intent-aware graph CF | 2020 | SIGIR | [arXiv:2007.01764](https://arxiv.org/abs/2007.01764) | ranking | inter | RecBole | 3 | |
| LightGCN | Linear neighborhood aggregation with layer averaging, no transforms | 2020 | SIGIR | [arXiv:2002.02126](https://arxiv.org/abs/2002.02126) | ranking | inter | [Official](https://github.com/gusye1234/LightGCN-PyTorch), RecBole | 1 | **Implemented** |
| SGL | LightGCN plus self-supervised graph augmentation (node/edge dropout) | 2021 | SIGIR | [arXiv:2010.10783](https://arxiv.org/abs/2010.10783) | ranking | inter | [Official](https://github.com/wujcan/SGL-Torch), RecBole | 2 | |
| UltraGCN | Skip message passing; approximate infinite-layer GCN with constraint losses | 2021 | CIKM | [arXiv:2110.15114](https://arxiv.org/abs/2110.15114) | ranking | inter | [Official (reczoo)](https://github.com/reczoo/RecZoo) | 2 | |
| GF-CF | Training-free linear plus ideal low-pass graph filter | 2021 | CIKM | [arXiv:2108.07567](https://arxiv.org/abs/2108.07567) | ranking | inter | [Official](https://github.com/yshenaw/GF_CF) | 2 | |
| SimGCL | Graph contrastive learning via uniform noise on embeddings | 2022 | SIGIR | [arXiv:2112.08679](https://arxiv.org/abs/2112.08679) | ranking | inter | SELFRec | 2 | |
| NCL | Neighborhood-enriched (structural plus prototype) contrastive learning | 2022 | WWW | [arXiv:2202.06200](https://arxiv.org/abs/2202.06200) | ranking | inter | [Official](https://github.com/RUCAIBox/NCL), RecBole | 2 | |
| HCCF | Hypergraph contrastive CF | 2022 | SIGIR | [arXiv:2204.12200](https://arxiv.org/abs/2204.12200) | ranking | inter | [Official](https://github.com/akaxlh/HCCF) | 3 | |
| SVD-GCN | Replace GCN with truncated-SVD spectral filter | 2022 | CIKM | [arXiv:2208.12689](https://arxiv.org/abs/2208.12689) | ranking | inter | — | 3 | |
| XSimGCL | SimGCL with cross-layer contrast and one forward pass | 2023 | TKDE | [arXiv:2209.02544](https://arxiv.org/abs/2209.02544) | ranking | inter | SELFRec | 2 | |
| LightGCL | SVD-guided graph augmentation for contrastive CF | 2023 | ICLR | [arXiv:2302.08191](https://arxiv.org/abs/2302.08191) | ranking | inter | [Official](https://github.com/HKUDS/LightGCL) | 2 | |
| BSPM | Blurring-sharpening (ODE) graph filter, training-free | 2023 | SIGIR | [arXiv:2211.09324](https://arxiv.org/abs/2211.09324) | ranking | inter | — | 3 | |
| Turbo-CF | Decomposition-free polynomial graph filter, under 1 second to fit | 2024 | SIGIR | [arXiv:2404.14243](https://arxiv.org/abs/2404.14243) | ranking | inter | [Official](https://github.com/jindeok/Turbo-CF) | 3 | |

## 5. Knowledge-graph-aware recommenders

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| CKE | Joint MF plus TransR KG, text, and image embeddings | 2016 | KDD | [10.1145/2939672.2939673](https://doi.org/10.1145/2939672.2939673) | ranking | inter, kg | RecBole | 3 | |
| RippleNet | Propagate user preferences outward over KG ripples | 2018 | CIKM | [arXiv:1803.03467](https://arxiv.org/abs/1803.03467) | ranking / ctr | inter, kg | RecBole | 3 | |
| KGCN | GCN over the item KG with user-specific relation weights | 2019 | WWW | [arXiv:1904.12575](https://arxiv.org/abs/1904.12575) | ranking / ctr | inter, kg | RecBole | 3 | |
| MKR | Multi-task recommendation plus KG embedding with cross units | 2019 | WWW | [arXiv:1901.08907](https://arxiv.org/abs/1901.08907) | ranking / ctr | inter, kg | RecBole | 3 | |
| KGNN-LS | KG GNN with label-smoothness regularization | 2019 | KDD† | [arXiv:1905.04413](https://arxiv.org/abs/1905.04413) | ranking / ctr | inter, kg | RecBole | 3 | |
| KGAT | Attentive propagation over the collaborative KG | 2019 | KDD | [arXiv:1905.07854](https://arxiv.org/abs/1905.07854) | ranking | inter, kg | [Official](https://github.com/xiangwang1223/knowledge_graph_attention_network), RecBole | 2 | |
| KGIN | Intent-aware relational paths over the KG | 2021 | WWW | [arXiv:2102.07057](https://arxiv.org/abs/2102.07057) | ranking | inter, kg | RecBole | 3 | |

## 6. Sequential and session-based recommendation

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| FPMC | MF plus a personalized first-order Markov chain | 2010 | WWW | [10.1145/1772690.1772773](https://doi.org/10.1145/1772690.1772773) | ranking | seq | RecBole | 2 | |
| GRU4Rec | Session-parallel GRU with ranking losses | 2016 | ICLR | [arXiv:1511.06939](https://arxiv.org/abs/1511.06939) | ranking | seq | [Official](https://github.com/hidasib/GRU4Rec), RecBole | 1 | **Implemented** |
| FOSSIL | FISM similarity fused with higher-order Markov chains | 2016 | ICDM | [arXiv:1609.09152](https://arxiv.org/abs/1609.09152) | ranking | seq | RecBole | 3 | |
| TransRec | Users as translation vectors between consecutive items | 2017 | RecSys | [arXiv:1707.02410](https://arxiv.org/abs/1707.02410) | ranking | seq | RecBole | 3 | |
| NARM | GRU encoder with attention over the session (local plus global) | 2017 | CIKM | [arXiv:1711.04725](https://arxiv.org/abs/1711.04725) | ranking | seq | RecBole | 2 | |
| GRU4Rec+ (top-k gains) | GRU4Rec with BPR-max/TOP1-max losses and extra negatives | 2018 | CIKM | [arXiv:1706.03847](https://arxiv.org/abs/1706.03847) | ranking | seq | [Official](https://github.com/hidasib/GRU4Rec) | 2 | |
| Caser | Horizontal and vertical CNN filters over the recent-item matrix | 2018 | WSDM | [arXiv:1809.07426](https://arxiv.org/abs/1809.07426) | ranking | seq | RecBole | 2 | |
| STAMP | Short-term attention/memory priority MLP | 2018 | KDD | [10.1145/3219819.3219950](https://doi.org/10.1145/3219819.3219950) | ranking | seq | RecBole | 3 | |
| SASRec | Causal self-attention (Transformer decoder) over item sequences | 2018 | ICDM | [arXiv:1808.09781](https://arxiv.org/abs/1808.09781) | ranking | seq | [Official](https://github.com/kang205/SASRec), RecBole | 1 | **Implemented** |
| NextItNet | Dilated causal residual CNN | 2019 | WSDM | [arXiv:1808.05163](https://arxiv.org/abs/1808.05163) | ranking | seq | RecBole | 3 | |
| RepeatNet | Repeat-vs-explore mixture for sessions | 2019 | AAAI | [arXiv:1812.02646](https://arxiv.org/abs/1812.02646) | ranking | seq | RecBole | 3 | |
| SR-GNN | Gated GNN over the session graph | 2019 | AAAI | [arXiv:1811.00855](https://arxiv.org/abs/1811.00855) | ranking | seq | [Official](https://github.com/CRIPAC-DIG/SR-GNN), RecBole | 1 | **Implemented** |
| GC-SAN | SR-GNN plus self-attention | 2019 | IJCAI | [10.24963/ijcai.2019/547](https://doi.org/10.24963/ijcai.2019/547) | ranking | seq | RecBole | 3 | |
| HGN | Hierarchical gating over item features and instances | 2019 | KDD | [arXiv:1906.09217](https://arxiv.org/abs/1906.09217) | ranking | seq | RecBole | 3 | |
| BERT4Rec | Bidirectional Transformer trained with a cloze (masked item) objective | 2019 | CIKM | [arXiv:1904.06690](https://arxiv.org/abs/1904.06690) | ranking | seq | [Official](https://github.com/FeiSun/BERT4Rec), RecBole | 1 | **Implemented** |
| TiSASRec | SASRec with time-interval-aware attention | 2020 | WSDM | [10.1145/3336191.3371786](https://doi.org/10.1145/3336191.3371786) | ranking | seq | — | 2 | |
| S3-Rec | Self-supervised pretraining with mutual-information objectives (attributes) | 2020 | CIKM | [arXiv:2008.07873](https://arxiv.org/abs/2008.07873) | ranking | seq, feat | [Official](https://github.com/RUCAIBox/CIKM2020-S3Rec), RecBole | 2 | |
| GCE-GNN | Global-context (cross-session) enhanced session GNN | 2020 | SIGIR | [arXiv:2106.05081](https://arxiv.org/abs/2106.05081) | ranking | seq | — | 2 | |
| LESSR | Lossless edge-order session graphs plus shortcut graph attention | 2020 | KDD | [10.1145/3394486.3403170](https://doi.org/10.1145/3394486.3403170) | ranking | seq | — | 3 | |
| SINE | Sparse-interest network (multi-interest, sequential) | 2021 | WSDM† | [arXiv:2102.09267](https://arxiv.org/abs/2102.09267) | ranking | seq | RecBole | 3 | |
| LightSANs | Low-rank decomposed self-attention | 2021 | SIGIR | [10.1145/3404835.3462978](https://doi.org/10.1145/3404835.3462978) | ranking | seq | RecBole | 3 | |
| CoSeRec | Robust (informative) augmentations for contrastive SR | 2021 | arXiv | [arXiv:2108.06479](https://arxiv.org/abs/2108.06479) | ranking | seq | [Official](https://github.com/YChen1993/CoSeRec) | 3 | |
| CL4SRec | SASRec plus contrastive loss (crop/mask/reorder augmentations) | 2022 | ICDE† | [arXiv:2010.14395](https://arxiv.org/abs/2010.14395) | ranking | seq | RecBole (via DuoRec repo) | 2 | |
| DuoRec | Model-level dropout plus supervised positive contrastive learning | 2022 | WSDM† | [arXiv:2110.05730](https://arxiv.org/abs/2110.05730) | ranking | seq | [Official](https://github.com/RuihongQiu/DuoRec) | 2 | |
| ICLRec | Intent (cluster) contrastive learning | 2022 | WWW† | [arXiv:2202.02519](https://arxiv.org/abs/2202.02519) | ranking | seq | [Official](https://github.com/salesforce/ICLRec) | 3 | |
| FMLP-Rec | All-MLP with learnable FFT frequency filters | 2022 | WWW | [arXiv:2202.13556](https://arxiv.org/abs/2202.13556) | ranking | seq | [Official](https://github.com/Woeee/FMLP-Rec) | 2 | |
| CORE | Consistent representation space via weighted item averaging | 2022 | SIGIR | [arXiv:2204.11067](https://arxiv.org/abs/2204.11067) | ranking | seq | [Official](https://github.com/RUCAIBox/CORE), RecBole | 3 | |
| MBHT | Multi-behavior hypergraph-enhanced Transformer | 2022 | KDD | [arXiv:2207.05584](https://arxiv.org/abs/2207.05584) | ranking | seq | — | 3 | |
| LinRec | L2-normalized linear attention for long sequences | 2023 | SIGIR | [10.1145/3539618.3591717](https://doi.org/10.1145/3539618.3591717) ([arXiv:2411.01537](https://arxiv.org/abs/2411.01537)) | ranking | seq | [Official](https://github.com/Applied-Machine-Learning-Lab/LinRec) | 3 | |
| FEARec | Frequency-enhanced hybrid attention plus contrastive learning | 2023 | SIGIR | [arXiv:2304.09184](https://arxiv.org/abs/2304.09184) | ranking | seq | RecBole | 3 | |
| MCLRec | Meta-optimized contrastive learning | 2023 | SIGIR | [arXiv:2304.07763](https://arxiv.org/abs/2304.07763) | ranking | seq | — | 3 | |
| AdaMCT | Adaptive mixture of CNN and Transformer | 2023 | CIKM | [arXiv:2205.08776](https://arxiv.org/abs/2205.08776) | ranking | seq | — | 3 | |
| gSASRec | SASRec with more negatives plus generalized BCE (calibrates overconfidence) | 2023 | RecSys | [arXiv:2308.07192](https://arxiv.org/abs/2308.07192) | ranking | seq | [Official](https://github.com/asash/gSASRec-pytorch) | 1 | **Implemented** |
| SASRec-CE | SASRec with full-softmax cross-entropy; beats BERT4Rec | 2023 | RecSys† | [arXiv:2309.07602](https://arxiv.org/abs/2309.07602) | ranking | seq | — | 1 | **Implemented** |
| DiffuRec | Diffusion over target-item embeddings | 2023 | TOIS† | [arXiv:2304.00686](https://arxiv.org/abs/2304.00686) | ranking | seq | [Official](https://github.com/WHUIR/DiffuRec) | 3 | |
| DreamRec | Guided diffusion generates an oracle next-item embedding | 2023 | NeurIPS | [arXiv:2310.20453](https://arxiv.org/abs/2310.20453) | ranking | seq | [Official](https://github.com/YangZhengyi98/DreamRec) | 3 | |
| BSARec | Self-attention plus a Fourier inductive-bias (beyond-attention) module | 2024 | AAAI | [arXiv:2312.10325](https://arxiv.org/abs/2312.10325) | ranking | seq | [Official](https://github.com/yehjin-shin/BSARec) | 2 | |
| LRURec | Linear recurrent units with parallel training and recurrent inference | 2024 | WSDM | [arXiv:2310.02367](https://arxiv.org/abs/2310.02367) | ranking | seq | [Official](https://github.com/yueqirex/LRURec) | 3 | |
| Mamba4Rec | Selective state-space (Mamba) blocks for sequential rec | 2024 | RelKD@KDD (workshop) | [arXiv:2403.03900](https://arxiv.org/abs/2403.03900) | ranking | seq | [Official](https://github.com/chengkai-liu/Mamba4Rec) | 2 | |
| SIGMA | Bidirectional gated Mamba with a partially flipped sequence | 2024 | arXiv† | [arXiv:2408.11451](https://arxiv.org/abs/2408.11451) | ranking | seq | — | 3 | |
| DARE | Decoupled attention/representation embeddings for long sequences | 2025 | ICLR | [arXiv:2410.02604](https://arxiv.org/abs/2410.02604) | ranking / ctr | seq | — | 3 | |

## 7. CTR prediction: feature-interaction models

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| LR (FTRL) | Logistic regression on one-hot features, FTRL-Proximal online | 2013 | KDD | [10.1145/2487575.2488200](https://doi.org/10.1145/2487575.2488200) | ctr | feat | FuxiCTR | 1 | **Implemented** |
| Poly2 | Degree-2 polynomial mapping with a linear model | 2010 | JMLR | [JMLR v11](https://www.jmlr.org/papers/v11/chang10a.html)† | ctr | feat | — | 3 | |
| FM | Factorized pairwise feature interactions | 2010 | ICDM | [10.1109/ICDM.2010.127](https://doi.org/10.1109/ICDM.2010.127) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| CCPM | CNN over field embeddings | 2015 | CIKM | [10.1145/2806416.2806603](https://doi.org/10.1145/2806416.2806603) | ctr | feat | FuxiCTR | 3 | |
| FNN | DNN initialized from FM embeddings | 2016 | ECIR | [arXiv:1601.02376](https://arxiv.org/abs/1601.02376) | ctr | feat | FuxiCTR, DeepCTR | 3 | |
| FFM | Field-aware latent vectors per field pair | 2016 | RecSys | [10.1145/2959100.2959134](https://doi.org/10.1145/2959100.2959134) | ctr | feat | FuxiCTR | 2 | |
| HOFM | Higher-order FM via ANOVA kernels | 2016 | NIPS | [arXiv:1607.07195](https://arxiv.org/abs/1607.07195) | ctr | feat | FuxiCTR | 3 | |
| PNN (IPNN/OPNN) | Inner/outer product layer before the MLP | 2016 | ICDM | [arXiv:1611.00144](https://arxiv.org/abs/1611.00144) | ctr | feat | FuxiCTR, DeepCTR | 2 | |
| Wide & Deep | Jointly trained linear (wide) plus DNN (deep) | 2016 | DLRS@RecSys | [arXiv:1606.07792](https://arxiv.org/abs/1606.07792) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| Deep Crossing | Residual MLP over concatenated embeddings | 2016 | KDD | [10.1145/2939672.2939704](https://doi.org/10.1145/2939672.2939704) | ctr | feat | FuxiCTR | 3 | |
| DeepFM | FM plus DNN sharing embeddings | 2017 | IJCAI | [arXiv:1703.04247](https://arxiv.org/abs/1703.04247) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| NFM | Bi-interaction pooling into an MLP | 2017 | SIGIR | [arXiv:1708.05027](https://arxiv.org/abs/1708.05027) | ctr | feat | FuxiCTR, DeepCTR | 2 | |
| AFM | Attention-weighted pairwise FM interactions | 2017 | IJCAI | [arXiv:1708.04617](https://arxiv.org/abs/1708.04617) | ctr | feat | FuxiCTR, DeepCTR | 2 | |
| DCN | Explicit bounded-degree cross network plus DNN | 2017 | ADKDD | [arXiv:1708.05123](https://arxiv.org/abs/1708.05123) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| FwFM | Field-pair weights on FM interactions | 2018 | WWW | [arXiv:1806.03514](https://arxiv.org/abs/1806.03514) | ctr | feat | FuxiCTR | 3 | |
| xDeepFM | Compressed interaction network (vector-wise explicit crosses) plus DNN | 2018 | KDD | [arXiv:1803.05170](https://arxiv.org/abs/1803.05170) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| AutoInt / AutoInt+ | Multi-head self-attention over field embeddings | 2019 | CIKM | [arXiv:1810.11921](https://arxiv.org/abs/1810.11921) | ctr | feat | FuxiCTR, DeepCTR | 1 | **Implemented** |
| FiBiNET | SENet field importance plus bilinear interaction | 2019 | RecSys | [arXiv:1905.09433](https://arxiv.org/abs/1905.09433) | ctr | feat | FuxiCTR, DeepCTR | 2 | |
| FGCNN | CNN-generated new features plus IPNN | 2019 | WWW | [arXiv:1904.04447](https://arxiv.org/abs/1904.04447) | ctr | feat | FuxiCTR | 3 | |
| FiGNN | Fields as graph nodes with gated GNN interactions | 2019 | CIKM | [arXiv:1910.05552](https://arxiv.org/abs/1910.05552) | ctr | feat | FuxiCTR | 3 | |
| ONN (NFFM) | Operation-aware (field-aware) embeddings plus DNN | 2019 | arXiv (Neural Networks 2020) | [arXiv:1904.12579](https://arxiv.org/abs/1904.12579) | ctr | feat | FuxiCTR | 3 | |
| DLRM | Meta's embedding-tables plus dot-interaction MLP reference model | 2019 | arXiv | [arXiv:1906.00091](https://arxiv.org/abs/1906.00091) | ctr | feat | [Official](https://github.com/facebookresearch/dlrm), FuxiCTR | 2 | |
| AFN / AFN+ | Logarithmic neurons learn adaptive-order interactions | 2020 | AAAI | [arXiv:1909.03276](https://arxiv.org/abs/1909.03276) | ctr | feat | FuxiCTR | 3 | |
| InterHAt | Hierarchical attention for interpretable high-order interactions | 2020 | WSDM | [10.1145/3336191.3371785](https://doi.org/10.1145/3336191.3371785) | ctr | feat | FuxiCTR | 3 | |
| DCN-V2 | Matrix-weight (optionally low-rank MoE) cross layers | 2021 | WWW | [arXiv:2008.13535](https://arxiv.org/abs/2008.13535) | ctr | feat | FuxiCTR, [TF-Recommenders](https://github.com/tensorflow/recommenders) | 1 | **Implemented** |
| FmFM | Field-matrixed FM (generalizes FwFM/FvFM) | 2021 | WWW | [arXiv:2102.12994](https://arxiv.org/abs/2102.12994) | ctr | feat | FuxiCTR | 3 | |
| MaskNet | Instance-guided masks with feature-wise multiplication | 2021 | DLP-KDD | [arXiv:2102.07619](https://arxiv.org/abs/2102.07619) | ctr | feat | FuxiCTR | 2 | |
| DESTINE | Disentangled self-attention for interactions | 2021 | CIKM | [arXiv:2101.03654](https://arxiv.org/abs/2101.03654) | ctr | feat | FuxiCTR | 3 | |
| EDCN | Bridge and regulation modules sharing info between parallel towers | 2021 | CIKM | [10.1145/3459637.3481915](https://doi.org/10.1145/3459637.3481915) | ctr | feat | FuxiCTR | 3 | |
| SAM | Self-attention variants for CTR, revisited | 2021 | SIGIR | [10.1145/3404835.3462936](https://doi.org/10.1145/3404835.3462936) | ctr | feat | FuxiCTR | 3 | |
| AOANet | Architecture- and operation-adaptive interaction network | 2021 | KDD | [10.1145/3447548.3467133](https://doi.org/10.1145/3447548.3467133) | ctr | feat | FuxiCTR | 3 | |
| DHEN | Hierarchical ensemble of heterogeneous interaction modules | 2022 | arXiv | [arXiv:2203.11014](https://arxiv.org/abs/2203.11014) | ctr | feat | — | 3 | |
| FiBiNet++ | Low-rank bilinear FiBiNET with a smaller model | 2022 | CIKM† | [arXiv:2209.05016](https://arxiv.org/abs/2209.05016) | ctr | feat | — | 3 | |
| APG | Adaptive per-instance parameter generation | 2022 | NeurIPS | [arXiv:2203.16218](https://arxiv.org/abs/2203.16218) | ctr | feat | FuxiCTR | 3 | |
| FinalMLP | Two MLP streams with feature gating plus bilinear fusion | 2023 | AAAI | [arXiv:2304.00902](https://arxiv.org/abs/2304.00902) | ctr | feat | FuxiCTR (official) | 1 | **Implemented** |
| FinalNet | Factorized interaction layer (FINAL) blocks | 2023 | SIGIR | [10.1145/3539618.3591988](https://doi.org/10.1145/3539618.3591988) | ctr | feat | FuxiCTR (official) | 2 | |
| EulerNet | Interactions in complex space via Euler's formula | 2023 | SIGIR | [arXiv:2304.10711](https://arxiv.org/abs/2304.10711) | ctr | feat | FuxiCTR, RecBole | 3 | |
| GDCN | Gated cross network plus field-level dimension optimization | 2023 | CIKM | [10.1145/3583780.3615089](https://doi.org/10.1145/3583780.3615089) ([arXiv:2311.04635](https://arxiv.org/abs/2311.04635)) | ctr | feat | [Official](https://github.com/anonctr/GDCN), FuxiCTR | 2 | |
| Wukong | Stacked FM blocks designed to follow a scaling law | 2024 | ICML | [arXiv:2403.02545](https://arxiv.org/abs/2403.02545) | ctr | feat | FuxiCTR | 3 | |
| FCN (formerly DCNv3) | Fused exponential plus linear cross networks | 2026 | KDD | [arXiv:2407.13349](https://arxiv.org/abs/2407.13349) | ctr | feat | — | 3 | |
| RankMixer | Token-mixing plus per-token (sparse-MoE) FFN; GPU-efficient scaled ranker | 2025 | arXiv | [arXiv:2507.15551](https://arxiv.org/abs/2507.15551) | ctr | feat, seq | — | 3 | |

## 8. CTR prediction: user-behavior (sequence-aware) models

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| DIN | Target attention over the behavior sequence | 2018 | KDD | [arXiv:1706.06978](https://arxiv.org/abs/1706.06978) | ctr (+ sampled ranking) | feat, seq | FuxiCTR, DeepCTR | 1 | **Implemented** |
| DIEN | GRU interest extractor plus attention-update GRU (AUGRU) | 2019 | AAAI | [arXiv:1809.03672](https://arxiv.org/abs/1809.03672) | ctr | feat, seq | FuxiCTR, DeepCTR | 1 | **Implemented** |
| DSIN | Session-split behaviors with self-attention plus Bi-LSTM | 2019 | IJCAI | [arXiv:1905.06482](https://arxiv.org/abs/1905.06482) | ctr | feat, seq | DeepCTR | 2 | |
| BST | Transformer over behaviors plus the target item | 2019 | DLP-KDD | [arXiv:1905.06874](https://arxiv.org/abs/1905.06874) | ctr | feat, seq | FuxiCTR, DeepCTR | 2 | |
| MIMN | Memory-network user interest center (UIC) for long sequences | 2019 | KDD | [arXiv:1905.09248](https://arxiv.org/abs/1905.09248) | ctr | feat, seq | — | 3 | |
| DMR | Deep match-to-rank: a U2I relevance subnet inside ranking | 2020 | AAAI | [10.1609/aaai.v34i01.5346](https://doi.org/10.1609/aaai.v34i01.5346) | ctr | feat, seq | FuxiCTR | 3 | |
| DMIN | Multi-head deep multi-interest network | 2020 | CIKM | [10.1145/3340531.3412092](https://doi.org/10.1145/3340531.3412092) | ctr | feat, seq | FuxiCTR | 3 | |
| DHAN | Hierarchical attention over attribute-level interests | 2020 | SIGIR | [arXiv:2005.12981](https://arxiv.org/abs/2005.12981) | ctr | feat, seq | — | 3 | |
| UBR4CTR | Retrieve relevant behaviors with a search engine, then attend | 2020 | SIGIR | [arXiv:2005.14171](https://arxiv.org/abs/2005.14171) | ctr | feat, seq | — | 3 | |
| SIM | Two-stage general search unit (GSU) plus exact search unit (ESU) for lifelong sequences | 2020 | CIKM | [arXiv:2006.05639](https://arxiv.org/abs/2006.05639) | ctr | feat, seq | FuxiCTR | 2 | |
| CAN | Feature co-action units (micro-MLPs) | 2022 | WSDM | [arXiv:2011.05625](https://arxiv.org/abs/2011.05625) | ctr | feat, seq | — | 3 | |
| ETA | LSH (SimHash) retrieval with end-to-end target attention | 2022 | DLP-KDD | [arXiv:2209.12212](https://arxiv.org/abs/2209.12212) | ctr | feat, seq | FuxiCTR | 2 | |
| SDIM | Hash-based sampling replaces top-k retrieval for long sequences | 2022 | CIKM | [arXiv:2205.10249](https://arxiv.org/abs/2205.10249) | ctr | feat, seq | FuxiCTR | 2 | |
| TWIN | Consistent GSU/ESU relevance metric for lifelong behavior | 2023 | KDD | [arXiv:2302.02352](https://arxiv.org/abs/2302.02352) | ctr | feat, seq | FuxiCTR | 2 | |
| TransAct | Real-time action Transformer plus batch user embedding (Pinterest) | 2023 | KDD | [arXiv:2306.00248](https://arxiv.org/abs/2306.00248) | ctr | feat, seq | FuxiCTR | 3 | |
| PEPNet | Parameter and embedding personalization gates (PPNet + EPNet) | 2023 | KDD | [arXiv:2302.01115](https://arxiv.org/abs/2302.01115) | ctr (multi-task) | feat, seq | FuxiCTR (PPNet) | 3 | |
| TWIN V2 | Hierarchical clustering compresses ultra-long sequences | 2024 | CIKM | [arXiv:2407.16357](https://arxiv.org/abs/2407.16357) | ctr | feat, seq | — | 3 | |
| LONGER | GPU-efficient long-sequence Transformer (global tokens, token merge) | 2025 | RecSys | [arXiv:2505.04421](https://arxiv.org/abs/2505.04421) | ctr | feat, seq | — | 3 | |
| MIRRN | Multi-granularity interest retrieval and refinement | 2025 | KDD† | listed in [FuxiCTR](https://github.com/reczoo/FuxiCTR) model zoo | ctr | feat, seq | FuxiCTR | 3 | |

## 9. Two-tower and embedding-based retrieval

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| DSSM | Two-tower deep semantic matching with softmax over clicks | 2013 | CIKM | [10.1145/2505515.2505665](https://doi.org/10.1145/2505515.2505665) | ranking | feat / text | FuxiCTR, RecBole | 1 | **Implemented** |
| Prod2Vec | word2vec over purchase sequences | 2015 | KDD | [arXiv:1606.07154](https://arxiv.org/abs/1606.07154) | ranking | seq | — | 3 | |
| YouTube DNN | Averaged watch-history embeddings with sampled-softmax candidate generation | 2016 | RecSys | [10.1145/2959100.2959190](https://doi.org/10.1145/2959100.2959190) | ranking | seq, feat | DeepMatch-style impls | 1 | **Implemented** |
| TDM | Tree-based deep model with beam search over an item tree | 2018 | KDD | [arXiv:1801.02294](https://arxiv.org/abs/1801.02294) | ranking | seq, feat | — | 3 | |
| Sampling-bias-corrected two-tower | In-batch softmax with streaming frequency (logQ) correction | 2019 | RecSys | [10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996) | ranking | feat, seq | TF-Recommenders | 1 | **Implemented** |
| MIND | Capsule dynamic routing yields multiple user-interest vectors | 2019 | CIKM | [arXiv:1904.08030](https://arxiv.org/abs/1904.08030) | ranking | seq, feat | — | 2 | |
| ComiRec | Controllable multi-interest (dynamic routing or self-attention) plus diversity aggregation | 2020 | KDD | [arXiv:2005.09347](https://arxiv.org/abs/2005.09347) | ranking | seq | [Official](https://github.com/THUDM/ComiRec) | 2 | |
| Mixed Negative Sampling | In-batch plus uniformly sampled negatives for two-tower | 2020 | WWW (companion) | [10.1145/3366424.3386195](https://doi.org/10.1145/3366424.3386195) | ranking | feat | — | 2 | |
| EBR (Facebook) | Unified embedding-based retrieval with hard-negative mining | 2020 | KDD | [arXiv:2006.11632](https://arxiv.org/abs/2006.11632) | ranking | feat, text | — | 3 | |
| PinnerSage | Multi-modal (clustered) user embeddings | 2020 | KDD | [arXiv:2007.03634](https://arxiv.org/abs/2007.03634) | ranking | seq, mm | — | 3 | |
| Deep Retrieval | Learnable discrete path structure for retrieval | 2020 | arXiv | [arXiv:2007.07203](https://arxiv.org/abs/2007.07203) | ranking | seq | — | 3 | |
| Cross-Batch Negative Sampling | Reuse embeddings from recent batches as negatives | 2021 | SIGIR | [arXiv:2110.15154](https://arxiv.org/abs/2110.15154) | ranking | feat | — | 3 | |
| PinnerFormer | Transformer user model trained with a dense all-action loss | 2022 | KDD | [arXiv:2205.04507](https://arxiv.org/abs/2205.04507) | ranking | seq, feat | — | 3 | |

## 10. Multi-task and multi-domain ranking (out of scope for now)

The library's tasks have a single label, so these are tracked but not scheduled.

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| Shared-Bottom | Hard parameter sharing with per-task towers | 1997 | Machine Learning | [10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734) | multi-task ctr (out of scope) | feat | FuxiCTR, DeepCTR | 2 | |
| ESMM | Model CTR and CTCVR over the entire space to fix CVR sample bias | 2018 | SIGIR | [arXiv:1804.07931](https://arxiv.org/abs/1804.07931) | multi-task ctr (out of scope) | feat | DeepCTR | 2 | |
| MMoE | Multi-gate mixture of experts | 2018 | KDD | [10.1145/3219819.3220007](https://doi.org/10.1145/3219819.3220007) | multi-task ctr (out of scope) | feat | FuxiCTR, DeepCTR | 2 | |
| PLE | Progressive layered extraction with shared and task-specific experts | 2020 | RecSys | [10.1145/3383313.3412236](https://doi.org/10.1145/3383313.3412236) | multi-task ctr (out of scope) | feat | FuxiCTR, DeepCTR | 2 | |
| AITM | Adaptive information transfer across a sequential conversion funnel | 2021 | KDD | [arXiv:2105.08489](https://arxiv.org/abs/2105.08489) | multi-task ctr (out of scope) | feat | — | 3 | |
| STAR | Star-topology multi-domain CTR model | 2021 | CIKM | [arXiv:2101.11427](https://arxiv.org/abs/2101.11427) | multi-domain ctr (out of scope) | feat | — | 3 | |
| ESCM² | Counterfactual-regularized ESMM | 2022 | SIGIR | [arXiv:2204.05125](https://arxiv.org/abs/2204.05125) | multi-task ctr (out of scope) | feat | — | 3 | |

## 11. Generative recommendation (semantic IDs, scaled sequential transducers)

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| DSI | Transformer memory as a differentiable search index (docid generation; ancestor of semantic IDs) | 2022 | NeurIPS | [arXiv:2202.06991](https://arxiv.org/abs/2202.06991) | ranking | text | — | 3 | |
| TIGER | RQ-VAE semantic IDs plus a seq2seq Transformer that decodes the next item's ID | 2023 | NeurIPS | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065) | ranking | seq, text | — (several unofficial ports) | 1 | |
| How to Index Item IDs | Compares random, title, sequential, and collaborative IDs for P5-style models | 2023 | SIGIR-AP | [arXiv:2305.06569](https://arxiv.org/abs/2305.06569) | ranking | seq, text | [OpenP5](https://github.com/agiresearch/OpenP5) | 3 | |
| HSTU | Hierarchical sequential transduction units; trillion-parameter generative recommender ("Actions Speak Louder than Words") | 2024 | ICML | [arXiv:2402.17152](https://arxiv.org/abs/2402.17152) | ranking (and ctr) | seq, feat | [Official](https://github.com/facebookresearch/generative-recommenders) | 1 | |
| LETTER | Learnable tokenizer: RQ-VAE plus collaborative and diversity regularizers | 2024 | CIKM | [arXiv:2405.07314](https://arxiv.org/abs/2405.07314) | ranking | seq, text | — | 2 | |
| EAGER | Two-stream (behavior plus semantic codes) generative recommender | 2024 | KDD | [arXiv:2406.14017](https://arxiv.org/abs/2406.14017) | ranking | seq, text | [Official (reczoo)](https://reczoo.github.io/EAGER) | 2 | |
| LIGER | Hybrid: semantic-ID generation followed by dense re-ranking | 2025 | TMLR | [arXiv:2411.18814](https://arxiv.org/abs/2411.18814) | ranking | seq, text | — | 2 | |
| FuXi-α | HSTU-style Transformer with explicit feature-interaction (temporal/positional) channels | 2025 | WWW | [arXiv:2502.03036](https://arxiv.org/abs/2502.03036) | ranking | seq | — | 3 | |
| ActionPiece | Context-aware BPE-like tokenization of action sequences | 2025 | ICML | [arXiv:2502.13581](https://arxiv.org/abs/2502.13581) | ranking | seq, feat | — | 2 | |
| OneRec | End-to-end encoder-decoder generative recommender with iterative DPO (Kuaishou) | 2025 | arXiv | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965) | ranking | seq, feat, mm | — | 3 | |
| COBRA | Cascaded sparse (semantic ID) then dense vector generation with BeamFusion | 2025 | NeurIPS | [arXiv:2503.02453](https://arxiv.org/abs/2503.02453) | ranking | seq, text | — | 3 | |
| MTGR | Generative (HSTU-style) ranking that keeps DLRM cross features (Meituan) | 2025 | arXiv | [arXiv:2505.18654](https://arxiv.org/abs/2505.18654) | ctr | seq, feat | — | 3 | |
| OneRec Technical Report | Scaled OneRec with a reward system and RL; about 25% of Kuaishou QPS | 2025 | arXiv | [arXiv:2506.13695](https://arxiv.org/abs/2506.13695) | ranking | seq, feat, mm | — | 3 | |
| GRID (Semantic-ID handbook) | Modular open framework benchmarking semantic-ID GR components | 2025 | CIKM† | [arXiv:2507.22224](https://arxiv.org/abs/2507.22224) | ranking | seq, text | [Official (Snap)](https://github.com/snap-research/GRID)† | 2 | |
| OneRec-V2 | Lazy decoder-only architecture scaled to 8B plus real-feedback preference alignment | 2025 | arXiv | [arXiv:2508.20900](https://arxiv.org/abs/2508.20900) | ranking | seq, feat, mm | — | 3 | |
| PLUM | Semantic IDs plus continued pretraining of a Gemini-class LLM for YouTube retrieval | 2026 | WWW | [arXiv:2510.07784](https://arxiv.org/abs/2510.07784) | ranking | seq, text, mm | — | 3 | |
| OneRec-Think | In-text reasoning (itemic alignment plus RL) for generative recommendation | 2026 | ACL | [arXiv:2510.11639](https://arxiv.org/abs/2510.11639) | ranking | seq, text | — | 3 | |

## 12. LLM-based and text-transferable recommenders

| Method | Idea | Year | Venue | Paper | Task | Inputs | Ref. impl | Tier | Status |
|---|---|---|---|---|---|---|---|---|---|
| UniSRec | Text-based universal sequence representations with MoE adapters | 2022 | KDD | [arXiv:2206.05941](https://arxiv.org/abs/2206.05941) | ranking | seq, text | [Official](https://github.com/RUCAIBox/UniSRec), RecBole | 2 | |
| M6-Rec | Generative pretrained LM as an open-ended recommender | 2022 | arXiv | [arXiv:2205.08084](https://arxiv.org/abs/2205.08084) | ranking / ctr | seq, text | — | 3 | |
| P5 | Unified text-to-text (T5) pretraining over five recommendation task families | 2022 | RecSys | [arXiv:2203.13366](https://arxiv.org/abs/2203.13366) | ranking | seq, text | [Official](https://github.com/jeykigung/P5) | 2 | |
| VQ-Rec | Vector-quantized text codes for transferable sequential rec | 2023 | WWW | [arXiv:2210.12316](https://arxiv.org/abs/2210.12316) | ranking | seq, text | [Official](https://github.com/RUCAIBox/VQ-Rec) | 3 | |
| Chat-REC | Prompt ChatGPT with conversational, candidate-augmented context | 2023 | arXiv | [arXiv:2303.14524](https://arxiv.org/abs/2303.14524) | ranking | seq, text | — | 3 | |
| TALLRec | LoRA instruction tuning of LLaMA for yes/no preference prediction | 2023 | RecSys | [arXiv:2305.00447](https://arxiv.org/abs/2305.00447) | ctr (binary) | seq, text | [Official](https://github.com/SAI990323/TALLRec) | 2 | |
| InstructRec | Recommendation as instruction following (Flan-T5) | 2023 | arXiv (TOIS)† | [arXiv:2305.07001](https://arxiv.org/abs/2305.07001) | ranking | seq, text | — | 3 | |
| Recformer | Items as key-value text; Longformer-based, ID-free | 2023 | KDD | [arXiv:2305.13731](https://arxiv.org/abs/2305.13731) | ranking | seq, text | [Official](https://github.com/AaronHeee/RecFormer) | 2 | |
| VIP5 | Multimodal P5 (vision plus text) with adapters | 2023 | EMNLP (Findings)† | [arXiv:2305.14302](https://arxiv.org/abs/2305.14302) | ranking | seq, text, mm | — | 3 | |
| GenRec | Fine-tuned LLaMA generates the next item's title | 2023 | arXiv (ECIR 2024 workshop)† | [arXiv:2307.00457](https://arxiv.org/abs/2307.00457) | ranking | seq, text | [Official](https://github.com/rutgerswiselab/GenRec) | 3 | |
| BIGRec | Bi-step grounding: LLM generates text, which is grounded to real items | 2023 | arXiv (TORS)† | [arXiv:2308.08434](https://arxiv.org/abs/2308.08434) | ranking | seq, text | [Official](https://github.com/SAI990323/BIGRec) | 2 | |
| LlamaRec | Two-stage: LRURec retrieval, then LLM verbalizer ranking | 2023 | PGAI@CIKM | [arXiv:2311.02089](https://arxiv.org/abs/2311.02089) | ranking | seq, text | [Official](https://github.com/Yueeeeeeee/LlamaRec) | 3 | |
| E4SRec | ID embeddings injected into an LLM with a linear prediction head | 2023 | arXiv | [arXiv:2312.02443](https://arxiv.org/abs/2312.02443) | ranking | seq | — | 3 | |
| RecRanker | Instruction-tuned LLM as a top-k ranker (pointwise, pairwise, listwise) | 2023 | arXiv (TOIS)† | [arXiv:2312.16018](https://arxiv.org/abs/2312.16018) | ranking | seq, text | — | 3 | |
| LLMRank | Zero-shot LLM ranking over candidate lists | 2024 | ECIR | [arXiv:2305.08845](https://arxiv.org/abs/2305.08845) | ranking | seq, text | [Official](https://github.com/RUCAIBox/LLMRank) | 2 | |
| LC-Rec | Learned VQ item indices plus alignment tuning of LLaMA | 2024 | ICDE | [arXiv:2311.09049](https://arxiv.org/abs/2311.09049) | ranking | seq, text | [Official](https://github.com/RUCAIBox/LC-Rec) | 2 | |
| LLaRA | Hybrid ID-plus-text item tokens with curriculum tuning | 2024 | SIGIR | [arXiv:2312.02445](https://arxiv.org/abs/2312.02445) | ranking | seq, text | [Official](https://github.com/ljy0ustc/LLaRA) | 2 | |
| IDGenRec | LLM-generated textual IDs jointly learned with the recommender | 2024 | SIGIR | [arXiv:2403.19021](https://arxiv.org/abs/2403.19021) | ranking | seq, text | — | 3 | |
| OpenP5 | Open platform with P5-style models and ID-indexing strategies | 2024 | SIGIR (resource) | [arXiv:2306.11134](https://arxiv.org/abs/2306.11134) | ranking | seq, text | [Official](https://github.com/agiresearch/OpenP5) | 3 | |
| CALRec | Contrastive alignment of generative LLMs (PaLM-2) for sequential rec | 2024 | RecSys | [arXiv:2405.02429](https://arxiv.org/abs/2405.02429) | ranking | seq, text | — | 3 | |
| KAR | LLM-generated knowledge augmentation fed to CTR models | 2024 | RecSys† | [arXiv:2306.10933](https://arxiv.org/abs/2306.10933) | ctr | feat, text | — | 3 | |
| ReLLa | Retrieval-enhanced LLM for lifelong behavior CTR | 2024 | WWW | [arXiv:2308.11131](https://arxiv.org/abs/2308.11131) | ctr | seq, text | — | 3 | |
| ClickPrompt | CTR model generates soft prompts for an LM | 2024 | WWW | [arXiv:2310.09234](https://arxiv.org/abs/2310.09234) | ctr | feat, text | — | 3 | |
| A-LLMRec | Align a frozen CF model's embeddings with an LLM | 2024 | KDD | [arXiv:2404.11343](https://arxiv.org/abs/2404.11343) | ranking | seq, text | [Official](https://github.com/ghdtjr/A-LLMRec) | 3 | |
| BinLLM | Collaborative embeddings binarized into text-like tokens | 2024 | ACL | [arXiv:2406.03210](https://arxiv.org/abs/2406.03210) | ctr | inter, text | — | 3 | |
| HLLM | Hierarchical item LLM plus user LLM | 2024 | arXiv | [arXiv:2409.12740](https://arxiv.org/abs/2409.12740) | ranking | seq, text | [Official](https://github.com/bytedance/HLLM) | 2 | |
| LLM-ESR | LLM semantic embeddings for long-tail sequential rec (dual-view) | 2024 | NeurIPS | [arXiv:2405.20646](https://arxiv.org/abs/2405.20646) | ranking | seq, text | — | 3 | |
| CoLLM | Collaborative embeddings mapped into the LLM token space | 2025 | TKDE | [arXiv:2310.19488](https://arxiv.org/abs/2310.19488) | ctr | inter, text | [Official](https://github.com/zyang1580/CoLLM) | 3 | |
| TokenRec | Masked vector-quantized tokenizer for IDs plus generative retrieval | 2025 | TKDE | [arXiv:2406.10450](https://arxiv.org/abs/2406.10450) | ranking | inter, text | — | 3 | |
| AlphaRec | Linear map from LLM text embeddings plus graph conv and contrastive loss | 2025 | ICLR | [arXiv:2407.05441](https://arxiv.org/abs/2407.05441) | ranking | inter, text | [Official](https://github.com/LehengTHU/AlphaRec) | 2 | |
| LLMEmb | Fine-tuned LLM as an item-embedding generator for SR | 2025 | AAAI | [arXiv:2409.19925](https://arxiv.org/abs/2409.19925) | ranking | seq, text | — | 3 | |
| RecGPT (Taobao) | LLM-driven intent-centric recommendation pipeline | 2025 | arXiv | [arXiv:2507.22879](https://arxiv.org/abs/2507.22879) | ranking | seq, text | — | 3 | |

---

## Suggested implementation order

These are the next 15, chosen to close the Tier 1 gaps first. They favor methods that are cheap to build on the existing classical and torch paths and that the "are we really making progress" line of work shows to be strong baselines.

1. **ItemKNN**: the most common neighborhood baseline. It is classical-path only and shares a similarity kernel with UserKNN.
2. **UserKNN**: comes almost for free once ItemKNN exists.
3. **RP3beta**: a strong, cheap graph-walk baseline (Dacrema et al.). Classical path.
4. **SLIM (ElasticNet)**: the canonical linear item-item model and the natural companion to EASE.
5. **iALS / WMF**: the standard implicit MF baseline. Tuned iALS is competitive with neural models (Rendle 2022). It can wrap `implicit` or use a pure NumPy ALS.
6. **PureSVD**: a one-line truncated SVD that is often surprisingly strong. It completes the Dacrema baseline set.
7. **Mult-VAE**: the reference VAE baseline. Its torch user-vector training is reusable for RecVAE and CDAE.
8. **NeuMF**: still expected by reviewers and a frequent "does neural beat MF" comparison point.
9. **GRU4Rec**: the canonical RNN sequential/session baseline. It reuses the SASRec data pipeline.
10. **BERT4Rec**: the canonical bidirectional baseline. Implement it together with **gSASRec / SASRec-CE** loss options so the library can reproduce the "SASRec vs BERT4Rec" findings.
11. **SR-GNN**: the standard session-graph baseline. It also validates session-based splits.
12. **DCN-V2**: the strongest classic explicit-interaction CTR model in BARS/FuxiCTR results. It shares the embedding and MLP code with DeepFM.
13. **xDeepFM** and **AutoInt+**: ubiquitous CTR comparison points that are cheap given the shared CTR backbone.
14. **FinalMLP**: a simple, near-SOTA modern CTR baseline in FuxiCTR benchmarks.
15. **TIGER (with RQ-VAE semantic IDs)**: the entry point to generative recommendation. Its RQ-VAE tokenizer can then be reused for LETTER, LC-Rec, and LIGER. **HSTU** should follow as the second generative baseline.

Close runners-up: SimGCL/XSimGCL (strong graph CF next to LightGCN), CL4SRec/DuoRec (contrastive sequential), Wide & Deep and FM (historical CTR anchors), SIM/ETA (long-sequence CTR, once the benchmarks carry long behavior histories), and the two-tower YouTube DNN and sampling-bias-corrected model (if a retrieval task is added).

# Graclus clustering stats (ML-1M and Alibaba-iFashion)

Structure-only statistics. Nothing here comes from training, so none of it says which K gives better recall.

## 1. Where things are

| What | Where | Made by |
|---|---|---|
| Graph given to Graclus (train split) | `outputs/graclus/<dataset>_train.graph` and `.map.npz` | `python scripts/prep_graclus.py` |
| Graclus output, one cluster id per node per line | `graclus1.2/<dataset>_train.graph.part.<K>` | `./graclus ../outputs/graclus/<graph> <K>` (WSL, inside `graclus1.2/`) |
| Co-occurrence stats (full data, not train split) | `outputs/graph_viz/{ml1m,ifashion}/stats.json` and 5 PNGs each | `python scripts/graph_viz.py --part <part file>` |

```bash
cd /mnt/c/dev_stuff/MAGCL_claude/graclus1.2
./graclus ../outputs/graclus/ml-1m_preprocessed_byrating_train.graph 16
./graclus ../outputs/graclus/Alibaba-iFashion_train.graph 64
# options: -o ncut|rassoc   -l <local search steps>   -b (boundary points only)   -e <partfile> <graph> (evaluate)
```

Graclus prints a summary and does not save it. Example, ML-1M K=16:
`#Vertices 9639, #Edges 674674, #Clusters 16`, `Cut value 13.279723, Balance 1.78`, `Total 1.231 s`.
The cut value is the normalized cut (worst case = K, every edge cut); balance is largest cluster / ideal size.

## 2. Clustering stats on the TRAIN split (what MAGCL uses)

Nodes = users + items in one bipartite graph; each node gets one cluster id.

**ML-1M**: 6,039 users, 3,600 items, 674,674 train edges, mean user degree 111.7, mean item degree 187.4.

| K | Cluster size min / median / max | Size CV | Users / items per cluster (median) | Edges inside | Random | Lift | Largest cluster's edge share | Item-popularity eta^2 | User-activity eta^2 |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2093 / 2455 / 2636 | 0.09 | 1544 / 935 | 45.7% | 25.4% | 1.8 | 27.3% | 0.02 | 0.00 |
| 8 | 592 / 1177 / 1820 | 0.27 | 778 / 433 | 30.9% | 13.5% | 2.3 | 18.3% | 0.05 | 0.11 |
| 16 | 285 / 573 / 1073 | 0.32 | 329 / 221 | 16.4% | 6.0% | 2.7 | 11.8% | 0.10 | 0.10 |
| 32 | 146 / 230 / 918 | 0.66 | 137 / 95 | 9.0% | 3.7% | 2.4 | 19.8% | 0.26 | 0.16 |
| 64 | 72 / 131 / 458 | 0.51 | 71 / 52 | 5.1% | 1.4% | 3.7 | 7.1% | 0.23 | 0.15 |

**Alibaba-iFashion**: 300,000 users, 74,380 items, 1,023,049 train edges, mean user degree 3.4, mean item degree 13.8.
The train split holds only 63.6% of the 1,607,813 interactions, not 80%: RecBole's per-user 8:1:1 split gives every user with 3 or more
interactions at least one validation and one test item, and 90% of iFashion users have fewer than 10 (median 4). Applying RecBole's rule to the
real degrees gives train / valid / test = 1,023,049 / 281,817 / 302,946, which matches the measured train size. ML-1M users have a median of 81
interactions, so its train split is 80.7%.

| K | Cluster size min / median / max | Size CV | Users / items per cluster (median) | Edges inside | Random | Lift | Largest cluster's edge share | Item-popularity eta^2 | User-activity eta^2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4382 / 24667 / 32897 | 0.33 | 18973 / 5492 | 44.1% | 6.2% | 7.2 | 8.0% | 0.03 | 0.02 |
| 32 | 1963 / 12364 / 21159 | 0.37 | 9607 / 2433 | 38.4% | 6.3% | 6.1 | 21.9% | 0.08 | 0.10 |
| 64 | 923 / 5898 / 24887 | 0.53 | 4667 / 1255 | 40.5% | 2.9% | 13.8 | 13.4% | 0.07 | 0.14 |
| 128 | 343 / 2877 / 27798 | 0.84 | 2251 / 623 | 37.3% | 4.1% | 9.0 | 19.2% | 0.07 | 0.20 |
| 256 | 10 / 1567 / 10118 | 0.51 | 1252 / 314 | 29.1% | 1.4% | 20.9 | 10.6% | 0.08 | 0.17 |

Column meanings:
- **Edges inside**: share of train interactions whose user and item are in the same cluster. The rest are "cut".
- **Random**: the same share if endpoints were matched at random with the same cluster volumes. **Lift** = inside / random.
- **Size CV**: standard deviation of cluster size / mean size (0 = perfectly balanced).
- **Largest cluster's edge share**: the biggest cluster's share of all edge endpoints.
- **eta^2**: share of the variance of log degree that the cluster label explains (0 = clusters ignore it, 1 = clusters are degree bins).
- Every node has at least 5 same-type cluster-mates at every K, so MAGCL's top-S (S=5) rule never starves.

## 3. Co-occurrence stats (FULL data, from `outputs/graph_viz/*/stats.json`)

Not built from the RecBole train split, so they only describe the data.
- **ML-1M:** all 836,478 interactions (the paper's Table 1 count), K=8 (`ml1m.graph.part.8`).
- **iFashion:** only the pre-split `data/Alibaba-iFashion/ifashion.train` file, 1,255,447 of the 1,607,813 interactions (the paper's Table 1 count);
  the 352,366 interactions in `ifashion.test` are not included (`scripts/to_graclus.py` reads `ifashion.train`). K=64 (`ifashion.graph.part.64`).

| Dataset facts | ML-1M | iFashion |
|---|---|---|
| Users / items | 6,039 / 3,628 | 300,000 / 81,614 |
| Interactions | 836,478 | 1,255,447 |
| Density | 0.0382 | 5.13e-5 |
| User degree mean / median / max | 138.51 / 81 / 1968 | 4.18 / 3 / 342 |
| Item degree mean / median / max | 230.56 / 87 / 3211 | 15.38 / 4 / 2239 |
| Clusters (K) | 8 | 64 |
| Interactions inside a cluster | 31.47% | 37.84% |

**User-user pairs**

| Stat | ML-1M | iFashion |
|---|---|---|
| Rows evaluated | 6,039 (all) | 10,000 (sample of 300,000) |
| Fraction of pairs with any overlap | 0.9401 | 0.002357 |
| Mean overlap, same cluster | 25.1190 | 0.019802 |
| Mean overlap, different cluster | 16.5140 | 0.001943 |
| Ratio same / different | 1.52x | 10.19x |
| Ordered pairs with Jaccard >= 0.8 | 0 | 875 |
| Nodes with a partner at Jaccard >= 0.8 | 0.0% | 3.06% |
| Nodes with no overlap with anyone | 0 | 19 |
| Each node's best Jaccard, percentiles 5/25/50/75/95 | 0.159 / 0.212 / 0.251 / 0.302 / 0.380 | 0.143 / 0.250 / 0.333 / 0.500 / 0.667 |

**Item-item pairs**

| Stat | ML-1M | iFashion |
|---|---|---|
| Rows evaluated | 3,628 (all) | 10,000 (sample of 81,614) |
| Fraction of pairs with any overlap | 0.7664 | 0.001321 |
| Mean overlap, same cluster | 26.3671 | 0.006817 |
| Mean overlap, different cluster | 18.8379 | 0.001329 |
| Ratio same / different | 1.40x | 5.13x |
| Ordered pairs with Jaccard >= 0.8 | 140 (= 70 unordered) | 764 |
| Nodes with a partner at Jaccard >= 0.8 | 0.99% | 4.98% |
| Nodes with no overlap with anyone | 0 | 32 |
| Each node's best Jaccard, percentiles 5/25/50/75/95 | 0.100 / 0.147 / 0.200 / 0.284 / 0.483 | 0.020 / 0.062 / 0.143 / 0.250 / 0.500 |

How these are computed (`scripts/graph_viz.py`, `side_stats`):
- **Overlap** of two nodes = number of shared neighbours (users: shared items; items: shared users).
- **Jaccard** = overlap / (deg(x) + deg(y) - overlap).
- **Mean overlap, same / different cluster** is averaged over ordered pairs (x, y), x != y, and includes pairs with zero overlap.
- **Pairs with Jaccard >= 0.8** counts ordered pairs (x, y) with overlap > 0, where x is an evaluated row. On ML-1M every row is evaluated, so each unordered pair is counted twice. On iFashion only 10,000 sampled rows are evaluated, so the count covers those rows only and is not comparable with ML-1M.
- **0.8** is MAGCL's keep threshold beta (Eq 3).

### 3a. Where "user-user" and "item-item" come from

The interaction data is bipartite: users connect only to items, so there are no direct user-user or item-item edges. These relations are
derived from two-step paths (user - item - user, item - user - item). They are the quantities MAGCL's high-order graph uses (Eqs 2-3).

With R the 0/1 users x items matrix (`scripts/graph_viz.py`, `side_stats`, line 67):
- **User-user overlap** = R R^T. Entry (x, y) = number of items users x and y share.
- **Item-item overlap** = R^T R. Entry (x, y) = number of users who touched both items.
- The script runs the same function twice, on R for the user side and on R^T for the item side.

Tiny example:

| | item1 | item2 | item3 |
|---|---|---|---|
| userA | 1 | 1 | 0 |
| userB | 0 | 1 | 1 |
| userC | 1 | 1 | 1 |

userA and userB share only item2: overlap = 1, Jaccard = 1 / (2 + 2 - 1) = 1/3. item1 and item2 are both used by userA and userC: overlap = 2.

### 3b. "Fraction of pairs with any overlap"

```python
"frac_pairs_with_any_overlap": n_pairs / (len(rows) * (k - 1))      # graph_viz.py:83
```
- **Numerator `n_pairs`**: for each evaluated row x, count the nodes y != x with overlap > 0 (self-pairs are zeroed first).
- **Denominator**: every evaluated row compared with the other k - 1 nodes. Pairs with zero overlap are included.
- Pairs are ordered, so (x, y) and (y, x) count separately.

| Case | Denominator | Fraction | Pairs with any overlap |
|---|---|---|---|
| ML-1M users | 6,039 x 6,038 = 36.5M | 0.9401 | about 34.3M |
| ML-1M items | 3,628 x 3,627 = 13.2M | 0.7664 | about 10.1M |
| iFashion users (10,000 sampled rows) | 10,000 x 299,999 = 3.0B | 0.002357 | about 7.1M |
| iFashion items (10,000 sampled rows) | 10,000 x 81,613 = 816M | 0.001321 | about 1.08M |

### 3c. Why iFashion's fraction is so low

Rough baseline: if every node picked its neighbours uniformly at random, the expected number of shared neighbours of two nodes is about
deg x deg / (size of the other side).

| Pair type | Typical degrees | Other side | Expected shared | Chance of >= 1 shared | Observed |
|---|---|---|---|---|---|
| ML-1M users | 138.5 items each | 3,628 items | 5.3 | about 99.5% | 94.0% |
| ML-1M items | 230.6 users each | 6,039 users | 8.8 | about 99.98% | 76.6% |
| iFashion users | 4.2 items each | 81,614 items | 0.0002 | about 0.02% | 0.236% |
| iFashion items | 15.4 users each | 300,000 users | 0.0008 | about 0.08% | 0.132% |

- **iFashion is sparse on both sides.** Median user degree is 3 of 81,614 items, and the median item has 4 of 300,000 users, so two nodes
  rarely share a neighbour.
- **ML-1M is dense.** Users have about 139 of 3,628 movies, so sharing at least one is nearly certain and overlap alone says little there;
  this is why MAGCL normalises with Jaccard.
- **Real data sits above the baseline on iFashion** (about 11x for users, 1.7x for items) because a few items are very popular (up to 2,239
  users) and the data has community structure. On ML-1M it sits below the baseline because many users and items have low degree.
- The baseline assumes uniform picks, so treat it as an order-of-magnitude guide.

## 4. What the numbers suggest (inferences from structure, not training results)

- **iFashion has far stronger community structure than ML-1M.** About 30-44% of iFashion edges stay inside a cluster at every K from 16 to 256 (lift 6-21x). ML-1M falls from 46% at K=4 to 5% at K=64 (lift 1.8-3.7x).
- **Co-occurrence agrees.** Same-cluster user pairs share 10x more items than different-cluster pairs on iFashion (item pairs 5x), against 1.4-1.5x on ML-1M.
- **MAGCL's beta = 0.8 rule rarely fires.** ML-1M has no user pair and about 70 item pairs (unordered) above 0.8, so the top-S rule builds almost all of the high-order graph there.
- **ML-1M clusters turn into popularity bins from K=32.** Item-popularity eta^2 is 0.02 at K=4 and 0.26 at K=32.
- **iFashion clusters group users by activity as K grows.** User-activity eta^2 rises from 0.02 at K=16 to 0.20 at K=128.
- **iFashion has one giant cluster that K does not split.** The largest cluster is 24,887 nodes at K=64 and 27,798 at K=128.
- **Existing recorded runs (ML-1M, K=16, test recall@20):** replace 0.2771, union 0.2931; plain MAGCL 0.2956. No iFashion MAGCL run is recorded yet.
- **Caveat:** Graclus is randomized and each K is one partition, so small differences between neighbouring K values are not meaningful.

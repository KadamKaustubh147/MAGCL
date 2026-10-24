# Graclus clustering on a literal 80% train split (ML-1M and Alibaba-iFashion)

Same stats as `graclus_stats.md`, but on a train split that really holds 80% of the interactions. Structure only: nothing here is a training result.

## 1. What changed and why

RecBole's default split here is per user (`group_by: user`): every user with 3 or more interactions gets at least one validation and one test item.
That leaves iFashion's train split with 63.6% of the data (90% of its users have fewer than 10 interactions). The new split is a plain random
80/10/10 over all interactions (`group_by: none`, seed 2020), set by the overlay `configs/split80.yaml`.

| | Per-user split (old) | Global 80/10/10 (new) |
|---|---|---|
| ML-1M train edges | 674,674 (80.7%) | 669,184 (80.0%) |
| iFashion train edges | 1,023,049 (63.6%) | 1,286,251 (80.0%) |
| ML-1M users / items with a train edge | 6,039 / 3,600 | 6,038 / 3,600 |
| iFashion users / items with a train edge | 300,000 / 74,380 | 296,705 / 78,059 |

Side effect: on iFashion, 3,295 users (1.1%) and 3,555 items have no train edge, because all of their interactions landed in validation or test.
Those users can still be evaluated but have no history, which the per-user split never allowed.

## 2. How it was made

```powershell
python scripts/prep_graclus.py --config configs/magcl.yaml configs/split80.yaml --out outputs/graclus_split80
python scripts/prep_graclus.py --config configs/magcl_ifashion.yaml configs/split80.yaml --out outputs/graclus_split80
```
```bash
# WSL, run INSIDE the output folder so the new part files cannot overwrite the old ones in graclus1.2/
cd /mnt/c/dev_stuff/MAGCL_claude/outputs/graclus_split80
../../graclus1.2/graclus ml-1m_preprocessed_byrating_train.graph <K>        # K = 4 8 16 32 64
../../graclus1.2/graclus Alibaba-iFashion_train.graph <K>                    # K = 16 32 64 128 256
```
Everything is in `outputs/graclus_split80/`: the `.graph` and `.map.npz` inputs (the map holds an edge fingerprint of this split),
the `.part.<K>` outputs, and `graclus_runs.log` with Graclus's console summaries. Nothing in `outputs/graclus/` or `graclus1.2/` was changed.

Graclus console summaries (normalized cut: worst case = K; balance = largest cluster / ideal size):

| Dataset | K | Cut value | Balance | Total time (s) |
|---|---|---|---|---|
| ML-1M (9,638 nodes, 669,184 edges) | 4 | 2.184 | 1.17 | 2.3 |
| | 8 | 5.551 | 1.46 | 1.0 |
| | 16 | 13.118 | 1.46 | 0.9 |
| | 32 | 28.740 | 4.51 | 0.9 |
| | 64 | 59.937 | 2.35 | 0.9 |
| iFashion (374,764 nodes, 1,286,251 edges) | 16 | 7.498 | 3.17 | 8.7 |
| | 32 | 17.421 | 3.21 | 8.9 |
| | 64 | 41.580 | 4.04 | 11.3 |
| | 128 | 91.054 | 5.14 | 17.8 |
| | 256 | 190.064 | 3.08 | 26.1 |

## 3. Cluster stats on the new split

Column meanings are the same as in `graclus_stats.md` (edges inside = share of train interactions whose user and item share a cluster; random =
that share for a random assignment with the same cluster volumes; lift = inside / random; CV = std / mean of cluster size; eta^2 = share of the
variance of log degree explained by the cluster label).

**ML-1M**: 6,038 users, 3,600 items, 669,184 train edges, mean user degree 110.8, mean item degree 185.9.

| K | Size min / median / max | CV | Users / items per cluster (median) | Edges inside | Random | Lift | Largest cluster's edge share | Item-pop eta^2 | User-act eta^2 |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2105 / 2354 / 2824 | 0.11 | 1528 / 913 | 45.4% | 25.2% | 1.8 | 27.4% | 0.02 | 0.02 |
| 8 | 738 / 1142 / 1754 | 0.26 | 697 / 413 | 31.2% | 13.9% | 2.3 | 22.8% | 0.05 | 0.08 |
| 16 | 399 / 563 / 882 | 0.22 | 349 / 206 | 17.4% | 6.5% | 2.7 | 13.2% | 0.08 | 0.11 |
| 32 | 120 / 235 / 1359 | 0.74 | 139 / 94 | 9.3% | 3.9% | 2.4 | 20.8% | 0.26 | 0.13 |
| 64 | 65 / 138 / 354 | 0.34 | 85 / 54 | 5.9% | 1.6% | 3.7 | 8.2% | 0.16 | 0.14 |

**Alibaba-iFashion**: 296,705 users, 78,059 items, 1,286,251 train edges, mean user degree 4.3, mean item degree 16.5.

| K | Size min / median / max | CV | Users / items per cluster (median) | Edges inside | Random | Lift | Largest cluster's edge share | Item-pop eta^2 | User-act eta^2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2326 / 18475 / 74260 | 0.81 | 14566 / 3115 | 56.1% | 12.0% | 4.7 | 23.0% | 0.03 | 0.04 |
| 32 | 559 / 7856 / 37604 | 0.91 | 6466 / 1156 | 48.5% | 7.1% | 6.8 | 13.1% | 0.04 | 0.05 |
| 64 | 152 / 5486 / 23671 | 0.66 | 4472 / 830 | 32.6% | 2.5% | 13.0 | 7.7% | 0.06 | 0.08 |
| 128 | 1 / 3074 / 15053 | 0.72 | 2324 / 527 | 26.5% | 1.1% | 23.6 | 3.8% | 0.05 | 0.09 |
| 256 | 1 / 1470 / 4508 | 0.58 | 1157 / 147 | 22.0% | 1.0% | 21.0 | 10.0% | 0.14 | 0.13 |

Every node has at least 5 same-type cluster-mates at every K. At K=128 and K=256 on iFashion, 10 clusters contain no users at all (items only).

## 4. Old split vs new split: share of edges inside a cluster

| ML-1M K | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Per-user split | 45.7% | 30.9% | 16.4% | 9.0% | 5.1% |
| Global 80% | 45.4% | 31.2% | 17.4% | 9.3% | 5.9% |

| iFashion K | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| Per-user split | 44.1% | 38.4% | 40.5% | 37.3% | 29.1% |
| Global 80% | 56.1% | 48.5% | 32.6% | 26.5% | 22.0% |
| Largest cluster, per-user split | 32,897 | 21,159 | 24,887 | 27,798 | 10,118 |
| Largest cluster, global 80% | 74,260 | 37,604 | 23,671 | 15,053 | 4,508 |
| Size CV, per-user split | 0.33 | 0.37 | 0.53 | 0.84 | 0.51 |
| Size CV, global 80% | 0.81 | 0.91 | 0.66 | 0.72 | 0.58 |

## 5. What stands out (structure only; causes not diagnosed)

- **ML-1M barely changes.** Its split moved from 80.7% to 80.0% train, and the edges-inside share is within about 1 point at every K.
- **iFashion changes a lot.** The train graph has 26% more edges.
  - At K=16 and 32, more edges stay inside (56.1% and 48.5% against 44.1% and 38.4%), but the clusters are much less balanced. At K=16 the largest cluster
    has 74,260 nodes, about 20% of the graph.
  - At K=64 to 256, fewer edges stay inside (32.6%, 26.5%, 22.0% against 40.5%, 37.3%, 29.1%), while lift against random stays high (13.0 to 23.6).
- **High K on iFashion has degenerate clusters.** K=128 and K=256 each contain a singleton cluster and 10 clusters with no users.
- **Cold users are new.** 1.1% of iFashion users have no train edge under the global split. Any results on this split are not comparable with results on the
  per-user split without rerunning the baselines.
- **Graclus is randomized and each K is one partition.** Small differences between neighbouring K values are not meaningful.

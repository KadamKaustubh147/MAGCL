"""Item -> cluster assignment for JEPA's blocks (config `jepa_cluster`, default `graclus` = the original behaviour).

All methods see ONLY the train split (R = users x items, the model's own interaction matrix) and return a vector
of contiguous cluster ids 0..p-1 with p <= n_clusters, so they are drop-in replacements for each other:

  graclus     multi-level Graclus matching on C_bar = R^T R / sqrt(d_i d_j)   (jepa.graclus_item_clusters; p lands in (n/2, n])
  metis       METIS k-way min-cut on the same C_bar, exactly n balanced parts  (needs `pymetis`: Linux/macOS wheels only)
  spectral    k-means on the top singular vectors of the degree-normalised R   (graph structure, but no cut objective)
  popularity  equal-sized buckets of items sorted by train degree               (control: no co-occurrence structure)
  random      uniformly random clusters                                         (control: no structure at all)

The two controls matter: if `random` scores like `graclus`, the clustering is not what JEPA's gain depends on.
Cluster COUNTS differ by method (graclus lands below n_clusters, the others hit it), so compare at the p each run logs.
"""

import numpy as np
import scipy.sparse as sp

METHODS = ("graclus", "metis", "spectral", "popularity", "random")


def _binary(R) -> sp.csr_matrix:
    R = R.tocsr().astype(np.float32)
    R.data[:] = 1.0
    return R


def _compact(assign) -> np.ndarray:
    """Relabel to contiguous ids 0..p-1 (METIS / k-means can leave a part empty; JEPA takes p = max id + 1)."""
    return np.unique(np.asarray(assign), return_inverse=True)[1].astype(np.int64)


def _item_graph(R) -> sp.csr_matrix:
    """C_bar without the diagonal: item-item co-occurrence normalised by sqrt(d_i d_j) (the same weights Graclus gets)."""
    R = _binary(R)
    deg = np.asarray(R.sum(axis=0)).ravel()
    C = (R.T @ R).tocoo()
    off = C.row != C.col
    w = C.data[off] / np.sqrt(deg[C.row[off]] * deg[C.col[off]])
    return sp.csr_matrix((w, (C.row[off], C.col[off])), shape=C.shape)


def metis_item_clusters(R, p: int) -> np.ndarray:
    try:
        import pymetis
    except ImportError as e:
        raise ImportError("jepa_cluster=metis needs `pymetis` (PyPI ships Linux/macOS wheels, none for Windows): "
                          "pip install pymetis") from e
    G = _item_graph(R)
    # METIS wants integer edge weights: C_bar is in (0, 1], so 1000 levels, never 0 (a 0-weight edge would be free to cut)
    w = np.maximum(np.rint(G.data * 1000), 1).astype(np.int64)
    _, parts = pymetis.part_graph(p, xadj=G.indptr.tolist(), adjncy=G.indices.tolist(), eweights=w.tolist())
    return _compact(parts)


def spectral_item_clusters(R, p: int, seed: int, dim: int = 32) -> np.ndarray:
    from scipy.sparse.linalg import svds
    from sklearn.cluster import KMeans

    R = _binary(R)
    du = np.asarray(R.sum(axis=1)).ravel()
    di = np.asarray(R.sum(axis=0)).ravel()
    inv = lambda d: np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    Rn = sp.diags(inv(du)) @ R @ sp.diags(inv(di))
    k = min(dim, min(Rn.shape) - 1)
    _, _, Vt = svds(Rn.astype(np.float64), k=k, random_state=seed)
    # Keep ALL top-k vectors and row-normalise (Ng-Jordan-Weiss). Do not drop the "trivial" top one: with c connected
    # components there are c singular values equal to 1 and SVD returns an arbitrary basis of that tied subspace, so
    # dropping any single vector can discard the one that separates the components. Row-normalised, the embedding is
    # invariant to rotations inside the tied subspace, so k-means sees the same geometry whatever basis svds picked.
    emb = Vt.T
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return _compact(KMeans(n_clusters=p, n_init=3, random_state=seed).fit_predict(emb))


def popularity_item_clusters(R, p: int) -> np.ndarray:
    deg = np.asarray(_binary(R).sum(axis=0)).ravel()
    n = len(deg)
    assign = np.empty(n, dtype=np.int64)
    assign[np.argsort(deg, kind="stable")] = np.arange(n) * p // n
    return _compact(assign)


def random_item_clusters(n_items: int, p: int, seed: int) -> np.ndarray:
    return _compact(np.random.default_rng(seed).integers(0, p, n_items))


def build_item_clusters(method: str, R, p: int, seed: int):
    """-> torch.LongTensor (n_items,) of cluster ids. R = TRAIN users x items (scipy sparse)."""
    import torch

    if method not in METHODS:
        raise ValueError(f"jepa_cluster must be one of {METHODS}, got {method!r}")
    if method == "graclus":
        from jepa_cf.jepa import graclus_item_clusters   # lazy: jepa.py imports this module
        return graclus_item_clusters(R, p).long()
    if p < 2:
        return torch.zeros(R.shape[1], dtype=torch.long)
    assign = {
        "metis": lambda: metis_item_clusters(R, p),
        "spectral": lambda: spectral_item_clusters(R, p, seed),
        "popularity": lambda: popularity_item_clusters(R, p),
        "random": lambda: random_item_clusters(R.shape[1], p, seed),
    }[method]()
    return torch.as_tensor(assign, dtype=torch.long)

"""Cluster-position hint for the JEPA predictor (`jepa_z: position`).

A fixed coordinate per item cluster, computed ONCE from TRAIN interactions (spectral embedding of the
cluster-cluster co-occurrence graph), then one learned linear layer m -> d. It is an address: it depends on
the cluster id only, never on which items a user hid.

  compute_cluster_positions   numpy, cached to disk, no recomputation during training
  PositionHint                the only learned part is `proj`; `pos` is a buffer
  apply_control               `shuffled` / `random` controls for the ablation
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)
_VERSION = 2  # bump whenever compute_cluster_positions changes, so old cache files are not served


def _cache_stem(cache_dir, train_users, train_items, item_cluster, p, m, topk, eps, source) -> Path:
    h = hashlib.sha1()
    for a in (train_users, train_items, item_cluster):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    h.update(repr((_VERSION, p, m, topk, eps, source)).encode())
    return Path(cache_dir) / f"pos_{h.hexdigest()[:16]}"


def compute_cluster_positions(train_users, train_items, item_cluster, p, m=8, topk=10, eps=1e-3,
                              source="pmi", cache_dir=None):
    """(pos float32 (p, m), diagnostics dict). Inputs are TRAIN-split arrays only; item_cluster[i] < 0 = no cluster.
    cache_dir=None disables the cache."""
    if not 1 <= m <= p - 1:
        raise ValueError(f"need 1 <= m <= p-1, got m={m}, p={p}")
    if source not in ("raw", "pmi"):
        raise ValueError(f"source must be raw|pmi, got {source!r}")
    u, it, cl = (np.asarray(a, dtype=np.int64) for a in (train_users, train_items, item_cluster))

    if cache_dir is not None:
        stem = _cache_stem(cache_dir, u, it, cl, p, m, topk, eps, source)
        npy, js = Path(str(stem) + ".npy"), Path(str(stem) + ".json")
        if npy.exists() and js.exists():
            return np.load(npy), json.loads(js.read_text())

    # 1-2. user x cluster counts N, cluster co-occurrence S = N^T N with a zero diagonal
    c = cl[it]
    ok = c >= 0
    N = sp.csr_matrix((np.ones(ok.sum()), (u[ok], c[ok])), shape=(int(u.max()) + 1, p))  # duplicates sum = counts
    S = (N.T @ N).toarray()
    np.fill_diagonal(S, 0)
    r = S.sum(axis=1)
    empty = r == 0
    if empty.any():
        log.warning("%d of %d clusters have no co-occurrence with any other cluster (empty or isolated), "
                    "their position is set to the neutral 0: ids %s", empty.sum(), p, np.flatnonzero(empty)[:10].tolist())

    # 3. edge weights, top-k per row, symmetrise
    if source == "raw":
        W = S.copy()
    else:
        T = S.sum()
        a, b = np.nonzero(S)
        W = np.zeros_like(S)
        W[a, b] = np.maximum(0.0, np.log(S[a, b] * T / (r[a] * r[b])))
    keep = np.argsort(-W, axis=1, kind="stable")[:, :topk]
    rows = np.arange(p)[:, None]
    sparse_W = np.zeros_like(W)
    sparse_W[rows, keep] = W[rows, keep]
    W = np.maximum(sparse_W, sparse_W.T)
    if W.max() <= 0:
        raise ValueError("no positive co-occurrence weight between any two clusters")

    # 4. components BEFORE the safety links
    components = int(connected_components(sp.csr_matrix(W > 0), directed=False)[0])
    log.info("cluster graph: %d connected components before eps links", components)

    # 5-7. safety links, normalise, eigendecompose (descending)
    W = W + eps * W.max() * (1.0 - np.eye(p))
    d = W.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    A = W / np.sqrt(np.outer(d_safe, d_safe))
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(-vals, kind="stable")
    vals, vecs = vals[order], vecs[:, order]

    # 8-9. skip the trivial eigenvector, standardise columns over the NON-empty clusters (so the neutral 0 given to
    # empty clusters below really is the column mean), fix signs
    live = np.flatnonzero(~empty)
    pos = vecs[:, 1:m + 1].copy()
    for j in range(m):
        mean, std = pos[live, j].mean(), pos[live, j].std()
        if std < 1e-8:
            log.warning("position column %d is constant (std %.2e), kept as zeros", j, std)
            pos[:, j] = 0.0
            continue
        col = (pos[:, j] - mean) / std
        pos[:, j] = col * np.sign(col[live][np.argmax(np.abs(col[live]))])
    pos[empty] = 0.0
    pos = pos.astype(np.float32)

    # 10. diagnostics
    dist = np.linalg.norm(pos[live][:, None] - pos[live][None], axis=-1)
    iu = np.triu_indices(len(live), 1)
    sizes = np.bincount(cl[cl >= 0], minlength=p)
    neighbours = {}
    for t in np.argsort(-sizes, kind="stable")[:5]:
        if empty[t]:
            continue
        dt = np.linalg.norm(pos[live] - pos[t], axis=1)
        dt[live == t] = np.inf
        neighbours[str(t)] = [int(live[j]) for j in np.argsort(dt, kind="stable")[:3]]  # str keys: same dict after a JSON cache hit
    diag = {
        "eigenvalues": vals[:m + 2].tolist(),
        "components": components,
        "n_empty_clusters": int(empty.sum()),
        "spectral_gap": float(vals[m] - vals[m + 1]) if p >= m + 2 else None,  # lambda_{m+1} - lambda_{m+2}
        "mean_pairwise_dist": float(dist[iu].mean()) if len(live) > 1 else None,  # non-empty clusters only
        "blur_fraction": float((dist[iu] < 0.5).mean()) if len(live) > 1 else None,  # pairs closer than 0.5
        "nearest_of_largest": neighbours,  # 5 clusters with the most items -> their 3 nearest clusters
    }

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(npy, pos)
        js.write_text(json.dumps(diag, indent=2))
    return pos, diag


def apply_control(pos: np.ndarray, control: str, seed: int) -> np.ndarray:
    """Ablation controls: `shuffled` = same rows, wrong clusters; `random` = fixed Gaussian, no structure."""
    if control == "none":
        return pos
    rng = np.random.default_rng(seed)
    if control == "shuffled":
        return pos[rng.permutation(len(pos))]
    if control == "random":
        return rng.standard_normal(pos.shape).astype(np.float32)
    raise ValueError(f"pos_control must be none|shuffled|random, got {control!r}")


class PositionHint(torch.nn.Module):
    def __init__(self, pos: np.ndarray, d: int):
        super().__init__()
        self.register_buffer("pos", torch.tensor(pos, dtype=torch.float32))  # (p, m), NOT trained
        self.proj = torch.nn.Linear(pos.shape[1], d, bias=False)             # the only learned part
        torch.nn.init.normal_(self.proj.weight, std=0.1)

    def forward(self, cluster_ids):  # (B,) long -> (B, d)
        return self.proj(self.pos[cluster_ids])

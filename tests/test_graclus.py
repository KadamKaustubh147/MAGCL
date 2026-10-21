"""Graph-level checks for the Graclus high-order variants (no RecBole needed).

  off      labels=None must reproduce the baseline Jaccard graph EXACTLY
  replace  every kept edge joins two nodes of the same cluster
  union    superset of both the Jaccard graph and the restricted graph

Run with `pytest tests/test_graclus.py` or `python tests/test_graclus.py`.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from magcl import graphs as base
from magcl_graclus import graphs as gr


def _data(seed=0, m=60, n=40, density=0.15, k=4):
    rng = np.random.default_rng(seed)
    R = sp.csr_matrix((rng.random((m, n)) < density).astype(np.float64))
    return R, rng.integers(0, k, m), rng.integers(0, k, n)


def test_off_matches_baseline():
    R, _, _ = _data()
    a, b = base.high_order_adj(R, 0.8, 5), gr.high_order_adj(R, 0.8, 5)
    assert (a != b).nnz == 0
    c = gr._row_norm(gr.high_order_binary(R, 0.8, 5))
    assert (a != c).nnz == 0


def test_replace_stays_inside_clusters():
    R, ul, il = _data()
    A, jac = gr.high_order_adj_graclus(R, 0.8, 5, "replace", ul, il, 5)
    coo = A.tocoo()
    labels = np.concatenate([ul, il])
    assert coo.nnz > 0 and (labels[coo.row] == labels[coo.col]).all()
    assert gr.fraction_inside(jac, labels) < 1.0      # random labels: plain Jaccard crosses clusters


def test_union_is_superset():
    R, ul, il = _data()
    U, jac = gr.high_order_adj_graclus(R, 0.8, 5, "union", ul, il, 5)
    Rr, _ = gr.high_order_adj_graclus(R, 0.8, 5, "replace", ul, il, 5)
    pat = lambda M: set(zip(*M.nonzero()))  # noqa: E731
    assert pat(jac) <= pat(U) and pat(Rr) <= pat(U)
    assert np.allclose(np.asarray(U.sum(axis=1)).ravel()[np.asarray(U.sum(axis=1)).ravel() > 0], 1.0)


if __name__ == "__main__":
    test_off_matches_baseline(); test_replace_stays_inside_clusters(); test_union_is_superset()
    print("ok")

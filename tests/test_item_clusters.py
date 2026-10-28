"""Item clusterers (src/jepa_cf/item_clusters.py): contract + 'two disconnected communities must be separated'."""

import numpy as np
import pytest
import scipy.sparse as sp

from jepa_cf import item_clusters as ic

N_ITEMS = 30


@pytest.fixture
def two_communities():
    """40 users; users 0-19 only touch items 0-14, users 20-39 only items 15-29 (dense, so each side is one blob)."""
    rng = np.random.default_rng(0)
    R = np.zeros((40, N_ITEMS), dtype=np.float32)
    R[:20, :15] = rng.random((20, 15)) < 0.6
    R[20:, 15:] = rng.random((20, 15)) < 0.6
    return sp.csr_matrix(R)


def _separates(assign):
    side = np.arange(N_ITEMS) >= 15
    return all(len(set(side[assign == c])) == 1 for c in np.unique(assign))   # no cluster mixes the two communities


def _contiguous(assign, p):
    assert assign.shape == (N_ITEMS,)
    assert set(np.unique(assign)) == set(range(int(assign.max()) + 1)) and assign.max() < p


@pytest.mark.parametrize("method", ic.METHODS)
def test_contract(method, two_communities):
    if method == "metis":
        pytest.importorskip("pymetis")
    assign = ic.build_item_clusters(method, two_communities, 4, seed=0).numpy()
    _contiguous(assign, 4)


@pytest.mark.parametrize("method", ["graclus", "metis", "spectral"])
def test_structure_aware_methods_split_the_communities(method, two_communities):
    if method == "metis":
        pytest.importorskip("pymetis")
    if method == "spectral":
        assign = ic.spectral_item_clusters(two_communities, 2, seed=0, dim=4)
    else:
        assign = ic.build_item_clusters(method, two_communities, 2, seed=0).numpy()
    assert _separates(assign)
    assert len(np.unique(assign)) == 2


def test_popularity_buckets_follow_degree_and_are_balanced(two_communities):
    assign = ic.popularity_item_clusters(two_communities, 5)
    deg = np.asarray(two_communities.sum(axis=0)).ravel()
    assert (np.diff(assign[np.argsort(deg, kind="stable")]) >= 0).all()
    assert np.bincount(assign).max() - np.bincount(assign).min() <= 1


def test_random_is_seeded():
    a, b = ic.random_item_clusters(500, 8, seed=1), ic.random_item_clusters(500, 8, seed=1)
    assert (a == b).all() and not (a == ic.random_item_clusters(500, 8, seed=2)).all()


def test_unknown_method(two_communities):
    with pytest.raises(ValueError, match="jepa_cluster"):
        ic.build_item_clusters("kmeans", two_communities, 4, seed=0)

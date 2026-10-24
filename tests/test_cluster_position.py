"""Tests for src/jepa_cf/cluster_position.py (synthetic data, no download). Run: pytest tests/test_cluster_position.py"""

import inspect
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_cf.cluster_position import PositionHint, apply_control, compute_cluster_positions


def planted(groups, items_per_cluster, n_users=600, per_user=12, noise=0.0, seed=0):
    """Users pick one group of clusters and draw items uniformly from that group's items (so big clusters dominate
    raw counts). Returns (users, items, item_cluster, p)."""
    rng = np.random.default_rng(seed)
    item_cluster = np.repeat(np.arange(len(items_per_cluster)), items_per_cluster)
    pool = [np.flatnonzero(np.isin(item_cluster, g)) for g in groups]
    users, items = [], []
    for u in range(n_users):
        g = rng.integers(len(groups))
        for _ in range(per_user):
            src = rng.integers(len(groups)) if rng.random() < noise else g
            users.append(u)
            items.append(rng.choice(pool[src]))
    return np.array(users), np.array(items), item_cluster, len(items_per_cluster)


GROUPS3 = [[0, 1, 2], [3, 4, 5], [6, 7]]
SIZES3 = [60, 6, 4, 50, 5, 3, 40, 2]  # deliberately unequal


def test_shape_and_normalization():
    u, i, ic, p = planted(GROUPS3, SIZES3)
    pos, _ = compute_cluster_positions(u, i, ic, p, m=3)
    assert pos.shape == (p, 3) and pos.dtype == np.float32
    assert np.allclose(pos.mean(axis=0), 0, atol=1e-4) and np.allclose(pos.std(axis=0), 1, atol=1e-4)


def test_determinism_and_cache(tmp_path, monkeypatch):
    u, i, ic, p = planted(GROUPS3, SIZES3)
    a, _ = compute_cluster_positions(u, i, ic, p, m=2)
    b, _ = compute_cluster_positions(u, i, ic, p, m=2)
    assert np.array_equal(a, b)

    first, diag = compute_cluster_positions(u, i, ic, p, m=2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("pos_*.npy"))) == 1 and len(list(tmp_path.glob("pos_*.json"))) == 1
    monkeypatch.setattr(np.linalg, "eigh", lambda *_: (_ for _ in ()).throw(AssertionError("recomputed on a cache hit")))
    second, diag2 = compute_cluster_positions(u, i, ic, p, m=2, cache_dir=tmp_path)
    assert np.array_equal(first, second) and diag == diag2
    with pytest.raises(AssertionError):  # a different setting is a different key -> really recomputes
        compute_cluster_positions(u, i, ic, p, m=3, cache_dir=tmp_path)


def test_planted_groups_pmi():
    u, i, ic, p = planted(GROUPS3, SIZES3)
    pos, diag = compute_cluster_positions(u, i, ic, p, m=2, source="pmi")
    group_of = {c: gi for gi, g in enumerate(GROUPS3) for c in g}
    for c in range(p):
        d = np.linalg.norm(pos - pos[c], axis=1)
        d[c] = np.inf
        assert group_of[int(np.argmin(d))] == group_of[c], f"cluster {c}: nearest neighbour is in another group"
    l1, l2, l3, l4 = diag["eigenvalues"]
    assert l3 - l4 > l2 - l3


def test_disconnected_graph_needs_eps():
    u, i, ic, p = planted([[0, 1, 2, 3], [4, 5, 6, 7]], [10] * 8)
    _, none = compute_cluster_positions(u, i, ic, p, m=2, eps=0.0)
    assert none["components"] >= 2 and sum(abs(v - 1) < 1e-6 for v in none["eigenvalues"]) >= 2
    _, linked = compute_cluster_positions(u, i, ic, p, m=2, eps=1e-3)
    assert sum(abs(v - 1) < 1e-6 for v in linked["eigenvalues"]) == 1


def test_signature_takes_only_train_arrays():
    params = set(inspect.signature(compute_cluster_positions).parameters)
    assert params == {"train_users", "train_items", "item_cluster", "p", "m", "topk", "eps", "source", "cache_dir"}


def test_pos_is_a_buffer_and_proj_trains():
    hint = PositionHint(np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32), d=4)
    assert "pos" in hint.state_dict() and not hint.pos.requires_grad
    assert "pos" not in dict(hint.named_parameters()) and "proj.weight" in dict(hint.named_parameters())
    hint(torch.tensor([0, 2, 4])).sum().backward()
    assert hint.proj.weight.grad is not None and hint.proj.weight.grad.abs().sum() > 0


def test_output_depends_on_cluster_only():
    hint = PositionHint(np.random.default_rng(1).standard_normal((5, 3)).astype(np.float32), d=4)
    out = hint(torch.tensor([3, 3, 1]))
    assert torch.equal(out[0], out[1]) and not torch.equal(out[0], out[2])


def test_empty_cluster_is_finite_and_warns(caplog):
    u, i, ic, _ = planted(GROUPS3, SIZES3)  # clusters 0..7 hold items; cluster 8 is declared but empty
    with caplog.at_level(logging.WARNING, logger="jepa_cf.cluster_position"):
        pos, diag = compute_cluster_positions(u, i, ic, 9, m=2)
    assert np.isfinite(pos).all() and diag["n_empty_clusters"] == 1
    assert (pos[8] == 0).all() and any("no co-occurrence" in r.message for r in caplog.records)
    live = pos[:8]  # the neutral 0 must be the mean of the real clusters, which stay standardised
    assert np.allclose(live.mean(axis=0), 0, atol=1e-4) and np.allclose(live.std(axis=0), 1, atol=1e-4)


def test_controls_keep_shape_and_change_values():
    pos = np.random.default_rng(2).standard_normal((6, 3)).astype(np.float32)
    shuffled, rnd = apply_control(pos, "shuffled", 0), apply_control(pos, "random", 0)
    assert shuffled.shape == rnd.shape == pos.shape
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(pos, axis=0)) and not np.array_equal(shuffled, pos)
    assert not np.array_equal(rnd, pos) and np.array_equal(rnd, apply_control(pos, "random", 0))  # fixed by the seed


def test_model_wiring_and_train_only():
    """jepa_z=position through the real model: finite loss, only proj.weight is new, edges disjoint from valid/test."""
    from jepa_cf.jepa import JEPA
    from magcl import compat  # noqa: F401
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    cfg = {
        "data_path": str(Path(__file__).resolve().parent.parent / "data"), "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
        "load_col": {"inter": ["user_id", "item_id"]}, "seed": 2020, "use_gpu": False, "worker": 0, "embedding_size": 8,
        "train_batch_size": 16, "eval_batch_size": 16, "epochs": 1,
        "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "RO", "group_by": "user", "mode": "full"},
        "metrics": ["Recall"], "topk": 5, "valid_metric": "Recall@5",
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "dynamic": False},
        "n_layers": 2, "reg_weight": 1e-5, "require_pow": True, "jepa_weight": 1.0, "mask_ratio": 0.5, "n_clusters": 8,
        "jepa_z": "position", "pos_dim_m": 2, "pos_topk": 5, "pos_eps": 1e-3, "pos_source": "pmi", "pos_control": "none",
    }
    config = Config(model=JEPA, dataset="smoke_jepa", config_dict=cfg)
    train, valid, test = data_preparation(config, create_dataset(config))
    model = JEPA(config, train.dataset).to(config["device"])

    pairs = lambda ds: set(zip(ds.inter_feat["user_id"].tolist(), ds.inter_feat["item_id"].tolist()))
    used = set(zip(model.edge_u.tolist(), model.edge_i.tolist()))
    assert used == pairs(train.dataset) and not used & pairs(valid.dataset) and not used & pairs(test.dataset)

    assert "position_hint.pos" in model.state_dict() and "position_hint.pos" not in dict(model.named_parameters())
    assert not hasattr(model, "W_z")
    model.train()
    losses = model.calculate_loss(next(iter(train)).to(model.device))
    assert all(torch.isfinite(v) for v in losses) and losses[2] > 0
    sum(losses).backward()
    assert model.position_hint.proj.weight.grad.abs().sum() > 0

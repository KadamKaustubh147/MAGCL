"""JEPA wiring check on the tiny smoke dataset: masks/blocks are consistent, all three `jepa_z`
variants give a finite loss and gradients, and the context graph really lacks the hidden edges.
Run with `pytest tests/test_jepa.py` or `python tests/test_jepa.py`.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_cf.jepa import JEPA
from magcl import compat  # noqa: F401  (scipy patch before recbole import)
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CONFIG = {
    "data_path": str(DATA_ROOT),
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "load_col": {"inter": ["user_id", "item_id"]},
    "seed": 2020,
    "use_gpu": False,
    "worker": 0,
    "embedding_size": 8,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "epochs": 1,
    "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "RO", "group_by": "user", "mode": "full"},
    "metrics": ["Recall", "NDCG"],
    "topk": 5,
    "valid_metric": "Recall@5",
    "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "dynamic": False},
    "n_layers": 2, "reg_weight": 1e-5, "require_pow": True,
    "jepa_weight": 0.1, "mask_ratio": 0.5, "n_clusters": 4, "jepa_z": "mean",
}


def _build(z, **extra):
    config = Config(model=JEPA, dataset="smoke_jepa", config_dict={**CONFIG, "jepa_z": z, **extra})
    train_data, _, _ = data_preparation(config, create_dataset(config))
    return JEPA(config, train_data.dataset).to(config["device"]), train_data


def test_masks_and_blocks():
    model, _ = _build("mean")
    model.train()
    model._resample()
    c = model._ctx
    deg = model.indptr[1:] - model.indptr[:-1]
    hidden_per_user = torch.bincount(c["hu"], minlength=model.n_users)
    assert (hidden_per_user == model.n_hide).all()
    assert (hidden_per_user[deg < 2] == 0).all() and (hidden_per_user <= (deg - 1).clamp(min=0)).all()  # one item stays visible
    # blocks partition the hidden edges and are single-cluster, single-user
    assert (model.item_cluster[c["hi"]] == c["block_cluster"][c["edge_block"]]).all()
    assert (c["hu"] == c["block_user"][c["edge_block"]]).all()
    # context graph = train graph minus hidden edges (each edge appears in both directions)
    assert c["adj"]._nnz() == 2 * (model.edge_u.numel() - c["hu"].numel())
    model.train()
    assert model._ctx is None  # next epoch -> fresh masks


def test_jepa_trains_without_nan():
    for z in ("mean", "learnable", "none"):
        model, train_data = _build(z)
        model.train()
        losses = model.calculate_loss(next(iter(train_data)).to(model.device))  # the Trainer does this move
        assert all(torch.isfinite(v) for v in losses), f"z={z}: non-finite loss {losses}"
        assert len(losses) == 3 and losses[2] > 0, f"z={z}: JEPA loss is zero, no block was predicted"
        sum(losses).backward()
        means = model.pop_loss_means()
        assert abs(means["bpr"] - losses[0].item()) < 1e-6 and abs(means["jepa_weighted"] - losses[2].item()) < 1e-6
        assert model.pop_loss_means()["bpr"] == 0  # reset after pop
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"z={z}: {name} gradient has NaN/inf"
        assert model.predictor[0].weight.grad is not None
        print(f"z={z}: (bpr, reg, jepa) =", [round(v.item(), 6) for v in losses])


def test_centered_keys_remove_the_shared_direction():
    """Every item row = one shared vector c (+ tiny noise) and the predictor outputs c: an uncentered key is matched
    perfectly (the trivial solution, loss ~0); a centered key is only noise, so the loss sits near its honest ~2."""
    losses = {}
    for center in (False, True):
        model, _ = _build("none", jepa_center_keys=center)
        c = torch.ones(model.latent_dim, device=model.device)
        noise = 0.01 * torch.randn(model.item_embedding.weight.shape, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            model.item_embedding.weight.copy_(c + noise.to(model.device))
            for lin in (model.predictor[0], model.predictor[2]):
                lin.weight.zero_()
            model.predictor[0].bias.zero_()
            model.predictor[2].bias.copy_(c)
        model.train()
        losses[center] = model.jepa_loss(torch.arange(model.n_users, device=model.device)).item()
        if center:  # the center is the popularity-weighted item mean, detached
            deg = torch.bincount(model.edge_i, minlength=model.n_items).float()
            manual = ((deg / deg.sum()).unsqueeze(1) * model.item_embedding.weight).sum(0)
            assert torch.allclose(model._key_center(), manual, atol=1e-5) and not model._key_center().requires_grad
    assert losses[False] < 0.05 and losses[True] > 1.0, losses


def test_context_key_source():
    """jepa_key_source=context builds the key from the hidden items' readout on G_ctx (stop-gradient): same masks, a
    different but finite target; the default stays layer0; bad values are rejected."""
    model, _ = _build("learnable", jepa_center_keys=True)
    assert model.key_source == "layer0"
    model.train()
    users = torch.arange(model.n_users, device=model.device)
    layer0 = model.jepa_loss(users).item()               # also builds this epoch's masks
    model.key_source = "context"
    context = model.jepa_loss(users)
    assert torch.isfinite(context) and abs(context.item() - layer0) > 1e-4
    context.backward()
    assert model.predictor[0].weight.grad.abs().sum() > 0 and model.user_embedding.weight.grad.abs().sum() > 0
    try:
        _build("none", jepa_key_source="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("bad jepa_key_source was accepted")


if __name__ == "__main__":
    test_masks_and_blocks()
    test_jepa_trains_without_nan()
    test_centered_keys_remove_the_shared_direction()
    print("JEPA smoke test OK")

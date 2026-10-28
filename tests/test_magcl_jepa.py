"""MAGCL + JEPA wiring on the tiny smoke_jepa dataset: six finite loss terms, the first five identical to plain MAGCL,
gradients from the JEPA term reach E0 and the predictor. Run: pytest tests/test_magcl_jepa.py"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_cf.magcl_jepa import MAGCLJEPA
from magcl_graclus import compat  # noqa: F401
from magcl_graclus.magcl_graclus import MAGCL
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

CONFIG = {
    "data_path": str(Path(__file__).resolve().parent.parent / "data"), "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
    "load_col": {"inter": ["user_id", "item_id"]}, "seed": 2020, "use_gpu": False, "worker": 0, "embedding_size": 8,
    "train_batch_size": 16, "eval_batch_size": 16, "epochs": 1,
    "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "RO", "group_by": "user", "mode": "full"},
    "metrics": ["Recall"], "topk": 5, "valid_metric": "Recall@5",
    "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "dynamic": False},
    "L": 3, "tau": 0.1, "gamma": 1.0, "alpha": 0.7, "theta": 0.05, "beta": 0.8, "S": 2,
    "lam1": 1e-3, "lam2": 1e-4, "lam3": 1e-4, "lam4": 1e-4, "normalize_cl": True,
    "jepa_weight": 0.5, "mask_ratio": 0.5, "n_clusters": 4, "jepa_z": "learnable",
}


def _data(**extra):
    config = Config(model=MAGCLJEPA, dataset="smoke_jepa", config_dict={**CONFIG, **extra})
    train, _, _ = data_preparation(config, create_dataset(config))
    return config, train


def test_six_terms_and_gradients():
    config, train = _data()
    torch.manual_seed(0)
    model = MAGCLJEPA(config, train.dataset).to(config["device"])
    model.train()
    batch = next(iter(train)).to(model.device)
    losses = model.calculate_loss(batch)
    assert len(losses) == 6 and all(torch.isfinite(v) for v in losses) and losses[5] > 0

    # the JEPA term alone must move E0 and the predictor
    model.zero_grad()
    losses[5].backward()
    assert model.E0.grad.abs().sum() > 0 and model.predictor[0].weight.grad.abs().sum() > 0

    means = model.pop_loss_means()
    assert set(means) == {"bpr", "cl", "layer", "unif", "reg", "jepa", "jepa_weighted", "jepa_const"}
    assert abs(means["jepa_weighted"] - losses[5].item()) < 1e-6 and model.pop_loss_means()["bpr"] == 0


def test_first_five_terms_are_plain_magcl():
    config, train = _data()
    torch.manual_seed(0)
    plain = MAGCL(config, train.dataset).to(config["device"])
    torch.manual_seed(0)
    both = MAGCLJEPA(config, train.dataset).to(config["device"])
    assert torch.equal(plain.E0, both.E0)  # same init -> same A_dn, so the comparison is fair
    batch = next(iter(train)).to(both.device)
    ref, got = plain.calculate_loss(batch), both.calculate_loss(batch)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ref, got[:5]))


def test_ema_target_on_magcl():
    """jepa_ema on the MAGCL backbone: the EMA copy follows E0 (the single table) and the JEPA term still moves E0 and the predictor."""
    config, train = _data(jepa_ema=0.9)
    model = MAGCLJEPA(config, train.dataset).to(config["device"])
    model.train()
    batch = next(iter(train)).to(model.device)
    model.calculate_loss(batch)
    first = model._ema_table.clone()
    with torch.no_grad():
        model.E0.add_(1.0)
    live = model.E0.detach().clone()
    model.zero_grad()
    losses = model.calculate_loss(batch)
    assert torch.allclose(model._ema_table, 0.9 * first + 0.1 * live, atol=1e-6) and losses[5] > 0
    losses[5].backward()
    assert model.E0.grad.abs().sum() > 0 and model.predictor[0].weight.grad.abs().sum() > 0

"""One end-to-end check that MAGCL trains through real RecBole plumbing: builds
the graphs, runs forward + calculate_loss + backward on a tiny dataset, and
asserts every loss term and every gradient is finite. No fixtures, no mocks --
if this fails, the model is broken; if it passes, the wiring to RecBole is
sound. Run with `pytest tests/test_model.py` or `python tests/test_model.py`.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from magcl import compat  # noqa: F401  (scipy patch before recbole import)
from magcl.model import MAGCL
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
    # MAGCL hyperparameters (§5.2 defaults; S shrunk for this tiny dataset)
    "L": 3, "tau": 0.1, "gamma": 1.0, "alpha": 0.7, "theta": 0.05,
    "beta": 0.8, "S": 2, "lam1": 1e-3, "lam2": 1e-4, "lam3": 1e-4, "lam4": 1e-4,
    "normalize_cl": True,
}


def test_magcl_trains_without_nan():
    config = Config(model=MAGCL, dataset="smoke", config_dict=CONFIG)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    model = MAGCL(config, train_data.dataset).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(train_data))
    losses = model.calculate_loss(batch)
    names = ["bpr", "cl", "layer", "unif", "reg"]
    for name, value in zip(names, losses):
        assert torch.isfinite(value), f"{name} loss is not finite: {value}"

    total = sum(losses)
    optimizer.zero_grad()
    total.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient has NaN/inf"

    print("bpr, cl, layer, unif, reg =", [round(v.item(), 6) for v in losses])
    print("MAGCL smoke test: all losses and gradients finite. OK")


if __name__ == "__main__":
    test_magcl_trains_without_nan()

"""Train MAGCL + the JEPA loss (src/jepa_cf/magcl_jepa.py) once with the STOCK recbole Trainer, log to MLflow.

Usage (ML-1M, MAGCL's best hyperparameters + JEPA):
    python scripts/run_magcl_jepa.py --config configs/magcl.yaml configs/magcl_jepa.yaml --override configs/best_ml1m_hparams.json
    python scripts/run_magcl_jepa.py --config configs/magcl_ifashion.yaml configs/magcl_jepa.yaml        # iFashion (tuned values are in the yaml)

--config takes several files (later ones win); --override takes any number of JSON files or inline JSON strings, merged in order.
Control run = the same command with a final override  '{"jepa_weight": 0}'  (plain MAGCL).

Per epoch MLflow gets train_bpr / train_cl / train_layer / train_unif / train_reg / train_jepa (raw D) / train_jepa_weighted,
plus valid recall; the Trainer itself only keeps the SUM of the loss tuple, so the parts are read from the model.
"""

import argparse
import json
from pathlib import Path

import mlflow

from jepa_cf.magcl_jepa import MAGCLJEPA
from magcl_graclus import compat  # noqa: F401  (scipy patch before recbole import)
from magcl_graclus.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

_HPARAM_KEYS = [
    "L", "tau", "gamma", "alpha", "theta", "beta", "S", "lam1", "lam2", "lam3", "lam4", "normalize_cl", "embedding_size",
    "jepa_weight", "mask_ratio", "n_clusters", "jepa_z", "pos_dim_m", "pos_topk", "pos_eps", "pos_source", "pos_control",
]
_GRACLUS_DEFAULTS = {"graclus_mode": "off", "S_gr": None, "graclus_part": None, "graclus_map": None}


def load_overrides(items):
    merged = {}
    for item in items:
        path = Path(item)
        merged.update(json.loads(path.read_text() if path.is_file() else item))
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", default=["configs/magcl.yaml", "configs/magcl_jepa.yaml"])
    parser.add_argument("--override", nargs="*", default=[], help="JSON files or inline JSON strings, merged in order")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: MAGCL-JEPA-<dataset>)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quiet", action="store_true", help="hide tqdm progress bars")
    args = parser.parse_args()

    overrides = load_overrides(args.override)
    config = Config(model=MAGCLJEPA, config_file_list=args.config, config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MAGCLJEPA(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    mlflow.set_experiment(args.experiment or f"MAGCL-JEPA-{config['dataset']}")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
        mlflow.log_params({k: (config[k] if k in config else d) for k, d in _GRACLUS_DEFAULTS.items()})
        mlflow.log_params(overrides)

        log_valid = valid_score_callback(config["valid_metric"])

        def on_epoch_end(epoch_idx, valid_score):
            log_valid(epoch_idx, valid_score)
            mlflow.log_metrics({f"train_{k}": v for k, v in model.pop_loss_means().items()}, step=epoch_idx)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=not args.quiet, callback_fn=on_epoch_end)
        log_train_loss_history(trainer)
        mlflow.log_metrics({f"best_valid_{sanitize_metric_name(k)}": float(v) for k, v in best_valid_result.items()})

        test_result = trainer.evaluate(test_data, show_progress=not args.quiet)
        mlflow.log_metrics({f"test_{sanitize_metric_name(k)}": float(v) for k, v in test_result.items()})

        print("\nbest valid:", dict(best_valid_result))
        print("test:", dict(test_result))


if __name__ == "__main__":
    main()

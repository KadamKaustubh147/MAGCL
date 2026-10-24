"""Train the first-version JEPA recommender (src/jepa_cf) once with the STOCK recbole Trainer, log to MLflow.

Usage:
    python scripts/run_jepa.py
    python scripts/run_jepa.py --override '{"jepa_z": "learnable", "n_clusters": 128}'
    python scripts/run_jepa.py --override '{"jepa_weight": 0}'   # = plain LightGCN, for the baseline
"""

import argparse
import json
from pathlib import Path

import mlflow

from jepa_cf.jepa import JEPA
from magcl import compat  # noqa: F401  (scipy patch before recbole import)
from magcl.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

_HPARAM_KEYS = ["n_layers", "reg_weight", "embedding_size", "jepa_weight", "mask_ratio", "n_clusters", "jepa_z", "jepa_center_keys", "jepa_key_source",
                "pos_dim_m", "pos_topk", "pos_eps", "pos_source", "pos_control"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jepa.yaml")
    parser.add_argument("--override", default="{}", help="JSON dict of config overrides, inline or a path to a .json file")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: JEPA-<dataset>)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quiet", action="store_true", help="hide tqdm progress bars")
    args = parser.parse_args()

    override_path = Path(args.override)
    overrides = json.loads(override_path.read_text() if override_path.is_file() else args.override)
    config = Config(model=JEPA, config_file_list=[args.config], config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = JEPA(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    mlflow.set_experiment(args.experiment or f"JEPA-{config['dataset']}")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
        mlflow.log_params(overrides)

        log_valid = valid_score_callback(config["valid_metric"])

        def on_epoch_end(epoch_idx, valid_score):
            # the Trainer only keeps the SUM of the loss tuple, so read the parts from the model
            log_valid(epoch_idx, valid_score)
            mlflow.log_metrics({f"train_{k}": v for k, v in model.pop_loss_means().items()}, step=epoch_idx)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=not args.quiet, callback_fn=on_epoch_end,
        )
        log_train_loss_history(trainer)
        mlflow.log_metrics({f"best_valid_{sanitize_metric_name(k)}": float(v) for k, v in best_valid_result.items()})

        test_result = trainer.evaluate(test_data, show_progress=not args.quiet)
        mlflow.log_metrics({f"test_{sanitize_metric_name(k)}": float(v) for k, v in test_result.items()})

        print("\nbest valid:", dict(best_valid_result))
        print("test:", dict(test_result))


if __name__ == "__main__":
    main()

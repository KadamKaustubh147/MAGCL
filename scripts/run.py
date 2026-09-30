"""Train MAGCL once, using the STOCK recbole.trainer.Trainer, and log to MLflow.

Usage:
    python scripts/run.py
    python scripts/run.py --config configs/magcl.yaml --override '{"lam1": 1e-4, "L": 2}'
    python scripts/run.py --override overrides.json   # a JSON file, sidesteps shell quoting entirely

Then: mlflow ui   (from the project root) -> open the run. Per-epoch charts:
total train loss and the configured valid_metric (recall@20). Full
Recall/NDCG@10/20/50 breakdown is logged as two snapshots (best-valid-epoch,
final test), not a per-epoch history -- see magcl/tracking.py's docstring for
why (no subclassing of Trainer, so only what it exposes publicly is loggable
every epoch).
"""

import argparse
import json
from pathlib import Path

import mlflow

from magcl import compat  # noqa: F401  (scipy patch before recbole import)
from magcl.model import MAGCL
from magcl.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

# Every MAGCL/paper hyperparameter, logged to MLflow's Params tab on every run
# regardless of whether it was overridden -- makes runs comparable at a glance.
_HPARAM_KEYS = [
    "L", "tau", "gamma", "alpha", "theta", "beta", "S",
    "lam1", "lam2", "lam3", "lam4", "normalize_cl", "embedding_size",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/magcl.yaml")
    parser.add_argument("--override", default="{}",
                         help="JSON dict of config overrides, inline or a path to a .json file "
                              "(a file sidesteps shell-quoting headaches with nested quotes)")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: MAGCL-<dataset>)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quiet", action="store_true", help="hide tqdm progress bars")
    args = parser.parse_args()

    override_path = Path(args.override)
    overrides = json.loads(override_path.read_text() if override_path.is_file() else args.override)
    config = Config(model=MAGCL, config_file_list=[args.config], config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MAGCL(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)  # stock RecBole, unmodified

    experiment = args.experiment or f"MAGCL-{config['dataset']}"
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
        mlflow.log_params(overrides)  # explicit CLI overrides, recorded again in case any key above missed one

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=not args.quiet,
            callback_fn=valid_score_callback(config["valid_metric"]),
        )
        log_train_loss_history(trainer)
        mlflow.log_metrics({f"best_valid_{sanitize_metric_name(k)}": float(v) for k, v in best_valid_result.items()})

        test_result = trainer.evaluate(test_data, show_progress=not args.quiet)
        mlflow.log_metrics({f"test_{sanitize_metric_name(k)}": float(v) for k, v in test_result.items()})

        print("\nbest valid:", dict(best_valid_result))
        print("test:", dict(test_result))


if __name__ == "__main__":
    main()

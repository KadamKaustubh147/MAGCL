"""Train SimGCL (RecBole-GNN, vendored in src/recbole_gnn) once and log to MLflow -- same CLI and same final
'best valid:' / 'test:' output lines as scripts/run.py, so scripts/benchmark_seeds.py can drive it with
--script scripts/run_simgcl.py.

    python scripts/run_simgcl.py --config configs/simgcl.yaml
    python scripts/run_simgcl.py --config configs/simgcl_ifashion.yaml --override '{"seed": 2021}'
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import mlflow

from magcl import compat  # noqa: F401  (scipy/torch.load patches before recbole import)
from magcl.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.utils import init_seed
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

_HPARAM_KEYS = ["n_layers", "reg_weight", "lambda", "eps", "temperature",
                "embedding_size", "learning_rate", "train_batch_size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simgcl.yaml")
    parser.add_argument("--override", default="{}", help="JSON dict of config overrides, inline or a path to a .json file")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: SimGCL-<dataset>)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quiet", action="store_true", help="hide tqdm progress bars")
    args = parser.parse_args()

    override_path = Path(args.override)
    overrides = json.loads(override_path.read_text() if override_path.is_file() else args.override)
    config = Config(model="SimGCL", config_file_list=[args.config], config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model("SimGCL")(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], "SimGCL")(config, model)

    mlflow.set_experiment(args.experiment or f"SimGCL-{config['dataset']}")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
        mlflow.log_params(overrides)

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

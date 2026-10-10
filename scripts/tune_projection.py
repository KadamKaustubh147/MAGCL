"""Bayesian search for MAGCL WITH a projection head (src/magcl_projection):
Ray Tune (OptunaSearch) driving many full RecBole runs, each its own MLflow run.
Adapted from scripts/tune.py -- same resume/--round/seeding behaviour (see that
file's docstring); only what differs is explained here.

Usage:
    python scripts/tune_projection.py --num-samples 30
    python scripts/tune_projection.py --num-samples 30 --no-projection   # CONTROL: same search, head off

What is searched: the SAME space as scripts/tune.py -- lam1-lam4 in
loguniform(1e-9, 1e-3), theta in uniform(0.02, 0.1) -- so results are directly
comparable to the earlier no-head sweep. Everything else (tau, normalize_cl, head
settings) is held fixed from --fixed (configs/projection_ml1m_normcl.json by
default; any lam/theta keys in it are ignored because they are searched), so the
ON and OFF sweeps differ in exactly one thing: whether the head exists.

Scale note: _info_nce returns a SUM over the ~8192 batch nodes. Measured at
initialisation on ML-1M with normalize_cl=true and lam=1, the CL term is ~1e5
(1.25e5 with the head) versus BPR (a mean) = 0.69, so the bottom ~2 decades of
lam1/lam2 are effectively "CL off" and lam ~5e-6 puts CL on BPR's scale.

The ON and OFF sweeps log to separate MLflow experiments (suffix -on / -off), so
Optuna's seeding never mixes their rewards.
"""

import argparse
import json
from pathlib import Path

import mlflow
import ray
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from magcl_projection import compat  # noqa: F401  (scipy patch before recbole import)
from magcl_projection.magcl_projection_network import MAGCL
from magcl_projection.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASE_CONFIG = str(_PROJECT_ROOT / "configs" / "magcl.yaml")
_FIXED_JSON = str(_PROJECT_ROOT / "configs" / "projection_ml1m_normcl.json")
_MLFLOW_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"  # absolute: Ray chdirs every trial (see scripts/tune.py)
_DATA_PATH = str(_PROJECT_ROOT / "data")                  # absolute, same reason

# Identical to scripts/tune.py's SEARCH_SPACE on purpose: same range, same five
# parameters, tau/gamma/alpha/L/beta/S fixed from the base config -- so these
# sweeps are directly comparable to the earlier no-head results.
SEARCH_SPACE = {
    "lam1": tune.loguniform(1e-9, 1e-3),
    "lam2": tune.loguniform(1e-9, 1e-3),
    "lam3": tune.loguniform(1e-9, 1e-3),
    "lam4": tune.loguniform(1e-9, 1e-3),
    "theta": tune.uniform(0.02, 0.1),
}

_HPARAM_KEYS = list(SEARCH_SPACE) + [
    "L", "tau", "gamma", "alpha", "beta", "S", "embedding_size", "normalize_cl",
]
# Head switches the model defaults when a YAML omits them -- read with a fallback, not config[k].
_PROJ_DEFAULTS = {"use_projection": True, "proj_type": "mlp", "proj_hidden": None, "proj_out": None, "reg_proj": False}


def _load_seed_points(experiment_name: str, metric_key: str):
    """Previously FINISHED trials of THIS experiment (earlier --round sweeps) as
    (points, rewards) for OptunaSearch. Never borrows from another experiment."""
    mlflow.set_tracking_uri(_MLFLOW_URI)
    exp = MlflowClient().get_experiment_by_name(experiment_name)
    points, rewards = [], []
    if exp is not None:
        for run in MlflowClient().search_runs([exp.experiment_id], filter_string="status = 'FINISHED'"):
            reward = run.data.metrics.get(metric_key)
            if reward is None or not set(SEARCH_SPACE) <= set(run.data.params):
                continue
            points.append({k: float(run.data.params[k]) for k in SEARCH_SPACE})
            rewards.append(reward)
    return points, rewards


def train_trial(trial_config: dict, fixed: dict, experiment_name: str, metric: str, base_config: str):
    """One Ray Tune trial: sampled lam1-4/theta on top of the fixed overrides."""
    mlflow.set_tracking_uri(_MLFLOW_URI)
    config = Config(
        model=MAGCL, config_file_list=[base_config],
        config_dict={**fixed, **trial_config, "data_path": _DATA_PATH},
    )
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MAGCL(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)  # stock RecBole, unmodified

    log_valid_score = valid_score_callback(metric)

    def report_each_epoch(epoch_idx, valid_score):
        log_valid_score(epoch_idx, valid_score)
        tune.report({metric: valid_score, "epoch": epoch_idx})

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
        mlflow.log_params({k: (config[k] if k in config else d) for k, d in _PROJ_DEFAULTS.items()})
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=False, callback_fn=report_each_epoch
        )
        log_train_loss_history(trainer)
        mlflow.log_metrics({f"best_valid_{sanitize_metric_name(k)}": float(v) for k, v in best_valid_result.items()})

        test_result = trainer.evaluate(test_data)
        mlflow.log_metrics({f"test_{sanitize_metric_name(k)}": float(v) for k, v in test_result.items()})

    tune.report({metric: best_valid_score})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_BASE_CONFIG)
    parser.add_argument("--fixed", default=_FIXED_JSON,
                         help="JSON of hyperparameters held fixed (lam1-4/theta in it are ignored: they are searched)")
    parser.add_argument("--no-projection", action="store_true",
                         help="CONTROL sweep: identical search with use_projection=false")
    parser.add_argument("--num-samples", type=int, default=30, help="number of trials")
    parser.add_argument("--gpu-per-trial", type=float, default=1.0)
    parser.add_argument("--cpu-per-trial", type=float, default=2.0)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--round", type=int, default=1,
                         help="bump to add trials after a round fully finished (see scripts/tune.py)")
    parser.add_argument("--object-store-mb", type=int, default=200)
    parser.add_argument("--ray-memory-mb", type=int, default=300)
    args = parser.parse_args()
    args.config = str(Path(args.config).resolve())  # absolute: Ray chdirs each trial

    fixed = {k: v for k, v in json.loads(Path(args.fixed).read_text()).items() if k not in SEARCH_SPACE}
    if args.no_projection:
        fixed["use_projection"] = False

    base_config = Config(model=MAGCL, config_file_list=[args.config])
    dataset_name, metric = base_config["dataset"], base_config["valid_metric"]
    suffix = "off" if args.no_projection else "on"
    experiment_name = args.experiment or f"MAGCL-proj-tuning-{dataset_name}-{suffix}"
    print(f"Experiment '{experiment_name}'  fixed={fixed}")

    ray.init(
        include_dashboard=False,
        object_store_memory=args.object_store_mb * 1024 * 1024,
        _memory=args.ray_memory_mb * 1024 * 1024,
    )
    trainable = tune.with_resources(
        lambda trial_config: train_trial(trial_config, fixed, experiment_name, metric, args.config),
        resources={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
    )
    storage_path = str(_PROJECT_ROOT / "ray_results")
    round_dir = f"{experiment_name}-r{args.round}"
    run_config = tune.RunConfig(name=round_dir, storage_path=storage_path)

    if tune.Tuner.can_restore(f"{storage_path}/{round_dir}"):
        print(f"Found an unfinished round at {storage_path}/{round_dir} -- resuming it.")
        tuner = tune.Tuner.restore(f"{storage_path}/{round_dir}", trainable, resume_errored=True)
    else:
        metric_key = f"best_valid_{sanitize_metric_name(metric)}"
        seed_points, seed_rewards = _load_seed_points(experiment_name, metric_key)
        if seed_points:
            print(f"Starting round {args.round}, seeded with {len(seed_points)} known trial(s).")
            search_alg = OptunaSearch(points_to_evaluate=seed_points, evaluated_rewards=seed_rewards)
        else:
            print(f"Starting round {args.round}: no prior trials for '{experiment_name}' -- exploring blind.")
            search_alg = OptunaSearch()
        tuner = tune.Tuner(
            trainable,
            param_space=SEARCH_SPACE,
            run_config=run_config,
            tune_config=tune.TuneConfig(search_alg=search_alg, metric=metric, mode="max", num_samples=args.num_samples),
        )
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode="max")
    print(f"\nBest trial config: {best.config}")
    print(f"Best trial {metric}: {best.metrics.get(metric)}")


if __name__ == "__main__":
    main()

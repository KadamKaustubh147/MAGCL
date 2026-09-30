"""Bayesian hyperparameter search for MAGCL: Ray Tune (OptunaSearch) driving
many full RecBole training runs, each one its own MLflow run.

Usage:
    python scripts/tune.py --num-samples 40
    python scripts/tune.py --num-samples 40 --gpu-per-trial 0.34   # 3 trials sharing the GPU

If this crashes or the machine loses power MID-sweep, just rerun the exact
same command (same --experiment, same --round): it auto-detects the
unfinished run under ./ray_results/<experiment_name>-r<round> and resumes it
-- completed trials are kept and Optuna's search state is restored, so it
picks up the search rather than starting over. --num-samples in that case
means "total trials for this round", not "additional trials".

To run MORE trials AFTER a round has fully finished (all its trials reached
TERMINATED), bump --round (e.g. --round 2): this starts a fresh Ray run under
a new ./ray_results/<experiment_name>-r2 directory (Tuner.restore() on an
already-complete round would find nothing left to do), but seeds Optuna with
every previously FINISHED trial from ALL earlier rounds in this --experiment
(read back from MLflow via points_to_evaluate/evaluated_rewards), so the
search continues where it left off instead of exploring blind again.

Then: mlflow ui   (from the project root) -> "MAGCL-tuning-<dataset>" experiment
-> open any trial -> per-epoch loss and Recall/NDCG@K charts (same
magcl.tracking.MlflowTrainer as scripts/run.py, so every trial is fully
inspectable, not just its final score).

--gpu-per-trial < 1.0 runs multiple trials concurrently on one physical GPU
(Ray schedules them onto the same card; VRAM is NOT reserved per trial, so
this only works if the model is small enough for several copies to fit --
MAGCL on ML-1M easily qualifies, using well under 1 GB of VRAM). The real
constraint is system RAM, not VRAM: one trial process measured ~1.8 GB RSS
(Python + PyTorch + RecBole + the loaded interaction data/graphs), and that
cost is per CONCURRENT trial, not per GPU. Default is 1.0 (one trial at a
time, safe regardless of free RAM); only raise it once `free RAM > 1.8 GB *
number of concurrent trials you want` (check with Get-CimInstance
Win32_OperatingSystem on Windows) -- otherwise the OS will start swapping and
trials will slow down or fail rather than actually run in parallel.
"""

import argparse
from pathlib import Path

import mlflow
import ray
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from magcl import compat  # noqa: F401  (scipy patch before recbole import)
from magcl.model import MAGCL
from magcl.tracking import log_train_loss_history, sanitize_metric_name, valid_score_callback
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASE_CONFIG = str(_PROJECT_ROOT / "configs" / "magcl.yaml")
_MLFLOW_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"
# MLflow's fluent API, with no tracking URI set, defaults to sqlite:///mlflow.db
# resolved against the CURRENT WORKING DIRECTORY -- and Ray chdirs every trial
# into its own isolated working directory, so without this, each trial creates
# and writes to its OWN separate mlflow.db instead of the shared project one
# (confirmed: found one mlflow.db per trial folder under Ray's session temp
# dir). Every trial must set this explicitly, same fix as _DATA_PATH below.

# magcl.yaml's data_path is the relative "./data/", resolved against the
# CURRENT WORKING DIRECTORY when RecBole reads it -- fine for scripts/run.py
# (run from the project root), but Ray workers run with a different cwd of
# their own, so the same relative path resolves to nowhere inside a trial
# and crashes with "Neither [./data/...] exists...". Every trial overrides
# it with this absolute path instead.
_DATA_PATH = str(_PROJECT_ROOT / "data")

# Only lam1-4 (Eq 30 weights) and theta (denoising threshold, Eq 8) are swept.
# tau, gamma, alpha, L, beta, S are fixed at the paper's stated/best-on-ML-1M
# values from configs/magcl.yaml (§5.2), not tuned here.
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

def _load_seed_points(experiment_name: str, metric_key: str):
    """Every previously FINISHED trial logged to THIS MLflow experiment
    (across all earlier --round sweeps of the SAME dataset), as (points,
    rewards) for OptunaSearch's points_to_evaluate/evaluated_rewards -- so a
    fresh Tuner still benefits from every trial already run on this dataset
    instead of exploring blind.

    Deliberately does NOT fall back to another dataset's results when this
    experiment has no history yet (e.g. the first round on a brand-new
    dataset): a reward is only meaningful on the scale of the dataset it was
    measured on. ML-1M's recall@20 sits around 0.27, but Alibaba-iFashion is
    far sparser (density 0.00007 vs. 0.03816 -- Table 1) and the paper's own
    Table 2 puts MAGCL's recall@20 there around 0.07 -- an order of magnitude
    lower. Seeding a new dataset's search with evaluated_rewards from a
    different reward distribution would miscalibrate Optuna's TPE sampler
    about what "good" looks like, doing more harm than starting blind.
    (A good hyperparameter *point* can still transfer across datasets --
    that's why the iFashion config's static lam1-4/theta defaults reuse our
    best ML-1M values as a warm start; only the reward doesn't transfer.)"""
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


def train_trial(trial_config: dict, experiment_name: str, metric: str, base_config: str):
    """One Ray Tune trial: build a MAGCL config from the trial's sampled
    hyperparameters, train with the paper's own early stopping, log every
    epoch to its own MLflow run, and report back to Ray/Optuna."""
    mlflow.set_tracking_uri(_MLFLOW_URI)  # see _MLFLOW_URI's comment: must be absolute
    config = Config(
        model=MAGCL, config_file_list=[base_config],
        config_dict={**trial_config, "data_path": _DATA_PATH},
    )
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MAGCL(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)  # stock RecBole, unmodified

    log_valid_score = valid_score_callback(metric)

    def report_each_epoch(epoch_idx, valid_score):
        # lets Ray's live trial table + Optuna's pruning (if ever enabled)
        # see progress before the trial finishes, not just its final score
        log_valid_score(epoch_idx, valid_score)
        tune.report({metric: valid_score})

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({k: config[k] for k in _HPARAM_KEYS})
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
    parser.add_argument("--config", default=str(_BASE_CONFIG), help="base config, e.g. configs/magcl_ifashion.yaml")
    parser.add_argument("--num-samples", type=int, default=40, help="number of trials")
    parser.add_argument("--gpu-per-trial", type=float, default=1.0,
                         help="fraction of the GPU per trial; <1.0 runs several trials at once (see module docstring)")
    parser.add_argument("--cpu-per-trial", type=float, default=2.0)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--round", type=int, default=1,
                         help="bump this to add more trials after a previous round fully finished "
                              "(see module docstring); trials from every earlier round in the same "
                              "--experiment are used to seed the new round's search")
    parser.add_argument("--object-store-mb", type=int, default=200,
                         help="Ray's object store size; this machine has very little free RAM, keep this small")
    parser.add_argument("--ray-memory-mb", type=int, default=300,
                         help="Ray's own task/actor memory accounting (soft, not an OS-enforced cap)")
    args = parser.parse_args()
    # Absolute, for the same reason _DATA_PATH is absolute: Ray chdirs every
    # trial into its own working directory before running it, so a relative
    # --config path resolves fine here (before any chdir) but breaks inside
    # every actual trial.
    args.config = str(Path(args.config).resolve())

    base_config = Config(model=MAGCL, config_file_list=[args.config])
    dataset_name, metric = base_config["dataset"], base_config["valid_metric"]
    experiment_name = args.experiment or f"MAGCL-tuning-{dataset_name}"

    ray.init(
        include_dashboard=False,
        object_store_memory=args.object_store_mb * 1024 * 1024,
        _memory=args.ray_memory_mb * 1024 * 1024,
    )
    trainable = tune.with_resources(
        lambda trial_config: train_trial(trial_config, experiment_name, metric, args.config),
        resources={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
    )
    # Fixed, predictable storage path per (experiment_name, round) -- not a
    # timestamp -- so a rerun after a crash/power-cut can find and resume
    # THIS round instead of silently starting a fresh one that forgets every
    # trial Optuna already completed within it.
    storage_path = str(_PROJECT_ROOT / "ray_results")
    round_dir = f"{experiment_name}-r{args.round}"
    run_config = tune.RunConfig(name=round_dir, storage_path=storage_path)

    if tune.Tuner.can_restore(f"{storage_path}/{round_dir}"):
        print(f"Found an unfinished round at {storage_path}/{round_dir} -- resuming it "
              "(completed trials are kept, Optuna's search state is restored).")
        tuner = tune.Tuner.restore(
            f"{storage_path}/{round_dir}", trainable, resume_errored=True,
        )
    else:
        metric_key = f"best_valid_{sanitize_metric_name(metric)}"
        seed_points, seed_rewards = _load_seed_points(experiment_name, metric_key)
        if seed_points:
            print(f"Starting round {args.round}, seeded with {len(seed_points)} trial(s) "
                  f"already known from '{experiment_name}'.")
            search_alg = OptunaSearch(points_to_evaluate=seed_points, evaluated_rewards=seed_rewards)
        else:
            print(f"Starting round {args.round}: no prior trials for '{experiment_name}' -- exploring blind.")
            search_alg = OptunaSearch()
        tuner = tune.Tuner(
            trainable,
            param_space=SEARCH_SPACE,
            run_config=run_config,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                metric=metric,
                mode="max",
                num_samples=args.num_samples,
            ),
        )
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode="max")
    print(f"\nBest trial config: {best.config}")
    print(f"Best trial {metric}: {best.metrics.get(metric)}")


if __name__ == "__main__":
    main()

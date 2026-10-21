"""MLflow logging helpers for the STOCK, unmodified recbole.trainer.Trainer --
no subclassing, no overriding. Both hooks used here are things Trainer already
exposes publicly:

  callback_fn        an actual parameter of Trainer.fit(), called every epoch
                      with (epoch_idx, valid_score) -- RecBole's own extension
                      point, not something we added.
  trainer.train_loss_dict   a plain instance attribute the stock fit() fills
                      in itself (one total-loss float per epoch); we just read
                      it back after fit() returns.

Trade-off, for the record: because nothing is overridden, we only get the ONE
configured valid_metric (e.g. recall@20) as a per-epoch chart, plus the TOTAL
training loss per epoch (bpr+cl+layer+unif+reg summed -- fit() itself collapses
the tuple before storing it, so the five parts can't be told apart afterwards).
The full Recall/NDCG@10/20/50 breakdown is still logged, just as single
snapshots (best-valid-epoch and final-test), not a per-epoch history.
"""

import mlflow


def sanitize_metric_name(name: str) -> str:
    """RecBole's metric keys look like 'recall@10'; MLflow metric names may
    only contain alphanumerics/underscore/dash/period/space/slash -- '@' is
    rejected outright. Used everywhere a RecBole metric name reaches MLflow."""
    return name.replace("@", "_at_")


def valid_score_callback(metric_name: str):
    """Returns a callback_fn for Trainer.fit(callback_fn=...) that logs the
    configured valid_metric to MLflow every epoch."""
    key = f"valid_{sanitize_metric_name(metric_name)}"

    def _callback(epoch_idx, valid_score):
        mlflow.log_metric(key, float(valid_score), step=epoch_idx)

    return _callback


def log_train_loss_history(trainer) -> None:
    """Call once after trainer.fit() returns: backfills the per-epoch total
    training loss from Trainer's own public train_loss_dict."""
    for epoch_idx, loss in trainer.train_loss_dict.items():
        mlflow.log_metric("train_loss_total", float(loss), step=epoch_idx)

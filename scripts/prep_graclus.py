"""Write the TRAIN-split user-item graph in Graclus's format, with RecBole's own ids.

    python scripts/prep_graclus.py                          # configs/magcl.yaml (ML-1M)
    python scripts/prep_graclus.py --config configs/magcl_ifashion.yaml

Then cluster it with Graclus (WSL), K is a hyperparameter:
    cd /mnt/c/dev_stuff/MAGCL_claude/graclus1.2
    ./graclus ../outputs/graclus/<dataset>_train.graph 16        # -> <dataset>_train.graph.part.16

Why not reuse scripts/to_graclus.py's file: that one reads the FULL interaction file, which
includes validation/test edges. MAGCL builds every graph from the train split only, so clusters
used by the model must be computed on the train split too -- otherwise test edges leak into
the high-order graph. This script goes through the same Config -> create_dataset ->
data_preparation path as scripts/run.py, so the split (seed 2020) and the ids are identical.

Outputs (outputs/graclus/):
    <dataset>_train.graph     Graclus adjacency list; nodes = users with edges, then items with edges
    <dataset>_train.map.npz   RecBole ids of those nodes + an edge fingerprint the model verifies
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent

from magcl_graclus import compat  # noqa: E402,F401  (scipy patch before recbole import)
from magcl_graclus.graphs import edge_fingerprint  # noqa: E402
from magcl_graclus.magcl_graclus import MAGCL  # noqa: E402
from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/magcl.yaml"], help="one or more configs, later ones override")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "graclus"))
    args = ap.parse_args()

    config = Config(model=MAGCL, config_file_list=args.config)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    train = train_data.dataset
    m, n = train.num(config["USER_ID_FIELD"]), train.num(config["ITEM_ID_FIELD"])
    R = train.inter_matrix(form="csr")[:m, :n]                 # exactly what the model's __init__ uses
    R.data[:] = 1.0

    users = np.where(np.asarray(R.sum(axis=1)).ravel() > 0)[0]   # drops RecBole's id-0 padding / cold nodes
    items = np.where(np.asarray(R.sum(axis=0)).ravel() > 0)[0]
    Rs = R[users][:, items]
    A = sp.bmat([[None, Rs], [Rs.T, None]], format="csr")
    A.sort_indices()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{config['dataset']}_train"
    with open(out / f"{name}.graph", "w", newline="\n") as f:
        f.write(f"{len(users) + len(items)} {Rs.nnz}\n")
        for r in range(len(users) + len(items)):
            f.write(" ".join(map(str, A.indices[A.indptr[r]:A.indptr[r + 1]] + 1)) + "\n")
    np.savez(out / f"{name}.map.npz", user_ids=users, item_ids=items,
             n_users=m, n_items=n, fingerprint=edge_fingerprint(R))
    print(f"{name}: {len(users)} users + {len(items)} items, {Rs.nnz} train edges "
          f"(of {m} x {n} RecBole ids) -> {out / (name + '.graph')}")
    print(f"\nNext (WSL):  cd /mnt/c/dev_stuff/MAGCL_claude/graclus1.2 && ./graclus ../outputs/graclus/{name}.graph 16")


if __name__ == "__main__":
    sys.exit(main())

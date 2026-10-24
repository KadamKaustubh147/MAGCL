"""How much does a Graclus clustering actually change MAGCL's high-order graph?

    python scripts/graclus_check.py --map outputs/graclus/ml-1m_preprocessed_byrating_train.map.npz \
                                    --part graclus1.2/ml-1m_preprocessed_byrating_train.graph.part.16

Builds the plain Jaccard top-S graph (Eqs 2-3) and the cluster-restricted one from the SAME train split
the model uses, and compares them. This is the go/no-go number for graclus_mode=replace/union: if almost
every plain Jaccard edge already stays inside a cluster, restriction changes almost nothing and the
model cannot differ from baseline MAGCL. (The model's own logger.info line with this number is not
reliable -- scripts/run*.py never call RecBole's init_logger, so logger.info goes nowhere.)
"""

import argparse
from pathlib import Path

import numpy as np

from magcl_graclus import compat  # noqa: F401  (scipy patch before recbole import)
from magcl_graclus.graphs import high_order_binary
from magcl_graclus.magcl_graclus import MAGCL
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/magcl.yaml")
    ap.add_argument("--map", required=True, help="<dataset>_train.map.npz from scripts/prep_graclus.py")
    ap.add_argument("--part", required=True, help="Graclus output, e.g. ...graph.part.16")
    ap.add_argument("--s-gr", type=int, default=None, help="top-S among cluster-mates (default: the config's S)")
    args = ap.parse_args()

    config = Config(model=MAGCL, config_file_list=[args.config])
    init_seed(config["seed"], config["reproducibility"])
    train = data_preparation(config, create_dataset(config))[0].dataset
    m, n = train.num(config["USER_ID_FIELD"]), train.num(config["ITEM_ID_FIELD"])
    R = train.inter_matrix(form="csr")[:m, :n]
    R.data[:] = 1.0

    ul, il = MAGCL._load_graclus_labels({"graclus_map": args.map, "graclus_part": args.part}, R)
    beta, s = config["beta"], config["S"]
    s_gr = args.s_gr or s
    plain = high_order_binary(R, beta, s)
    restricted = high_order_binary(R, beta, s_gr, ul, il)

    sizes = np.bincount(np.concatenate([ul[ul >= 0], il[il >= 0]]))
    print(f"clusters: {len(sizes)}  nodes per cluster min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
    print(f"beta={beta}  S={s}  S_gr={s_gr}\n")
    print(f"{'side':<7}{'plain edges':>12}{'inside cluster':>16}{'removed':>10}{'added':>9}{'nodes changed':>16}")
    for name, sl, labels in (("users", slice(0, m), ul), ("items", slice(m, m + n), il)):
        P, Q = plain[sl, sl].tocsr(), restricted[sl, sl].tocsr()
        coo = P.tocoo()
        inside = float((labels[coo.row] == labels[coo.col]).mean())
        both = P.multiply(Q).nnz
        has_edges = np.unique(coo.row)
        changed = np.unique((P != Q).nonzero()[0])
        print(f"{name:<7}{P.nnz:>12}{inside:>15.1%}{P.nnz - both:>10}{Q.nnz - both:>9}{len(changed) / len(has_edges):>15.1%}")

    inside_all = float(np.mean([(np.concatenate([ul, il])[r] == np.concatenate([ul, il])[c]).mean()
                                for r, c in [plain.nonzero()]]))
    print(f"\nall plain Jaccard edges inside one cluster: {inside_all:.1%}")
    print("verdict:", "restriction removes almost nothing -> expect a tie with baseline MAGCL" if inside_all > 0.95 else
          "restriction really changes the graph -> a different result from baseline is possible")


if __name__ == "__main__":
    main()

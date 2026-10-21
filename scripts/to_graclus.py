"""Write the user-item bipartite graph in Graclus's adjacency-list format.

    python scripts/to_graclus.py            # ML-1M -> outputs/graclus/ml1m.graph
    python scripts/to_graclus.py ifashion   # iFashion train split

Node numbering (1-based, as Graclus wants): users are 1..m, items are m+1..m+n.
Header line is "<#nodes> <#undirected edges>"; line k lists the neighbours of node k.
<name>.ids.npz keeps the original user/item ids so Graclus's output (one cluster
label per line, in node order) can be mapped back.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1] if len(sys.argv) > 1 else "ml1m"

if name == "ml1m":
    df = pd.read_csv(ROOT / "data/ml-1m_preprocessed_byrating/ml-1m_preprocessed_byrating.inter",
                     sep="\t", usecols=[0, 1], names=["u", "i"], header=0)
else:  # iFashion train split: "user,item" per line, no header
    df = pd.read_csv(ROOT / "data/Alibaba-iFashion/ifashion.train", names=["u", "i"], header=None)
df = df.drop_duplicates()

u, user_ids = pd.factorize(df["u"])
i, item_ids = pd.factorize(df["i"])
m, n = len(user_ids), len(item_ids)
R = sp.csr_matrix((np.ones(len(df), dtype=np.int8), (u, i)), shape=(m, n))
A = sp.bmat([[None, R], [R.T, None]], format="csr")
A.sort_indices()

out = ROOT / "outputs" / "graclus"
out.mkdir(parents=True, exist_ok=True)
with open(out / f"{name}.graph", "w", newline="\n") as f:
    f.write(f"{m + n} {R.nnz}\n")
    for r in range(m + n):
        f.write(" ".join(map(str, A.indices[A.indptr[r]:A.indptr[r + 1]] + 1)) + "\n")
np.savez(out / f"{name}.ids.npz", user_ids=user_ids.to_numpy(), item_ids=item_ids.to_numpy())
print(f"{name}: {m} users + {n} items = {m + n} nodes, {R.nnz} edges -> {out / (name + '.graph')}")

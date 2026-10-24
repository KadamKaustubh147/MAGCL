"""Visualise the user-item graph with Graclus clusters + user-user / item-item co-occurrence stats.

Needs the two files from the Graclus workflow:
    python scripts/to_graclus.py                                   -> outputs/graclus/ml1m.graph
    (WSL) ./graclus ../outputs/graclus/ml1m.graph 8                -> ml1m.graph.part.8

Usage:
    python scripts/graph_viz.py --part graclus1.2/ml1m.graph.part.8
    python scripts/graph_viz.py --part <file> --name ml1m --out outputs/graph_viz

Co-occurrence is MAGCL's own Eq 2 quantity: for two items, |N(x) & N(y)| = users who
touched both; JSC = |N(x) & N(y)| / |N(x) | N(y)|; same for two users over item sets.
The graph is read back from the .graph file Graclus itself consumed (users are
nodes 1..m, items m+1..m+n), so labels always line up with the nodes.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
BETA = 0.8   # MAGCL's Jaccard keep-threshold (configs/magcl.yaml)


def read_graph(path: Path, n_users: int) -> sp.csr_matrix:
    """Biadjacency R (users x items) from Graclus's adjacency-list file."""
    with open(path) as f:
        n_nodes, n_edges = map(int, f.readline().split()[:2])
        rows, cols = [], []
        for node in range(n_users):                       # only the user lines are needed
            nbrs = np.fromstring(f.readline(), dtype=np.int64, sep=" ") - n_users - 1
            rows.append(np.full(len(nbrs), node)); cols.append(nbrs)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_nodes - n_users))
    assert R.nnz == n_edges, f"edge count mismatch: {R.nnz} vs header {n_edges}"
    return R


def side_stats(R: sp.csr_matrix, labels: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """Co-occurrence/Jaccard stats for the ROWS of R (pass R.T for the other side),
    self-pairs and zero-overlap pairs excluded as in graphs._jaccard_side. `labels` are
    Graclus clusters of these rows, used for the within- vs across-cluster comparison.
    `rows` = a sample of row ids to evaluate (each still compared against ALL rows); None = all.
    Needed at iFashion scale (300k users): the full k x k pass would take hours."""
    k = R.shape[0]
    rows = np.arange(k) if rows is None else rows
    chunk = max(50, min(400, int(3e7 // k)))               # keep the dense chunk x k block ~240 MB
    deg = np.asarray(R.sum(axis=1)).ravel()
    Rt = R.T.tocsc()
    edges = np.linspace(0, np.log10(max(deg.max(), 2)), 61)
    h_cnt, h_jac = np.zeros(60), np.zeros(100)
    n_pairs = n_beta = 0
    sum_in = sum_out = 0.0
    pairs_in = pairs_out = 0
    max_jac = np.zeros(len(rows))
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        e, ar = s + len(r), np.arange(len(r))
        inter = (R[r] @ Rt).toarray()
        inter[ar, r] = 0
        jac = np.where(inter > 0, inter / (deg[r, None] + deg[None, :] - inter), 0.0)
        same = labels[r, None] == labels[None, :]
        same[ar, r] = False
        diff = labels[r, None] != labels[None, :]
        sum_in += float(inter[same].sum()); pairs_in += int(same.sum())     # int(): np.int32 overflows past 2.1e9 pairs on Windows
        sum_out += float(inter[diff].sum()); pairs_out += int(diff.sum())
        mask = inter > 0
        v, j = inter[mask], jac[mask]
        n_pairs += v.size; n_beta += int((j >= BETA).sum())
        h_cnt += np.histogram(np.log10(v), edges)[0]
        h_jac += np.histogram(j, np.linspace(0, 1, 101))[0]
        max_jac[s:e] = jac.max(axis=1)
    return {
        "nodes": k, "rows_evaluated": len(rows),
        "frac_pairs_with_any_overlap": n_pairs / (len(rows) * (k - 1)),
        "mean_overlap_over_all_pairs_same_cluster": sum_in / max(pairs_in, 1),
        "mean_overlap_over_all_pairs_different_cluster": sum_out / max(pairs_out, 1),
        "pairs_jaccard_ge_beta": n_beta,
        "nodes_with_a_partner_jaccard_ge_beta": float((max_jac >= BETA).mean()),
        "nodes_with_no_overlap_at_all": int((max_jac == 0).sum()),
        "max_jaccard_per_node_pctiles_5_25_50_75_95": np.percentile(max_jac, [5, 25, 50, 75, 95]).round(3).tolist(),
        "_edges": edges, "_h_cnt": h_cnt, "_h_jac": h_jac,
    }


def binned(order: np.ndarray, n_bins: int) -> sp.csr_matrix:
    """(bins x n) 0/1 matrix grouping already-ordered nodes into contiguous bins."""
    n = len(order)
    n_bins = min(n_bins, n)
    return sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n) * n_bins // n, order)), shape=(n_bins, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, help="Graclus output, e.g. graclus1.2/ml1m.graph.part.8")
    ap.add_argument("--name", default="ml1m")
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=10000,
                    help="co-occurrence stats use at most this many randomly sampled nodes per side "
                         "(each compared against ALL nodes); ML-1M is below it, so exact there")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "graph_viz"))
    args = ap.parse_args()
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    gdir = ROOT / "outputs" / "graclus"   # to_graclus.py writes <name>.ids.npz, prep_graclus.py writes <name>.map.npz
    ids = np.load(next(p for p in (gdir / f"{args.name}.ids.npz", gdir / f"{args.name}.map.npz") if p.exists()))
    m, n = len(ids["user_ids"]), len(ids["item_ids"])
    R = read_graph(ROOT / "outputs" / "graclus" / f"{args.name}.graph", m)
    labels = np.loadtxt(args.part, dtype=int)
    assert len(labels) == m + n, f"{args.part} has {len(labels)} labels, graph has {m + n} nodes"
    ucl, icl = labels[:m], labels[m:]
    k = int(labels.max()) + 1
    udeg, ideg = np.asarray(R.sum(1)).ravel(), np.asarray(R.sum(0)).ravel()

    # ---- stats ----
    coo = R.tocoo()
    intra = labels[:m][coo.row] == icl[coo.col]
    clusters = [{"cluster": c, "users": int((ucl == c).sum()), "items": int((icl == c).sum()),
                 "edges_inside": int(intra[(ucl[coo.row] == c)].sum()),
                 "edges_leaving": int((~intra[(ucl[coo.row] == c)]).sum() + ((~intra) & (icl[coo.col] == c)).sum())}
                for c in range(k)]
    stats = {"users": m, "items": n, "interactions": int(R.nnz), "density": R.nnz / (m * n), "clusters": k,
             "fraction_of_interactions_inside_a_cluster": float(intra.mean()),
             "user_degree": {"mean": float(udeg.mean()), "median": float(np.median(udeg)), "max": float(udeg.max())},
             "item_degree": {"mean": float(ideg.mean()), "median": float(np.median(ideg)), "max": float(ideg.max())},
             "per_cluster": clusters}
    rng = np.random.default_rng(args.seed)
    sample = lambda k: None if k <= args.max_rows else np.sort(rng.choice(k, args.max_rows, replace=False))  # noqa: E731
    print("user-user ..."); us = side_stats(R, ucl, sample(m))
    print("item-item ..."); its = side_stats(R.T.tocsr(), icl, sample(n))
    stats["user_user"] = {a: b for a, b in us.items() if not a.startswith("_")}
    stats["item_item"] = {a: b for a, b in its.items() if not a.startswith("_")}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps({a: b for a, b in stats.items() if a != "per_cluster"}, indent=2, default=float))
    for c in clusters:
        print(c)

    # ---- fig 1: degrees ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, d, nm in zip(ax, (udeg, ideg), ("user", "item")):
        a.hist(d, bins=np.logspace(0, np.log10(d.max() + 1), 50), color="#4c78a8")
        a.set(xscale="log", yscale="log", xlabel=f"{nm} degree", ylabel="count", title=f"{nm} degree distribution")
    fig.tight_layout(); fig.savefig(out / "1_degrees.png", dpi=140); plt.close(fig)

    # ---- fig 2: co-occurrence and Jaccard distributions ----
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for r, (nm, s) in enumerate((("user-user", us), ("item-item", its))):
        e = s["_edges"]
        ax[r, 0].bar(10 ** e[:-1], s["_h_cnt"], width=np.diff(10 ** e), align="edge", color="#59a14f")
        ax[r, 0].set(xscale="log", yscale="log", xlabel="co-occurrence |N(x)&N(y)|", ylabel="# pairs", title=f"{nm}: overlap size")
        ax[r, 1].bar(np.linspace(0, 1, 101)[:-1], s["_h_jac"], width=0.01, align="edge", color="#e15759")
        ax[r, 1].axvline(BETA, color="k", ls="--", label=f"beta={BETA}")
        ax[r, 1].set(yscale="log", xlabel="Jaccard (Eq 2)", ylabel="# pairs", title=f"{nm}: Jaccard"); ax[r, 1].legend()
    fig.tight_layout(); fig.savefig(out / "2_cooccurrence_hist.png", dpi=140); plt.close(fig)

    # ---- fig 3: interaction matrix, degree-sorted vs Graclus-cluster-sorted (binned edge density) ----
    uord, iord = np.lexsort((-udeg, ucl)), np.lexsort((-ideg, icl))

    def density(uo, io):
        Bu, Bi = binned(uo, args.bins), binned(io, args.bins)
        return (Bu @ R @ Bi.T).toarray() / np.outer(np.asarray(Bu.sum(1)).ravel(), np.asarray(Bi.sum(1)).ravel())
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, (uo, io, t) in zip(ax, ((np.argsort(-udeg), np.argsort(-ideg), "sorted by degree"), (uord, iord, f"sorted by {k} Graclus clusters"))):
        im = a.imshow(density(uo, io), aspect="auto", cmap="magma", norm=matplotlib.colors.PowerNorm(0.4))
        a.set(xlabel="items", ylabel="users", title=f"interaction density, {t}"); fig.colorbar(im, ax=a, label="P(edge) in bin")
    fig.tight_layout(); fig.savefig(out / "3_interaction_matrix.png", dpi=140); plt.close(fig)

    # ---- fig 4: co-occurrence matrices, cluster-sorted (mean overlap per pair, binned) ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, (Rm, o, nm) in zip(ax, ((R, uord, "user-user"), (R.T.tocsr(), iord, "item-item"))):
        B = binned(o, args.bins)
        X = B @ Rm
        sz = np.asarray(B.sum(1)).ravel()
        im = a.imshow(np.log1p((X @ X.T).toarray() / np.outer(sz, sz)), aspect="auto", cmap="viridis")
        a.set(title=f"{nm}: mean co-occurrence per pair (log1p), cluster-sorted"); fig.colorbar(im, ax=a, label="log(1 + mean overlap)")
    fig.tight_layout(); fig.savefig(out / "4_cooccurrence_matrices.png", dpi=140); plt.close(fig)

    # ---- fig 5: sampled bipartite subgraph from the 3 biggest clusters, coloured by Graclus label ----
    rng = np.random.default_rng(args.seed)
    big = np.argsort(-np.bincount(labels, minlength=k))[:3]
    users, items = [], []
    for c in big:
        cu, ci = np.where(ucl == c)[0], np.where(icl == c)[0]
        users += list(rng.choice(cu, min(40, len(cu)), replace=False)); items += list(ci[np.argsort(-ideg[ci])][:25])
    sub = R[users][:, items].tocoo()
    G = nx.Graph(); G.add_nodes_from([("u", u) for u in users]); G.add_nodes_from([("i", i) for i in items])
    G.add_edges_from((("u", users[r]), ("i", items[c])) for r, c in zip(sub.row, sub.col))
    pos = nx.spring_layout(G, seed=args.seed, k=0.35, iterations=200)
    fig, a = plt.subplots(figsize=(8.5, 8))
    nx.draw_networkx_edges(G, pos, ax=a, alpha=0.15, width=0.5)
    for kind, shape, cl in (("u", "o", ucl), ("i", "s", icl)):
        nodes = [x for x in G if x[0] == kind]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=a, node_shape=shape, node_size=26 if kind == "u" else 40,
                               node_color=[plt.cm.tab10(int(cl[x[1]]) % 10) for x in nodes], linewidths=0.4, edgecolors="k")
    a.set_title("sampled subgraph, 3 largest Graclus clusters (colour = cluster, circle = user, square = item)"); a.axis("off")
    fig.tight_layout(); fig.savefig(out / "5_subgraph.png", dpi=140); plt.close(fig)
    print(f"\nwrote figures + stats.json to {out}")


if __name__ == "__main__":
    main()

"""The three graphs of MAGCL (Eqs 2-8, 9-12, 15).

All three are built ONCE, before training, from the training interactions and the
initial embedding table, and never change afterwards. The paper describes them as
a "Graph Construction and Denoising Module" that precedes the learning module
(Fig. 1), and gives no recomputation schedule.

Every graph lives on the same node set: user `u` is node `u`, item `i` is node
`m + i`, where `m` is the number of users. So all three adjacency matrices are
(m+n) x (m+n) and the convolutions are plain sparse matmuls against one shared
embedding table E0.

  G     user-item interactions, symmetrically normalised        -> Eq (9)
  G_hi  high-order user-user / item-item Jaccard graph          -> Eqs (2),(3),(10)
  G_dn  trust-score denoised user-item graph                    -> Eqs (4)-(8),(11),(12)
"""

import numpy as np
import scipy.sparse as sp
import torch


# pytorch does not understand scipy shit so we use to torch here
def to_torch_sparse(A: sp.spmatrix) -> torch.Tensor:
    """scipy sparse -> coalesced torch sparse float32 (what torch.sparse.mm wants)."""
    
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


def _row_norm(A: sp.spmatrix) -> sp.csr_matrix:
    """Divide every row by its sum. Empty rows stay empty (no division by zero)."""
    s = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / np.maximum(s, 1e-12)) @ A).tocsr()


def interaction_adj(R: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Eq (9): the bipartite graph G with symmetric normalisation D^-1/2 A D^-1/2.

    A = [[0, R], [R.T, 0]] over the (m+n) nodes, so one sparse matmul against the
    embedding table performs Eq (9) for users and items at the same time. This is
    exactly LightGCN propagation, and it is also the single hop used by Eqs (4)-(5)
    to compute trust scores.

    Returns the normalised adjacency and the raw node degrees |N_G(x)|.
    """
    A = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1.0)))
    return (dinv @ A @ dinv).tocsr(), deg


def activity(deg: np.ndarray) -> np.ndarray:
    """Eq (15): d_x = log|N_G(x)| / mean_{v in V} log|N_G(v)|.

    Normalised, log-dampened node activity used by the fusion weight (Eq 14).
    V is "the set of all nodes", i.e. users and items pooled. The mean is taken
    over nodes that actually have an edge: RecBole reserves id 0 on each side as
    a padding node with degree 0, and those rows would otherwise drag it down.
    """
    log_deg = np.log(np.maximum(deg, 1.0))
    return (log_deg / log_deg[deg > 0].mean()).astype(np.float32)


# this step is just a filtering step in the end the weights are 0 or 1 like a normal user item graph, but now we add connections to user user and item item interactions
def _jaccard_side(R: sp.csr_matrix, beta: float, top_s: int) -> sp.csr_matrix:
    """Eqs (2)-(3) for one side. Rows of `R` are the nodes being related.

    Eq (2)  JSC_{x,y} = |N(x) & N(y)| / |N(x) | N(y)|
    Eq (3)  keep y if JSC_{x,y} >= beta,  OR  y is among x's top-S by JSC.

    Self-pairs and pairs with no shared neighbour are never kept: they carry no
    collaborative signal, and including them would let top-S pad rows with
    arbitrary zero-similarity nodes.

    The returned matrix is BINARY. Eq (3) stores the JSC value, but Eq (10)
    aggregates neighbours with a flat 1/|N(x)| and never multiplies by that
    value -- unlike Eq (11), which puts its weight in the numerator explicitly.
    So here JSC selects edges; it is not a convolution weight.

    Stays sparse throughout -- an earlier version computed `R @ R.T` in dense
    chunks (`.toarray()`), which is O(k^2) in the number of nodes on this side
    no matter how it's chunked (chunking bounds memory per chunk, not total
    work). Fine at ML-1M's scale (k~6k) but ~2500x more work at Alibaba-
    iFashion's scale (k up to 300k users), making it impractically slow there.
    `R @ R.T` kept sparse only has a nonzero at (x,y) where x and y already
    share a neighbour -- exactly the `inter > 0` validity condition the old
    code applied right after densifying -- so restricting to that sparsity
    structure changes nothing about which pairs are considered; it only skips
    ever materialising the zeros. Verified bit-for-bit identical to the old
    dense version on the real ML-1M interaction matrix (both P and Q sides).
    """
    R = R.tocsr()
    k = R.shape[0]
    top_s = min(top_s, k - 1)
    deg = np.asarray(R.sum(axis=1)).ravel().astype(np.float64)

    inter = (R @ R.T).tocsr()                   # nonzero only where x,y share a neighbour
    inter = inter - sp.diags(inter.diagonal())  # drop x == y (setdiag(0) would warn on CSR)
    inter.eliminate_zeros()

    rows, cols = [], []
    indptr, indices, data = inter.indptr, inter.indices, inter.data
    for x in range(k):
        start, end = indptr[x], indptr[x + 1]
        if start == end:
            continue
        cols_x = indices[start:end]
        inter_x = data[start:end]
        jsc_x = inter_x / np.maximum(deg[x] + deg[cols_x] - inter_x, 1e-12)

        if len(jsc_x) <= top_s:
            keep_x = np.ones(len(jsc_x), dtype=bool)  # fewer neighbours than S: keep all
        else:
            kth = np.partition(jsc_x, -top_s)[-top_s]
            keep_x = (jsc_x >= beta) | (jsc_x >= kth)

        rows.append(np.full(keep_x.sum(), x))
        cols.append(cols_x[keep_x])

    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(k, k))


def high_order_adj(R: sp.csr_matrix, beta: float, top_s: int) -> sp.csr_matrix:
    """Eqs (2),(3),(10): the high-order collaborative graph, row-normalised.

    P relates users to users (shared items), Q relates items to items (shared
    users). The graph is block-diagonal [[P, 0], [0, Q]] -- it never links a user
    to an item, which is what makes it a complementary view of G.

    Row normalisation 1/|N_hi(x)| is Eq (10).
    """
    P = _row_norm(_jaccard_side(R, beta, top_s))
    Q = _row_norm(_jaccard_side(R.T.tocsr(), beta, top_s))
    return sp.bmat([[P, None], [None, Q]], format="csr")


def denoised_adj(
    R: sp.csr_matrix, A_ui: sp.csr_matrix, e0: torch.Tensor, theta: float
) -> tuple[sp.csr_matrix, float]:
    """Eqs (4)-(8),(11),(12): the trust-score denoised graph.

    Eqs (4)-(5)  one symmetrically normalised hop over G from the INITIAL
                 embeddings -- literally `A_ui @ E0`, the same operator as Eq (9).
                 Neighbourhood-averaged embeddings are used instead of raw ones
                 because a single noisy node embedding is unreliable (§4.1.2).
    Eq (6)       cosine similarity between the hopped user and item vectors.
    Eq (7)       T_{u,i} = (cos + 1) / 2, mapped into [0, 1].
    Eq (8)       hard denoising: drop the edge if T <= theta;
                 soft denoising: otherwise keep T itself as the edge weight.
    Eqs (11),(12) user rows are normalised by their own row sums of R_hat,
                 item rows by the column sums -- note these differ, so the
                 denoised graph is NOT symmetric.

    Worth knowing: cosines of freshly initialised embeddings sit near 0, so
    T is near 0.5 for almost every edge, while the paper tunes theta in
    [0.02, 0.1]. Hard denoising therefore drops close to nothing and this view
    is essentially a softly reweighted copy of G. That is what the paper
    specifies; the returned keep-ratio lets us log and confirm it.
    """
    with torch.no_grad():
        z1 = torch.sparse.mm(to_torch_sparse(A_ui).to(e0.device), e0)  # Eqs (4)-(5)
        u, i = R.nonzero()
        # Pulls out the hopped embedding for each interacting user and item. The + R.shape[0] (i.e., +m) is the same node-numbering trick from the very start of the file — items live at indices m..m+n-1 inside z1, so item i is actually row m+i.
        zu, zi = z1[u], z1[R.shape[0] + i]
        cos = (zu * zi).sum(1) / (zu.norm(dim=1) * zi.norm(dim=1) + 1e-12)  # Eq (6)
        trust = ((cos + 1.0) / 2.0).cpu().numpy()                           # Eq (7)

    keep = trust > theta                                                    # Eq (8)
    R_hat = sp.csr_matrix((trust[keep], (u[keep], i[keep])), shape=R.shape)

    user_block = _row_norm(R_hat)                                           # Eq (11)
    col_sum = np.maximum(np.asarray(R_hat.sum(axis=0)).ravel(), 1e-12)
    item_block = (R_hat @ sp.diags(1.0 / col_sum)).T.tocsr()                # Eq (12)
    adj = sp.bmat([[None, user_block], [item_block, None]], format="csr")
    return adj, float(keep.mean())

"""First-version JEPA recommender ("Predicting Hidden Neighborhoods in Latent Space", draft v0.3, §3-4).

LightGCN + BPR, plus one extra loss. Every epoch each user (>= 2 train edges) hides ceil(rho*deg)
of their items; the hidden items are grouped by item cluster into blocks. A 2-layer MLP maps the
user's embedding on the context graph (train graph minus hidden edges) plus the block's cluster
embedding z_k to the block's answer key -- the mean of the block's own layer-0 item rows, detached
(same tables as the encoder, no momentum). Loss = squared distance of the L2-normalised vectors (Eq 7).

z is the only conditioning signal (config `jepa_z`):
  mean       sg(mean of all layer-0 rows of the cluster's items)   [default, Eq 5 bottom]
  learnable  a free vector per cluster                              [Eq 5 top]
  none       no hint (the z-free baseline of Proposition 1)

Item clusters (§3): multi-level Graclus (torch_cluster) on the item-item graph C_bar = R^T R
normalised by sqrt(C_ii C_jj), train split only. torch_cluster's graclus is a single-level matching
(clusters of <= 2 items), so we contract and repeat until <= `n_clusters` clusters remain; the actual
count therefore lands in (n_clusters/2, n_clusters].

The JEPA part lives in JEPAMixin so it can sit on another backbone: JEPA below = LightGCN + BPR + JEPA,
src/jepa_cf/magcl_jepa.py = MAGCL + JEPA.

Everything in the paper's roadmap (EMA target, other answer keys, uniformity, cross-layer, denoised
view, ...) and the leak tests are deliberately NOT here.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from recbole.model.general_recommender import LightGCN

from jepa_cf.cluster_position import PositionHint, apply_control, compute_cluster_positions
from magcl import compat  # noqa: F401  (scipy patch; must load before recbole touches scipy)

POSITION_CACHE = Path(__file__).resolve().parents[2] / "outputs" / "cluster_position"


def graclus_item_clusters(R, p: int) -> torch.Tensor:
    """Item -> cluster id (0..n-1, n <= p) by repeated Graclus matching on C_bar (Eq 1).
    # ponytail: torch_cluster weighs edges by raw (summed) weight, not true normalised-cut Graclus;
    # fine for a first version, swap in the real graclus binary (scripts/prep_graclus.py) if clusters look bad."""
    from torch_cluster import graclus_cluster

    R = R.tocsr().astype(np.float32)
    R.data[:] = 1.0
    deg = np.asarray(R.sum(axis=0)).ravel()
    C = (R.T @ R).tocoo()
    off = C.row != C.col
    row, col = torch.from_numpy(C.row[off]).long(), torch.from_numpy(C.col[off]).long()
    w = torch.from_numpy((C.data[off] / np.sqrt(deg[C.row[off]] * deg[C.col[off]])).astype(np.float32))

    n = R.shape[1]
    assign = torch.arange(n)
    while n > p:
        cl = graclus_cluster(row, col, w, n)
        uniq, cl = torch.unique(cl, return_inverse=True)
        if len(uniq) == n:  # no edges left to match
            break
        assign, n = cl[assign], len(uniq)
        row, col = cl[row], cl[col]
        keep = row != col
        merged = torch.sparse_coo_tensor(torch.stack([row[keep], col[keep]]), w[keep], (n, n)).coalesce()
        (row, col), w = merged.indices(), merged.values()
    return assign


class JEPAMixin:
    """The JEPA loss, independent of the backbone. The host model (a RecBole GeneralRecommender) must provide:
    n_users, n_items, n_layers, latent_dim, device, logger, interaction_matrix (TRAIN split, users x items),
    get_ego_embeddings() -> the (n_users + n_items, d) layer-0 table, users first, and item_embedding.weight (its item rows).
    Call _init_jepa(config) at the end of the host's __init__."""

    _ctx = None  # per-epoch masks / context graph, built lazily in jepa_loss()

    def _init_jepa(self, config):
        self.rho = config["mask_ratio"]
        self.lam = config["jepa_weight"]
        # Centered keys (off by default): subtract the popularity-weighted mean item row from every block key. Early in
        # training all rows share one dominant direction, so an uncentered key is predictable by a constant (loss ~0.08,
        # not a real match); centering removes that shared direction and the loss starts near its honest value.
        self.center_keys = config["jepa_center_keys"] if "jepa_center_keys" in config else False
        # Where the answer key comes from: the layer-0 item rows (default, the draft's Eq 4) or the hidden items' readout
        # on the masked context graph (draft Q3 "context graph": no hidden edge of the user is in it, still stop-gradient).
        self.key_source = config["jepa_key_source"] if "jepa_key_source" in config else "layer0"
        if self.key_source not in ("layer0", "context"):
            raise ValueError(f"jepa_key_source must be layer0|context, got {self.key_source!r}")
        self.z_mode = config["jepa_z"]
        if self.z_mode not in ("mean", "learnable", "none", "position"):
            raise ValueError(f"jepa_z must be mean|learnable|none|position, got {self.z_mode!r}")

        R = self.interaction_matrix.tocsr()
        R.sum_duplicates()
        R.sort_indices()
        dev = self.device
        deg = torch.from_numpy(np.diff(R.indptr)).long().to(dev)
        self.edge_u = torch.repeat_interleave(torch.arange(self.n_users, device=dev), deg)  # CSR order: grouped by user
        self.edge_i = torch.from_numpy(R.indices).long().to(dev)
        self.indptr = torch.from_numpy(R.indptr).long().to(dev)
        # at least one item stays visible; users with < 2 edges hide nothing (no JEPA signal for them)
        n_hide = torch.ceil(self.rho * deg.float()).long().clamp(max=(deg - 1).clamp(min=0))
        self.n_hide = torch.where(deg >= 2, n_hide, torch.zeros_like(n_hide))

        item_deg = torch.bincount(self.edge_i, minlength=self.n_items).float()
        self.item_w = (item_deg / item_deg.sum()).unsqueeze(0)  # (1, n_items): hidden items are samples of train edges

        self.item_cluster = graclus_item_clusters(R, config["n_clusters"]).to(dev)
        self.p = int(self.item_cluster.max()) + 1
        self.cluster_size = torch.bincount(self.item_cluster, minlength=self.p).clamp(min=1).unsqueeze(1).float()
        self.logger.info(f"JEPA: {self.p} Graclus item clusters (asked <= {config['n_clusters']}), z={self.z_mode}, "
                         f"rho={self.rho}, lambda={self.lam}, center_keys={self.center_keys}")

        d = self.latent_dim
        if self.z_mode == "learnable":
            self.cluster_emb = torch.nn.Embedding(self.p, d)
        if self.z_mode in ("mean", "learnable"):
            self.W_z = torch.nn.Linear(d, d, bias=False)
        if self.z_mode == "position":  # has its own learned m -> d layer, so no W_z on top
            users, items = self.edge_u.cpu().numpy(), self.edge_i.cpu().numpy()
            # these are the model's own TRAIN edges (the matrix BPR trains on); data_preparation hands the model only the train split
            assert len(users) == R.nnz == len(items)
            pos, diag = compute_cluster_positions(
                users, items, self.item_cluster.cpu().numpy(), self.p, m=config["pos_dim_m"], topk=config["pos_topk"],
                eps=config["pos_eps"], source=config["pos_source"], cache_dir=POSITION_CACHE)
            self.logger.info(f"JEPA cluster positions (m={config['pos_dim_m']}): {json.dumps(diag)}")
            self.position_hint = PositionHint(apply_control(pos, config["pos_control"], config["seed"]), d)
        hid = config["jepa_pred_hidden"] if "jepa_pred_hidden" in config else d  # predictor width (default d)
        self.predictor = torch.nn.Sequential(torch.nn.Linear(d, hid), torch.nn.ReLU(), torch.nn.Linear(hid, d))

    def train(self, mode=True):
        """RecBole's Trainer calls model.train() at the start of every epoch -> masks resampled per epoch (Alg. 1, line 1)."""
        if mode:
            self._ctx = None
        return super().train(mode)

    def _norm_adj(self, users, items):
        n = self.n_users + self.n_items
        row, col = torch.cat([users, items + self.n_users]), torch.cat([items + self.n_users, users])
        deg = torch.bincount(row, minlength=n).clamp(min=1).float()
        w = deg[row].rsqrt() * deg[col].rsqrt()
        return torch.sparse_coo_tensor(torch.stack([row, col]), w, (n, n)).coalesce()

    @torch.no_grad()
    def _resample(self):
        """Hidden sets H_u, context graph G_ctx, blocks B_u^(k) (Eq 2) -- all vectorised on the device."""
        E = self.edge_u.numel()
        # u + U[0,1) sorts edges by user, randomly within each user
        order = torch.argsort(self.edge_u.double() + torch.rand(E, device=self.device, dtype=torch.float64))
        eu, ei = self.edge_u[order], self.edge_i[order]
        hidden = torch.arange(E, device=self.device) - self.indptr[eu] < self.n_hide[eu]
        hu, hi = eu[hidden], ei[hidden]
        blocks, edge_block = torch.unique(hu * self.p + self.item_cluster[hi], return_inverse=True)
        block_user = blocks // self.p
        self._ctx = {
            "adj": self._norm_adj(eu[~hidden], ei[~hidden]),
            "hu": hu, "hi": hi, "edge_block": edge_block,
            "block_user": block_user, "block_cluster": blocks % self.p,
            "blocks_per_user": torch.bincount(block_user, minlength=self.n_users),  # |K_u|
        }

    def _context_embeddings(self, adj):
        """Same LightGCN readout as forward(), but on G_ctx."""
        x = self.get_ego_embeddings()
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        return torch.stack(layers).mean(dim=0)

    def _z(self, clusters):
        if self.z_mode == "learnable":
            return self.cluster_emb(clusters)
        rows = self.item_embedding.weight.detach()
        sums = torch.zeros(self.p, rows.shape[1], device=rows.device).index_add_(0, self.item_cluster, rows)
        return (sums / self.cluster_size)[clusters]

    def _key_center(self, table=None):
        """Detached popularity-weighted mean of the item rows (the distribution the hidden items are drawn from)."""
        table = self.item_embedding.weight if table is None else table
        return (self.item_w @ table.detach()).squeeze(0)

    def jepa_loss(self, users):
        if self._ctx is None:
            self._resample()
        c = self._ctx
        in_batch = torch.zeros(self.n_users, dtype=torch.bool, device=self.device)
        in_batch[users] = True
        sel = in_batch[c["hu"]]  # hidden edges of this mini-batch's users
        if not sel.any():
            return torch.zeros((), device=self.device)

        blocks, inv = torch.unique(c["edge_block"][sel], return_inverse=True)
        E = self._context_embeddings(c["adj"])                   # readout on G_ctx; s_x^u = E[u]
        table = E[self.n_users:] if self.key_source == "context" else self.item_embedding.weight
        rows = table[c["hi"][sel]].detach()                      # sg(e_i): layer-0 rows (Eq 4) or their G_ctx readout
        key = torch.zeros(len(blocks), rows.shape[1], device=self.device).index_add_(0, inv, rows)
        key = key / torch.bincount(inv).unsqueeze(1)             # Eq 4: block mean
        if self.center_keys:
            key = key - self._key_center(table)                  # mean of (e_i - mu) over the block = block mean - mu

        bu = c["block_user"][blocks]
        h = E[bu]
        if self.z_mode == "position":
            h = h + self.position_hint(c["block_cluster"][blocks])
        elif self.z_mode != "none":
            h = h + self.W_z(self._z(c["block_cluster"][blocks]))
        pred = self.predictor(h)
        dist = (F.normalize(pred, dim=1) - F.normalize(key, dim=1)).pow(2).sum(dim=1)  # D, Eq 7
        # mean over a user's blocks, then over the batch's users
        return (dist / c["blocks_per_user"][bu]).sum() / torch.unique(bu).numel()


class JEPA(JEPAMixin, LightGCN):
    """Needs one sampled negative per positive (RecBole train_neg_sample_args), like LightGCN."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_jepa(config)
        self._parts = torch.zeros(3, device=self.device)  # running sums of (bpr, reg, jepa) for pop_loss_means()
        self._steps = 0

    def calculate_loss(self, interaction):
        """(BPR, reg, lambda * JEPA) -- RecBole sums the tuple (Eq 8). LightGCN's own version returns BPR + reg
        as one tensor, so the two are recomputed here to keep them apart."""
        self.restore_user_e = self.restore_item_e = None  # clear the eval cache, as LightGCN does
        user = interaction[self.USER_ID]
        pos_item, neg_item = interaction[self.ITEM_ID], interaction[self.NEG_ITEM_ID]

        user_all, item_all = self.forward()
        u_e = user_all[user]
        bpr = self.mf_loss((u_e * item_all[pos_item]).sum(dim=1), (u_e * item_all[neg_item]).sum(dim=1))
        reg = self.reg_weight * self.reg_loss(
            self.user_embedding(user), self.item_embedding(pos_item), self.item_embedding(neg_item),
            require_pow=self.require_pow).squeeze()  # EmbLoss returns shape [1]
        jepa = self.jepa_loss(user)  # raw D, before lambda

        self._parts += torch.stack([bpr, reg, jepa]).detach()
        self._steps += 1
        return bpr, reg, self.lam * jepa

    def pop_loss_means(self):
        """Mean per-step value of each loss part since the last call (one epoch), then reset.
        'jepa' is the raw distance D (comparable across lambda), 'jepa_weighted' is what enters the total."""
        bpr, reg, jepa = (self._parts / max(self._steps, 1)).tolist()
        self._parts.zero_()
        self._steps = 0
        return {"bpr": bpr, "reg": reg, "jepa": jepa, "jepa_weighted": self.lam * jepa}

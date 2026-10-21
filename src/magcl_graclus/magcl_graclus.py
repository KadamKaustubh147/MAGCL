"""MAGCL (Eqs 9-30) as a RecBole GeneralRecommender.

The three graphs (Eqs 2-8) are built once in __init__ by magcl.graphs and never
change afterwards. This file only implements what actually trains:

  forward()          fusion chain (Eqs 9,13,14) + two auxiliary views (Eqs 20-22)
  calculate_loss()   BPR + multi-view CL + cross-layer CL + uniformity + reg (Eq 30)

One deliberate deviation from the literal paper text, controlled by config
`normalize_cl` (default True): Eqs 23/24/26 print exp(e_v^T e_x / tau) with
UNNORMALIZED inner products. At tau=0.1 those logits reach the thousands and
overflow; every real GCL implementation (SGL, SimGCL, RecBole's own) L2-
normalizes both sides first. `normalize_cl: false` reproduces the literal
formula so the overflow can be demonstrated rather than assumed.

Second deviation (this file only), controlled by config `graclus_mode`
(default "off" = baseline MAGCL): the high-order graph A_hi (Eqs 2-3, 10) is
built with Graclus cluster labels computed offline on the TRAIN split
(scripts/prep_graclus.py -> graclus1.2 -> a .part.K file):

  off      untouched Jaccard graph
  replace  neighbours must share the node's Graclus cluster; beta / top-S (S_gr)
           are applied among cluster-mates only
  union    Jaccard edges OR cluster-restricted edges

Only A_hi changes; forward() and every loss are the baseline's.
"""

import numpy as np
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

from magcl_graclus import compat  # noqa: F401  (scipy patch; must load before recbole touches scipy)
from magcl_graclus.graphs import (
    activity, denoised_adj, edge_fingerprint, fraction_inside, high_order_adj, high_order_adj_graclus,
    interaction_adj, to_torch_sparse,
)

# Eq (14)'s denominator c*d_x + l can cross zero (c in [-1,1], d_x can exceed 1
# for high-activity nodes, l can be as small as 1) -- a real crossing, not just
# a "should never happen" edge case, so the guard has to keep gradients bounded,
# not merely avoid literal division by zero. d(gamma/x)/dx = -gamma/x^2, so this
# value also upper-bounds how large |eta|'s gradient can get.
_FUSION_EPS = 0.1


class MAGCL(GeneralRecommender):
    """Needs one sampled negative per positive (RecBole train_neg_sample_args)."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config["embedding_size"]  # GeneralRecommender doesn't set this itself

        # ---- hyperparameters (§5.2), one field per paper symbol ----
        self.L = config["L"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.theta = config["theta"]
        self.beta = config["beta"]
        self.top_s = config["S"]
        self.lam1 = config["lam1"]
        self.lam2 = config["lam2"]
        self.lam3 = config["lam3"]
        self.lam4 = config["lam4"]
        self.normalize_cl = config["normalize_cl"] if "normalize_cl" in config else True

        # ---- Graclus switches ----
        self.graclus_mode = config["graclus_mode"] if "graclus_mode" in config else "off"   # off | replace | union
        self.top_s_gr = config["S_gr"] if "S_gr" in config else self.top_s                  # top-S among cluster-mates
        if self.graclus_mode not in ("off", "replace", "union"):
            raise ValueError(f"graclus_mode must be off|replace|union, got {self.graclus_mode!r}")

        m, n = self.n_users, self.n_items
        R = dataset.inter_matrix(form="csr")[:m, :n]  # TRAIN split only

        # Theta = single embedding table E0 (m+n) x d, Xavier init (paper's only
        # trainable parameters -- every graph below is parameter-free).
        self.E0 = torch.nn.Parameter(torch.empty(m + n, self.embedding_size))
        torch.nn.init.xavier_uniform_(self.E0)

        A_ui, deg = interaction_adj(R)                                   # Eq (9)
        self.register_buffer("A_ui", to_torch_sparse(A_ui))
        self.register_buffer("d_act", torch.from_numpy(activity(deg)))   # Eq (15)

        if self.graclus_mode == "off":
            A_hi = high_order_adj(R, self.beta, self.top_s)              # Eqs (2),(3),(10)
        else:
            user_labels, item_labels = self._load_graclus_labels(config, R)
            A_hi, jaccard = high_order_adj_graclus(
                R, self.beta, self.top_s, self.graclus_mode, user_labels, item_labels, self.top_s_gr)
            inside = fraction_inside(jaccard, np.concatenate([user_labels, item_labels]))
            self.logger.info(
                f"MAGCL-Graclus [{self.graclus_mode}]: {inside:.1%} of the plain Jaccard top-S edges already stay "
                f"inside one Graclus cluster; A_hi has {A_hi.nnz} edges vs {jaccard.nnz} for plain Jaccard")
        self.register_buffer("A_hi", to_torch_sparse(A_hi))

        # Trust scores need E0 at its Xavier-initialised values -- built now,
        # before any training step touches E0.
        A_dn, keep_ratio = denoised_adj(R, A_ui, self.E0.detach(), self.theta)  # Eqs (4)-(8),(11),(12)
        self.register_buffer("A_dn", to_torch_sparse(A_dn))
        self.logger.info(f"MAGCL: denoised graph kept {keep_ratio:.1%} of edges (theta={self.theta})")

        # Full-sort eval calls full_sort_predict() once PER BATCH (RecBole's own
        # dataloader chunking), but forward() -- the whole L-layer graph
        # propagation -- doesn't depend on which users are in that batch, so
        # recomputing it every batch is pure waste. Cached here and reused for
        # the rest of one eval pass; invalidated in calculate_loss() since
        # training changes E0. Registered via other_parameter_name (RecBole's
        # own mechanism, same as LightGCN's restore_user_e/restore_item_e) so a
        # checkpoint reload for the final test evaluation restores the cache
        # consistent with the reloaded E0 -- without this, a naive plain-
        # attribute cache would silently keep stale embeddings from whichever
        # epoch trained last, not the best epoch the checkpoint reloads to.
        self.restore_e = None
        self.other_parameter_name = ["restore_e"]

    @staticmethod
    def _load_graclus_labels(config, R):
        """Per-RecBole-id cluster labels (users, items) from Graclus's .part file.

        prep_graclus.py wrote the graph nodes in the order [users with edges, items with edges]
        and saved the RecBole ids of those nodes in the .map.npz; the .part file has one label
        per node in that same order. Nodes with no edges (RecBole's id-0 padding, cold nodes) get
        UNIQUE negative labels, so they have no cluster-mates and therefore no restricted
        neighbours. The stored edge fingerprint guards against a different split/seed."""
        mapping = np.load(config["graclus_map"])
        if int(mapping["fingerprint"]) != edge_fingerprint(R):
            raise ValueError(
                f"{config['graclus_map']} was built from a different train split than this run's "
                "(edge fingerprint mismatch) -- rerun scripts/prep_graclus.py with the same config/seed")
        users, items = mapping["user_ids"], mapping["item_ids"]
        part = np.loadtxt(config["graclus_part"], dtype=np.int64)
        if len(part) != len(users) + len(items):
            raise ValueError(f"{config['graclus_part']} has {len(part)} labels, expected {len(users) + len(items)}")
        user_labels = -1 - np.arange(R.shape[0], dtype=np.int64)
        item_labels = -1 - np.arange(R.shape[1], dtype=np.int64)
        user_labels[users] = part[:len(users)]
        item_labels[items] = part[len(users):]
        return user_labels, item_labels

    def _fusion_weight(self, z_ui: torch.Tensor, z_hi: torch.Tensor, layer: int) -> torch.Tensor:
        """Eq (14): eta_x = gamma / (c(z_ui_x, z_hi_x) * d_x + l)."""
        cos = F.cosine_similarity(z_ui, z_hi, dim=1)
        denom = cos * self.d_act + layer
        sign = torch.where(denom >= 0, 1.0, -1.0)
        safe_denom = torch.where(denom.abs() < _FUSION_EPS, sign * _FUSION_EPS, denom)
        return self.gamma / safe_denom

    def forward(self):
        """Returns (e, e_hi, e_dn, fused_layers):

          e            fused view,        Eqs (16)-(17), pooled over l=0..L
          e_hi         high-order view,   Eq (20),        pooled over l=1..L only
          e_dn         denoised view,     Eqs (21)-(22),  pooled over l=1..L only
          fused_layers list [z^0, z^1, ..., z^L] -- z^{floor(L/2)} feeds Eq (26)

        The G and G_hi convolutions share one fused chain: z^l feeds BOTH next-
        layer convs (Eq 13's "serves as the input to the next graph convolution
        layer"). The G_dn chain is independent and starts from E0 (Eqs 11-12
        never mention z^l, only z^{Gdn,l-1}).
        """
        fused = self.E0
        fused_layers = [fused]
        hi_layers = []
        for l in range(1, self.L + 1):
            
            z_ui = torch.sparse.mm(self.A_ui, fused)         # Eq (9), input = fused z^{l-1}
            z_hi = torch.sparse.mm(self.A_hi, fused)         # Eq (10), input = fused z^{l-1}
            eta = self._fusion_weight(z_ui, z_hi, l)         # Eq (14)
            fused = z_ui + eta.unsqueeze(1) * z_hi           # Eq (13)
            fused_layers.append(fused)
            hi_layers.append(z_hi)

        z_dn = self.E0
        dn_layers = []
        for _ in range(self.L):
            z_dn = torch.sparse.mm(self.A_dn, z_dn)          # Eqs (11)-(12)
            dn_layers.append(z_dn)

        e = torch.stack(fused_layers, dim=0).mean(dim=0)     # Eqs (16)-(17): l = 0..L
        e_hi = torch.stack(hi_layers, dim=0).mean(dim=0)     # Eq (20): l = 1..L, no z^0
        e_dn = torch.stack(dn_layers, dim=0).mean(dim=0)     # Eqs (21)-(22): l = 1..L, no z^0
        return e, e_hi, e_dn, fused_layers

    def _info_nce(self, anchor_table: torch.Tensor, key_table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Shared core of Eqs (23),(24),(26): pooled in-batch InfoNCE.

        idx = the batch node set B (Eq 25's "users or positive items in the
        current mini-batch") -- NOT deduplicated: if a user appears twice in a
        batch it is two separate anchors, each with its own term in the sum,
        matching the paper's literal Sigma_{v in B}. (Contrast with uniformity
        below, whose Uniq(...) operator explicitly asks for deduplication.)

        Returns the SUM over B, as printed (not a mean) -- see the module-level
        deviation note for why this matters for lambda1/lambda2's scale.
        """
        a, k = anchor_table[idx], key_table[idx]
        if self.normalize_cl:
            a = F.normalize(a, dim=1)
            k = F.normalize(k, dim=1)
        pos = (a * k).sum(dim=1) / self.tau                  # e_v^T e_v^view / tau
        logits = (a @ k.T) / self.tau                        # e_v^T e_x^view / tau, all x in B
        return (torch.logsumexp(logits, dim=1) - pos).sum()  # Sigma_v -log(...)

    def _batch_nodes(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """B = users U positive items of the current mini-batch (item ids already m-offset)."""
        return torch.cat([u, i])

    def multiview_cl(self, e, e_hi, e_dn, batch_idx) -> torch.Tensor:
        """Eqs (23)-(25)."""
        l_dn = self._info_nce(e, e_dn, batch_idx)   # Eq (23)
        l_hi = self._info_nce(e, e_hi, batch_idx)   # Eq (24)
        return l_dn + self.alpha * l_hi              # Eq (25)

    def cross_layer_cl(self, e, fused_layers, batch_idx) -> torch.Tensor:
        """Eq (26): anchor = final fused e_v, key = fused mid-layer z^{floor(L/2)}."""
        z_mid = fused_layers[self.L // 2]
        return self._info_nce(e, z_mid, batch_idx)

    def _uniformity_side(self, e: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Eqs (27)/(28): log E[exp(-2||f(x)-f(x')||^2)] over deduplicated pairs.

        ||a-b||^2 is expanded as ||a||^2+||b||^2-2a.b (the gram form) instead of
        calling .norm() on a difference: that form's backward is undefined at
        distance 0, and this loss actively pulls points together, so distance-0
        pairs are a real training-time occurrence, not a corner case.
        """
        idx = torch.unique(idx)                      # Eqs 27/28's Uniq(...) operator
        f = F.normalize(e[idx], dim=1)                # f(.) = L2 normalisation
        if f.shape[0] < 2:
            return f.new_zeros(())
        sq_norm = (f * f).sum(dim=1)
        dist2 = (sq_norm.unsqueeze(1) + sq_norm.unsqueeze(0) - 2.0 * (f @ f.T)).clamp_min(0.0)
        potential = torch.exp(-2.0 * dist2)
        n = f.shape[0]
        mean_potential = (potential.sum() - potential.diagonal().sum()) / (n * (n - 1))
        return torch.log(mean_potential + 1e-12)

    def uniformity(self, e, u, i) -> torch.Tensor:
        """Eq (29), measured on the fused view e."""
        return 0.5 * (self._uniformity_side(e, u) + self._uniformity_side(e, i))

    def bpr_loss(self, e, u, i, j) -> torch.Tensor:
        """Eq (19). RecBole convention: mean over the batch, not the printed sum
        (this is the standard BPR reduction; only the CL terms' sum-vs-mean
        choice interacts with the paper's lambda tuning range)."""
        y_pos = (e[u] * e[i]).sum(dim=1)
        y_neg = (e[u] * e[j]).sum(dim=1)
        return -F.logsigmoid(y_pos - y_neg).mean()

    def calculate_loss(self, interaction) -> tuple:
        """Eq (30). Returns weighted parts, NOT a running total -- RecBole's
        trainer does `loss = sum(returned tuple)` for backward and logs every
        element separately (train_loss1, train_loss2, ... in the epoch line and
        TensorBoard/MLflow). Returning a total alongside the parts would double
        count everything once summed."""
        if self.restore_e is not None:  # training changes E0 -- invalidate the eval cache
            self.restore_e = None

        u = interaction[self.USER_ID]
        i = interaction[self.ITEM_ID] + self.n_users
        j = interaction[self.NEG_ITEM_ID] + self.n_users

        e, e_hi, e_dn, fused_layers = self.forward()
        batch_idx = self._batch_nodes(u, i)

        bpr = self.bpr_loss(e, u, i, j)
        cl = self.lam1 * self.multiview_cl(e, e_hi, e_dn, batch_idx)
        layer = self.lam2 * self.cross_layer_cl(e, fused_layers, batch_idx)
        unif = self.lam3 * self.uniformity(e, u, i)
        reg = self.lam4 * (self.E0 ** 2).sum()  # Theta = ALL trainable params (the full table)
        return bpr, cl, layer, unif, reg

    def predict(self, interaction) -> torch.Tensor:
        u = interaction[self.USER_ID]
        i = interaction[self.ITEM_ID] + self.n_users
        e, _, _, _ = self.forward()
        return (e[u] * e[i]).sum(dim=1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """Eq (18) scored against every item at once (RecBole excludes train-seen items itself).

        Cached across calls within one evaluation pass -- see self.restore_e's
        comment in __init__. Matches RecBole's own LightGCN convention exactly
        (predict() stays uncached there too; only full_sort_predict benefits,
        since that's the one called once per eval batch)."""
        if self.restore_e is None:
            self.restore_e, _, _, _ = self.forward()
        u = interaction[self.USER_ID]
        return self.restore_e[u] @ self.restore_e[self.n_users:].T

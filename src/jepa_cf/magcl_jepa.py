"""MAGCL + the JEPA auxiliary loss (draft T8, the "add" variant: MAGCL is untouched, one more loss term).

Built on src/magcl_graclus's MAGCL (graclus_mode "off" = plain MAGCL; "union"/"replace" also work). MAGCL keeps a single
layer-0 table E0 (users first, then items), which is exactly what JEPAMixin needs:
  * the JEPA context encoder is plain LightGCN propagation of E0 over the MASKED train graph (MAGCL's own A_hi / A_dn are
    built once from the full train graph, so using them would leak the hidden edges);
  * the answer key is the mean of the hidden items' E0 rows, detached;
  * the JEPA gradient reaches E0 through the context embedding, so it shapes the same table MAGCL trains.

calculate_loss returns MAGCL's five terms plus lambda * JEPA: (bpr, cl, layer, unif, reg, jepa).
"""

import types

import numpy as np
import torch

from jepa_cf.jepa import JEPAMixin
from magcl_graclus.magcl_graclus import MAGCL


class MAGCLJEPA(JEPAMixin, MAGCL):
    _PARTS = ("bpr", "cl", "layer", "unif", "reg", "jepa")

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)  # TRAIN split, as MAGCL's R
        self.latent_dim, self.n_layers = self.embedding_size, self.L
        self._init_jepa(config)
        self._parts = torch.zeros(len(self._PARTS), device=self.device)  # running sums for pop_loss_means()
        self._steps = 0

    @property
    def item_embedding(self):
        """The item rows of MAGCL's single table (a view: gradients still reach E0)."""
        return types.SimpleNamespace(weight=self.E0[self.n_users:])

    def get_ego_embeddings(self):
        return self.E0

    def calculate_loss(self, interaction):
        losses = super().calculate_loss(interaction)  # MAGCL's (bpr, cl, layer, unif, reg); also clears the eval cache
        jepa = self.jepa_loss(interaction[self.USER_ID])  # raw D, before lambda
        self._parts += torch.stack([*losses, jepa]).detach()
        self._steps += 1
        return (*losses, self.lam * jepa)

    def pop_loss_means(self):
        """Mean per-step value of each loss part since the last call (one epoch), then reset. 'jepa' is the raw
        distance D (comparable across lambda); the parts before it are already weighted as in MAGCL's Eq 30."""
        means = dict(zip(self._PARTS, (self._parts / max(self._steps, 1)).tolist()))
        self._parts.zero_()
        self._steps = 0
        means["jepa_weighted"] = self.lam * means["jepa"]
        return means

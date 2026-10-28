"""AdaMCL (adaptive multi-view fusion contrastive learning for graph CF) on RecBole 1.2.0.

Copied from the authors' code (C:\\dev_stuff\\AdaMCL\\AdaMCL\\AdaMCL.py, written for RecBole 1.0.0 / torch 1.7.1).
The model maths is untouched: forward(), the three InfoNCE terms and the loss are line-for-line the original.
What changed, and why:

1. Neighbour-graph cache is keyed by SEED. The original cached the top-K user/item similarity matrix as
   s_Myadj_<TopK>_<u>_<i>.npz next to the dataset, ignoring the split. RecBole's random 8:1:1 split depends on
   the seed, and this matrix is built from the TRAIN interactions, so reusing a file from another seed would
   leak that seed's test items into training. Now: <adamcl_cache_dir>/<dataset>_seed<seed>_<TopK>_<u>_<i>.npz
   (set adamcl_cache_dir: null to disable caching).
2. GetAdj: the per-entry python loop (36M iterations on ML-1M) is vectorised and the heapq top-K replaced by a
   lexsort with the same tie-break (largest value, then largest index), so the selected neighbours are identical.
   Lists the original built but never used (neighbour values/degrees/max_len) are dropped.
3. get_norm_adj_mat builds the adjacency with coo_matrix instead of dok_matrix._update (gone in modern SciPy)
   and no longer writes deg.txt as a side effect.
4. The sparse_mm cuSPARSE workaround is gone (torch 1.7.1 on RTX 40xx only); torch.sparse.mm works on torch 2.6.
5. EmbLoss(require_pow=True) is in RecBole 1.2.0, so the hand-copied emb_pow_loss is no longer needed.
"""

import os
from time import time

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from scipy.sparse import coo_matrix


def to_torch_sparse(m) -> torch.Tensor:
    m = m.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data), m.shape).coalesce()


class AdaMCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(AdaMCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.TopK = config['TopK']
        self.cache_dir = config['adamcl_cache_dir']
        self.cache_tag = f"{config['dataset']}_seed{config['seed']}"
        self.user_threshold = config['user_threshold']
        self.item_threshold = config['item_threshold']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.info_temp = config['ssl_temp']
        self.info_reg = config['info_reg']
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.hyper_layer = config['hyper_layer']

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat, self.deg = self.get_norm_adj_mat()
        self.norm_adj_mat = self.norm_adj_mat.to(self.device)
        self.deg[self.deg == 0] = 1
        log_deg = np.log(self.deg)
        mean_deg = np.mean(log_deg)
        log_deg = log_deg / mean_deg
        self.log_deg = torch.tensor(log_deg).reshape(-1, 1).to(self.device).to(torch.float32)

        self.norm_myadj_mat = self.get_mydata().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    @staticmethod
    def argmax_top_k(a, top_k=50):
        # == the original heapq.nlargest over (value, index) pairs: largest value first, ties -> larger index
        order = np.lexsort((np.arange(len(a)), a))[::-1][:top_k]
        return order.astype(np.intc)

    def GetAdj(self, R, topK, train_N, num_users, low_pass):
        """Jaccard similarity of rows of R, then per row: the top-K most similar OR everything above low_pass
        (whichever list is longer), row-normalised. Returns a [num_users, num_users] csr matrix."""
        union = R.dot(R.T).tocsr()

        value = union.data
        row_pointers = union.indptr
        column_index = union.indices

        train_N = np.array(train_N)
        row_index = np.repeat(np.arange(num_users), np.diff(row_pointers))
        value = value / (train_N[0][row_index] + train_N[0][column_index] - value)   # Jaccard
        value[column_index == row_index] = 0

        norm_adj = coo_matrix((value, (row_index, column_index)), shape=(num_users, num_users))
        norm_adj.data[np.isinf(norm_adj.data)] = 0.
        norm_adj.data = np.where(norm_adj.data > low_pass, norm_adj.data, 0.)
        norm_adj.eliminate_zeros()
        norm_adj = norm_adj.tocsr()
        norm_row_pointers = norm_adj.indptr
        norm_column_index = norm_adj.indices
        norm_data = norm_adj.data
        user_topk_columns = []
        user_topk_values = []
        user_topk_rows = []
        for i in range(num_users):
            start = row_pointers[i]
            end = row_pointers[i + 1]
            norm_start = norm_row_pointers[i]
            norm_end = norm_row_pointers[i + 1]

            value_index = self.argmax_top_k(value[start:end], topK) + start
            column1 = column_index[value_index]
            column2 = norm_column_index[norm_start:norm_end]

            fin_column = column1 if len(column1) > len(column2) else column2
            fin_value = value[value_index] if len(column1) > len(column2) else norm_data[norm_start:norm_end]

            f_sum = np.sum(fin_value)
            user_topk_columns.extend(fin_column)
            user_topk_values.extend(fin_value / f_sum)
            user_topk_rows.extend(np.array([i]).repeat(len(fin_column)))
        matrix = coo_matrix((user_topk_values, (user_topk_rows, user_topk_columns)), shape=(num_users, num_users))
        return matrix.tocsr()

    def get_mydata(self):
        cache = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache = os.path.join(self.cache_dir, '%s_%d_%f_%f.npz' % (
                self.cache_tag, self.TopK, self.user_threshold, self.item_threshold))
        if cache and os.path.isfile(cache):
            t3 = time()
            Myadj_mat = sp.load_npz(cache)
            print('already load topK matrix', Myadj_mat.shape, time() - t3)
        else:
            t4 = time()
            R = self.interaction_matrix.tolil()
            topK = self.TopK
            train_N = np.sum(R, dtype=np.float32, axis=1).reshape(-1, )
            train_M = np.sum(R.T, dtype=np.float32, axis=1).reshape(-1, )
            U_u = self.GetAdj(R, topK, train_N, self.n_users, self.user_threshold)
            I_i = self.GetAdj(R.T, topK, train_M, self.n_items, self.item_threshold)

            # users-with-users and items-with-items only; no user-item block (that is the LightGCN graph's job)
            Myadj_mat = sp.block_diag([U_u, I_i], format='lil', dtype=np.float32)
            Myadj_mat.setdiag(0)
            Myadj_mat = Myadj_mat.tocsr()

            print('already creat topK matrix', Myadj_mat.shape, time() - t4)
            if cache:
                sp.save_npz(cache, Myadj_mat)

        return to_torch_sparse(Myadj_mat)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix, and the node degrees.
        """
        inter_M = self.interaction_matrix
        n = self.n_users + self.n_items
        row = np.concatenate([inter_M.row, inter_M.col + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M.row])
        A = coo_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)).tocsr()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        deg = np.array(sumArr.flatten())[0]
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D

        return to_torch_sparse(L), deg

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        embeddings_y = []
        embeddings_z = []
        for layer_idx in range(1, self.n_layers + 1):
            all_embeddings1 = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            all_embeddings2 = torch.sparse.mm(self.norm_myadj_mat, all_embeddings)

            embeddings_y.append(all_embeddings1)
            embeddings_z.append(all_embeddings2)

            a_1 = torch.nn.functional.normalize(all_embeddings1, p=2, dim=1)
            b_1 = torch.nn.functional.normalize(all_embeddings1 + all_embeddings2, p=2, dim=1)
            sim_1 = torch.multiply(a_1, b_1).sum(dim=1).reshape(-1, 1)
            sim_1 = torch.clamp(sim_1, min=0.0)
            beta = self.gamma / (layer_idx + sim_1 * self.log_deg)
            all_embeddings2 = torch.multiply(all_embeddings2, beta)
            all_embeddings = all_embeddings1 + all_embeddings2
            embeddings_list.append(all_embeddings)

        hyper_layer_embedding = embeddings_list[self.hyper_layer]

        user_all_embeddings_x, item_all_embeddings_x = torch.split(hyper_layer_embedding, [self.n_users, self.n_items])

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        lightgcn_all_embeddings_y = torch.stack(embeddings_y, dim=1)
        lightgcn_all_embeddings_y = torch.mean(lightgcn_all_embeddings_y, dim=1)

        lightgcn_all_embeddings_z = torch.stack(embeddings_z, dim=1)
        lightgcn_all_embeddings_z = torch.mean(lightgcn_all_embeddings_z, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_y, item_all_embeddings_y = torch.split(lightgcn_all_embeddings_y, [self.n_users, self.n_items])
        user_all_embeddings_z, item_all_embeddings_z = torch.split(lightgcn_all_embeddings_z, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, user_all_embeddings_x, item_all_embeddings_x, \
               user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z, item_all_embeddings_z

    def creat_infoloss(self, users, a_embeddings, b_embeddings,
                       ssl_temp, ssl_reg):
        # argument settings
        batch_users = torch.unique(users)
        user_emb1 = torch.nn.functional.normalize(a_embeddings[batch_users], p=2, dim=1)
        user_emb2 = torch.nn.functional.normalize(b_embeddings[batch_users], p=2, dim=1)
        # user
        user_pos_score = torch.multiply(user_emb1, user_emb2).sum(dim=1)
        user_ttl_score = torch.matmul(user_emb1, user_emb2.t())
        user_pos_score = torch.exp(user_pos_score / ssl_temp)
        user_ttl_score = torch.exp(user_ttl_score / ssl_temp).sum(dim=1)
        user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()
        ssl_loss = ssl_reg * user_ssl_loss

        return ssl_loss

    def creat_sslloss(self, user, pos_item, user_all_embeddings, item_all_embeddings, user_all_embeddings_x,
                      item_all_embeddings_x, user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z,
                      item_all_embeddings_z):
        user_ssl = self.creat_infoloss(user, user_all_embeddings, user_all_embeddings_x, self.ssl_temp, self.ssl_reg)

        item_ssl = self.creat_infoloss(pos_item, item_all_embeddings, item_all_embeddings_x, self.ssl_temp,
                                       self.ssl_reg)

        user_infoy = self.creat_infoloss(user, user_all_embeddings_y, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoy = self.creat_infoloss(pos_item, item_all_embeddings_y, item_all_embeddings, self.info_temp,
                                         self.info_reg)

        user_infoz = self.creat_infoloss(user, user_all_embeddings_z, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoz = self.creat_infoloss(pos_item, item_all_embeddings_z, item_all_embeddings, self.info_temp,
                                         self.info_reg)

        return user_ssl + item_ssl, user_infoy + item_infoy + self.alpha * (user_infoz + item_infoz)

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, \
        user_all_embeddings_x, item_all_embeddings_x, \
        user_all_embeddings_y, item_all_embeddings_y, \
        user_all_embeddings_z, item_all_embeddings_z = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=True)

        ssl_loss, info_loss = self.creat_sslloss(user, pos_item, user_all_embeddings, item_all_embeddings,
                                                 user_all_embeddings_x, item_all_embeddings_x,
                                                 user_all_embeddings_y, item_all_embeddings_y,
                                                 user_all_embeddings_z, item_all_embeddings_z)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, info_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()[:2]

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()[:2]
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

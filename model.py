import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax
import torch
from torch import nn
import numpy as np
import sys
import time


def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params['depth'],
        num_classes=hyper_params['num_items']
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get='ntk')

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        K_train = kernel_fn(X_train, X_train) # user * user
        K_predict = kernel_fn(X_predict, X_train) # user * user
        K_reg = (K_train + jnp.abs(reg) * jnp.trace(K_train) * jnp.eye(K_train.shape[0]) / K_train.shape[0]) # user * user
        return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))
        # sp.linalg.solve(K_reg, X_train, sym_pos=True)) -> user * item

    return kernelized_rr_forward, kernel_fn

def FullyConnectedNetwork( 
    depth,
    W_std = 2 ** 0.5, 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk'
):
    activation_fn = stax.Relu()
    dense = functools.partial(stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth): layers += [dense(1024), activation_fn] 
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    return stax.serial(*layers)


class FUGCF(nn.Module):
    def __init__(self, adj_mat, norm_adj, user_sv, item_sv, adj_mat_pos, user_sv_f, item_sv_f, device='cuda:0'):
        super(FUGCF, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.norm_adj = norm_adj.to(device)
        self.user_sv = user_sv.to(device) # (K, M)
        self.item_sv = item_sv.to(device) # (K, N)
        self.adj_mat_pos = adj_mat_pos.to(device)
        self.user_sv_f = user_sv_f.to(device)
        self.item_sv_f = item_sv_f.to(device)


    def generate_T2(self, T1):
        indices = T1._indices()
        values = T1._values()
        max_col_index = T1.size(1)
        new_indices = torch.stack([
            indices[0],
            (indices[1] + 1) % max_col_index
        ])
        new_values = values.clone()
        T2 = torch.sparse_coo_tensor(new_indices, new_values, T1.size(), device=T1.device)
        return T2


    def preprocess_sparse_matrix(self, T):
        indices, values = T._indices(), T._values()
        row_indices = indices[0]
        unique_rows = torch.unique(row_indices)
        new_indices = []
        new_values = []

        for row in unique_rows:
            mask = (row_indices == row)
            row_indices_in_col = indices[:, mask]
            row_values = values[mask]
            new_indices.append(row_indices_in_col[:, 0])
            new_values.append(row_values[0])

        new_indices = torch.stack(new_indices, dim=1)
        new_values = torch.tensor(new_values, device=T.device)

        return torch.sparse_coo_tensor(new_indices, new_values, T.size(), device=T.device)

    def compute_P(self, T1, T2, mat_expo):
        indices1, values1 = T1._indices(), T1._values()
        indices2, values2 = T2._indices(), T2._values()

        def indices_to_set(indices):
            return set(tuple(idx.cpu().numpy()) for idx in indices.T)

        set1 = indices_to_set(indices1)
        set2 = indices_to_set(indices2)
        union_indices = torch.tensor(list(set1 | set2), dtype=torch.long, device=T1.device).T

        aligned_values1 = torch.zeros(union_indices.shape[1], device=T1.device)
        aligned_values2 = torch.zeros(union_indices.shape[1], device=T2.device)

        for idx, union_idx in enumerate(union_indices.T):
            pos1 = ((indices1 == union_idx.unsqueeze(1)).all(dim=0)).nonzero(as_tuple=True)
            if len(pos1[0]) > 0:
                aligned_values1[idx] = values1[pos1[0][0]]
            pos2 = ((indices2 == union_idx.unsqueeze(1)).all(dim=0)).nonzero(as_tuple=True)
            if len(pos2[0]) > 0:
                aligned_values2[idx] = values2[pos2[0][0]]

        T_diff_values = aligned_values1 - aligned_values2
        T_diff_values[T_diff_values == 0] = 1e-5

        mat_expo = mat_expo[union_indices[0], union_indices[1]]
        P_values = mat_expo / T_diff_values

        P = torch.sparse_coo_tensor(union_indices, P_values, T1.size(), device=T1.device)
        return P


    def forward(self, lambda_mat, lambda_mat_pos):
        ###### Exposure probability matrix ######
        A = self.item_sv_f @ (torch.diag(1/lambda_mat_pos)) @ self.user_sv_f.T
        mat_expo = torch.mm(self.adj_mat, A)

        # Normalization (col)
        epsilon = 1e-8
        col_min = torch.min(mat_expo, dim=0, keepdim=True).values
        mat_expo_shifted = mat_expo - col_min + epsilon
        col_sum = torch.sum(mat_expo_shifted, dim=0, keepdim=True)
        mat_expo = mat_expo_shifted / col_sum

        ###### GNN ######
        adj_mat_pos_transposed = torch.sparse_coo_tensor(
            self.adj_mat_pos._indices()[[1, 0]], 
            self.adj_mat_pos._values(),
            self.adj_mat_pos.size()[::-1],
            device=self.adj_mat_pos.device
        )
        coeff = (self.user_sv.shape[0] * self.item_sv.shape[0]) ** 0.5
        item_sv_gnn = adj_mat_pos_transposed @ self.user_sv / coeff
        user_sv_gnn = self.adj_mat_pos @ self.item_sv / coeff

        ###### Concat GNN+SVD ######

        # item_sv_gnn = torch.cat((item_sv_gnn, self.item_sv), dim=1) # LR-GCCF
        # user_sv_gnn = torch.cat((user_sv_gnn, self.user_sv), dim=1) # LR-GCCF
        # lambda_mat = torch.cat((lambda_mat, lambda_mat), dim=0)     # LR-GCCF
        item_sv_gnn = (item_sv_gnn + self.item_sv) / 2 # LightGCN
        user_sv_gnn = (user_sv_gnn + self.user_sv) / 2 # LightGCN

        ###### Get rating matrix using closed-form solution ######
        A = item_sv_gnn @ (torch.diag(1/lambda_mat)) @ user_sv_gnn.T
        B = A @ self.adj_mat_pos.to_dense()
        mat_rating = torch.mm(self.norm_adj, B)
        T1 = self.preprocess_sparse_matrix(self.adj_mat_pos)
        T2 = self.generate_T2(T1)

        time_1 = time.time()
        mat_adjust = self.compute_P(T1, T2, mat_expo)
        time_2 = time.time()
        adjustment_ratio = 0.2
        mat_rating += mat_adjust.to_dense() * adjustment_ratio

        return mat_rating
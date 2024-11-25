import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

import time
import copy
import random
import numpy as np
import torch
from jax import numpy as jnp
import sys
import model
from parse import parse_args
from utils import log_end_epoch, get_item_propensity, get_common_path, set_seed, preprocess_svd, preprocess_mat_pos, preprocess_ease, convert_sp_mat_to_sp_tensor



args = parse_args()

def train(hyper_params, data):
    from model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(hyper_params)
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    adj_mat = data.data['train_matrix'] + data.data['val_matrix']
    adj_mat_pos = data.data['train_matrix_pos'] + data.data['val_matrix']
    PATH = os.getcwd()
    adj_mat, norm_adj, ut, s, vt = preprocess_svd(hyper_params['load'], hyper_params['dataset'], adj_mat, hyper_params['k'], os.path.join(PATH, 'checkpoints'), device)
    adj_mat_pos, ut_f, s_pos, vt_f = preprocess_mat_pos(hyper_params['load'], hyper_params['dataset'], adj_mat_pos, hyper_params['k'], os.path.join(PATH, 'checkpoints'), device)
    print("adj_mat.shape: ", adj_mat.shape)

    train_model = model.FUGCF(adj_mat, norm_adj, ut, vt, adj_mat_pos, ut_f, vt_f, device)

    sampled_matrix = jnp.array(sampled_matrix.todense())

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)
    
    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric, best_lamda = None, None

    print("length of s is: ", len(s))
    s = s.to(device)
    print("length of s_pos is: ", len(s_pos))
    s_pos = s_pos.to(device)
    rating = train_model(s, s_pos)
    test_metrics, preds = evaluate(rating, hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
    
    # MSE
    adj_mat = data.data['train_matrix'] + data.data['val_matrix']
    adj_mat = jnp.array(convert_sp_mat_to_sp_tensor(adj_mat).to_dense())
    err = (preds - adj_mat) ** 2
    mse = sum(sum(err)) / (adj_mat.shape[0] * adj_mat.shape[1])
    print("\nMSE value: {}".format(mse))

    # Return metrics with the best lamda on the test-set
    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics

def main(hyper_params, gpu_id = None):
    if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax.config import config
    if 'float64' in hyper_params and hyper_params['float64'] == True: config.update('jax_enable_x64', True)

    from data import Dataset

    os.makedirs("./results/logs/", exist_ok=True)
    hyper_params['log_file'] = "./results/logs/" + get_common_path(hyper_params) + ".txt"
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params) # Updated w/ data-stats

    return train(hyper_params, data)

if __name__ == "__main__":
    from hyper_params import hyper_params
    set_seed(hyper_params['seed'])
    device = torch.device('cuda:4')
    main(hyper_params)

# Some of our codes are from https://github.com/tkipf/pygcn

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from pygcn.utils import load_data, accuracy
from pygcn.models import CrossDomainGCN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# kernel creation
def laplacian_based_kernel(k, p=2):
    I = torch.eye(k.shape[0])
    L = I - k
    Kl = (I - p/(p+1) * L)
    return Kl #torch.linalg.solve(Kl, I)


def feature_based_kernel(k, gamma=0.1, a3 = 1):
    I = torch.eye(k.shape[0]).cuda()
    K_ = torch.linalg.solve(a3 * I + k, I)
    K_ = (1 + gamma) * I - gamma * torch.matmul(k, K_)
    return torch.linalg.solve(K_, I)


def inverse_cosine(k):
    from scipy.fftpack import idct
    k = k * np.pi / 4.0
    return torch.FloatTensor(idct(k.numpy()))
    # return torch.cos(k * np.pi / 4.0)


def diffusion_process(k, sigma2, n=100):
    t = -sigma2 / 2.0 * k
    I = torch.eye(k.shape[0]).cuda()
    return torch.matrix_power(I + t/n, n)

setup_seed(10086)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.') # require parameter tuning
parser.add_argument('--multi_kernel_r', type=float, default=0.1, help='r for multiple kernel learning.') # require cross-validation
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_name = "wisconsin"
data_info = {"actor": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}
    , "cornell": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}
    , "texas": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}
    , "wisconsin": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}
    , "cora": {"label_num": 140, "valid_num": 500, "gamma": 10}
    , "pubmed": {"label_num": 60, "valid_num": 500, "gamma": 10}
    , "citeseer": {"label_num": 120, "valid_num": 500, "gamma": 10}
    , "chameleon": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}
    , "squirrel": {"label_num": 0.6, "valid_num": 0.2, "gamma": 0.01}}

# Load data
adj, features, labels, idx_train, idx_val, idx_test, K = load_data(kernel=True, label_num=data_info.get(data_name).get("label_num")
                                                                   , dataset=data_name, valid_num=data_info.get(data_name).get("valid_num")
                                                                   , sparse_k=False)
# obtain Z
print("done load, begin to train")
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    K = K.cuda()

###  The following code could be used to check the assortativity
# import networkx as nx
# graph_nx = nx.from_edgelist(adj.coalesce().indices().cpu().numpy().T)
# coef = nx.degree_assortativity_coefficient(graph_nx)

p = 2
gamma = data_info.get(data_name).get("gamma")   # 10 for the assortative graphs, or 0.01 for the disassortative graphs
a3 = 1       # 1 for last full experiment
Kl = adj.to_dense()
Kf2 = feature_based_kernel(K, gamma=gamma, a3=a3)

if args.cuda:
    kernel_list = [Kl.cuda(), Kf2.cuda()]


def train(epoch, optimizer, r, model):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, kernel_list, r)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, kernel_list, r)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    model.eval()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'weight: {:.4f}'.format(model.gc1.weight.sum()))
    return acc_val


def test(model, r):
    model.eval()
    output = model(features, kernel_list, r)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'mkl: {:.4f}'.format(model.mkl.data.item()))
    return acc_test


# Train model
def main_train(optimizer, r, model):
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch, optimizer, r, model)
    print("Optimization Finished with lr = ", optimizer.defaults.get("lr"))
    print("Total time elapsed: {:.4f}s".format((time.time() - t_total)))


if __name__ == "__main__":
    model = CrossDomainGCN(nfeat_f=features.shape[1],
                           nhid=args.hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args.dropout, num_gc=2, append_x=True, fix_weight=False,
                           mkl_type=CrossDomainGCN.MKL_TYPE_HEURISTIC).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    main_train(optimizer, args.multi_kernel_r, model)


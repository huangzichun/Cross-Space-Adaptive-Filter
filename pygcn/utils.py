import math

import numpy
import numpy as np
import scipy.sparse as sp
import torch
import scipy
from numpy.linalg import inv
import torch.nn.functional as F
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(10086)

# raw code, which results in inconsistent encoding
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


def load_data(path="../data/cora/", dataset="cora", kernel=False, label_num=100, valid_num=500, sparse_k=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    if dataset in ["cornell", "texas", "wisconsin", "chameleon", "squirrel"]:
        import io
        s = io.BytesIO(open("{}{}.content".format(path, dataset), "rb").read().replace(b',', b'\t'))
        idx_features_labels = np.genfromtxt(s, dtype=np.dtype(str))
    else:
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))

    if dataset == "actor":
        # feature dimension=932
        def func(s):
            z = np.zeros(932)
            z[np.array(s, dtype=int)] = 1
            return z
        features = [func(s.split(",")) for s in idx_features_labels[:, 1]]
        tmp = np.zeros((idx_features_labels.shape[0], 932+2))
        tmp[:, 0] = idx_features_labels[:,0]
        tmp[:, -1] = idx_features_labels[:,-1]
        tmp[:, 1:-1] = features
        idx_features_labels = tmp

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = one_hot(idx_features_labels[:, -1])

    label_num = label_num if label_num >= 1 else int(label_num * labels.shape[0])
    valid_num = valid_num if valid_num >= 1 else int(valid_num * labels.shape[0])
    print("training num={}, valid={}, test ={}".format(label_num, valid_num, features.shape[0]-valid_num-label_num))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    K = None
    if kernel:
        pairwise_dists = squareform(pdist(features.todense(), 'euclidean'))  # euclidean
        if sparse_k:
            pairwise_dists = np.apply_along_axis(top_k_values, 1, pairwise_dists)
        pairwise_dists = (pairwise_dists + pairwise_dists.T) / 2.0
        s = np.sum(pairwise_dists) / (pairwise_dists != 0).sum()  # since it contains zero values
        print("mean distance, s=", s)

        # KNN graph based on features, Gaussian Kernel
        testd = features.shape[1] * math.sqrt(s)
        K = (scipy.exp(-pairwise_dists ** 2 / s ** 2) * (pairwise_dists != 0))

        # adj = sparse.csr_matrix(np.linalg.inv((sp.eye(adj.shape[0]) - 0.1 * adj).todense()))  # Katz kernel
        # K = np.eye(K.shape[0]) + 0.01 * np.linalg.inv((sp.eye(K.shape[0]) + 0.01 * K))
        # K = np.dot(K, np.linalg.inv((sp.eye(K.shape[0]) * 0.01 + K)))
        K = sparse.csr_matrix(K)
        K = normalize(K + sp.eye(K.shape[0]))
        # adj = K

    features = normalize(features)

    idx_train = range(label_num)
    idx_val = range(label_num, label_num + valid_num)
    idx_test = range(label_num + valid_num, features.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if K is not None:
        K = sparse_mx_to_torch_sparse_tensor(K)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, K


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_kernel(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.diagonal()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def top_k_values(array, topk=20):
    indexes = array.argsort()[:topk]
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    array[list(B.difference(A))]=0
    return array


def get_kernel(adj, type="default", param_dict=None):
    I = torch.eye(adj.shape[0])
    # One-step Random Walk
    if type == "default":
        return adj.to_dense()
    elif type == "ridge":  # KRR
        return
    elif type == "laplacian":  # LP
        if param_dict is None or param_dict.get() is None:
            param_dict = {"a": 1}
        L = I - adj
        L = I + param_dict.get("a") * L
        return torch.linalg.solve(L, I)

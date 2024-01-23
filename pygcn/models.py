import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import numpy as np
import random
from pygcn.utils import normalize


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(10086)


class CrossDomainGCN(nn.Module):
    # Multiple Kernel Learning Algorithms, M Gönen · 2011
    # 1. Fixed Rules
    # Cristianini and Shawe-Taylor, 2000, An Introduction to Support Vector Machines and Other Kernel-Based Learning Methods
    MKL_TYPE_SUMMATION = 1
    MKL_TYPE_MULTIPLICATION = 2
    # 2. Heuristic Approaches
    # Methods for the combination of kernel matrices within a support vector framework
    MKL_TYPE_HEURISTIC = 3
    # 3. Similarity Optimizing Linear Approaches with Arbitrary Kernel Weights
    MKL_TYPE_Linear = 4
    # 4.SM
    MKL_TYPE_SM = 5

    def __init__(self, nfeat_f, nhid, nclass, dropout, num_gc, append_x=True, fix_weight=True, mkl_type=MKL_TYPE_SUMMATION
                 , Z=None, append_z=False, Y=None):
        super(CrossDomainGCN, self).__init__()
        self.num_gcs = num_gc
        self.append_x = append_x
        self.nhid_mid = (nhid + nfeat_f) if append_x else nhid

        # network domain
        self.gc1 = GraphConvolution(nfeat_f, nhid)
        self.gcs = nn.ModuleList()
        for i in range(max(0, self.num_gcs - 2)):
            self.gcs.append(GraphConvolution(self.nhid_mid, nhid))
        self.gc2 = GraphConvolution(self.nhid_mid, nclass)

        # classifier
        self.mlp = nn.LogSoftmax() if not append_z else nn.Sequential(nn.Linear(2 * nclass, nclass), nn.LogSoftmax())
        self.mkl = nn.Parameter(torch.ones(1, 1) * 0.5) if not fix_weight else (torch.ones(1, 1) * 0.5)

        self.dropout = dropout
        self.mkl_type = mkl_type

        self.Z = Z
        self.Y = Y
        self.append_z =append_z

    def get_multi_kernels_from_list(self, kernel_list, r):
        assert len(kernel_list) > 0, "give me a list"
        if len(kernel_list) == 1:
            return kernel_list[0]
        else:
            adj, fadj = kernel_list[:2]
            k = self.get_multi_kernels(adj, fadj, r)
            if len(kernel_list) > 2:
                for ki in kernel_list[2:]:
                    k = self.get_multi_kernels(k, ki, r)
        return k

    def get_multi_kernels(self, adj, fadj, r):
        # unit
        if self.mkl_type == CrossDomainGCN.MKL_TYPE_SUMMATION:
            k = adj + fadj
        elif self.mkl_type == CrossDomainGCN.MKL_TYPE_MULTIPLICATION:
            k = adj * fadj
            k = normalize(k)
            k = torch.FloatTensor(k)
        elif self.mkl_type == CrossDomainGCN.MKL_TYPE_HEURISTIC:
            # r = 0.5    # 0.5 for left
            k = (adj + fadj) / 2.0 + r * (adj - fadj).matmul(adj - fadj)
            # k = normalize(k)
            # k = torch.FloatTensor(k)
        elif self.mkl_type == CrossDomainGCN.MKL_TYPE_SM:
            assert self.Z is not None, "Z is required"
            assert self.Y is not None, "Y is required"
            Y = torch.cat((self.Y, self.Z.argmax(1)), 0).unsqueeze(1)
            Y = Y == Y.t()
            # k = (adj + fadj) / 2.0 + r * (Y.matmul((adj - fadj).matmul(adj - fadj))).matmul(Y)
            k = (adj + fadj) / 2.0 + r * torch.abs(adj - fadj) * Y + 10 * torch.eye(Y.shape[0]).cuda()  # AV
        else:
            k = (adj * self.mkl.clamp(0, 1)) + (fadj * (1 - self.mkl.clamp(0, 1)))
        return k

    def forward(self, x_, adj_list, r):
        k = self.get_multi_kernels_from_list(adj_list, r)

        x = F.relu(self.gc1(x_, k))
        x = F.dropout(x, self.dropout, training=self.training)
        hx = torch.cat((x, x_), 1) if self.append_x else x
        for i in range(max(0, self.num_gcs - 2)):
            x = F.relu(self.gcs[i](hx, k))
            x = F.dropout(x, self.dropout, training=self.training)
            hx = torch.cat((x, x_), 1) if self.append_x else x

        if self.num_gcs > 1:
            x = self.gc2(hx, k)
            hx = x # torch.cat((x, x_), 1) if self.append_x else x

        if self.append_z:
            hx = torch.cat((hx, self.Z), 1)

        y = self.mlp(hx)
        return y



import numpy as np
from numpy import int64
from sklearn import preprocessing


def SC(data):
    # number of samples * number of features
    N = data.shape[0]

    # normalize software metrics
    data = preprocessing.scale(data)

    # construct weight graph
    W = np.dot(data, data.T)
    W[W <= 0] = 0
    W = W - np.diag(np.diag(W))

    Dnsqrt = np.diag(1 / np.sqrt(np.sum(W, axis=1) + np.finfo(float).eps))
    I = np.eye(N)
    # Lsym = I - np.dot(np.dot(Dnsqrt, W), Dnsqrt)
    Lsym = I - Dnsqrt @ W @ Dnsqrt
    Lsym = 0.5 * (Lsym + Lsym.T)

    # perform the eigen decomposition, svd descending by default
    # U, S, V = np.linalg.svd(Lsym)
    # S_reordering = np.argsort(S)
    # # S = S[S_reordering]
    # U = U[:, S_reordering]
    #
    # v = np.dot(Dnsqrt, U[:, 1])
    # v = v / np.sqrt(np.sum(v ** 2, axis=0))

    # eigh ascending by default
    val, U = np.linalg.eigh(Lsym)
    v = Dnsqrt @ U[:, 1]
    v = v / np.sqrt(np.sum(v ** 2, axis=0))

    # divide the data set into two clusters
    preLabel = (v > 0)

    # label the defective and non-defective clusters
    rs = np.sum(data, axis=1)

    if np.mean(rs[v > 0]) < np.mean(rs[v < 0]):
        preLabel = (v < 0)

    clus_label = preLabel.astype(int64)

    return clus_label

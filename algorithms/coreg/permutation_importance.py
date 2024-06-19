import copy
import numpy as np
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2


def perm_importance(model, X, y):
    n = len(y)
    acc = []
    acc2 = []
    for i in range(len(X)):
        x = X[i]
        for j in range(x.shape[1]):
            X1 = copy.deepcopy(X)
            temp = x[:, j]
            temp = np.random.permutation(temp)
            X1[i][:, j] = temp
            clus_label = model.fit_predict(X1)

            predict_y = [labelCluster(x, clus_label) for x in X1]
            predict_y = np.mean(predict_y, axis=0)

            predict_y2 = labelCluster_v2(clus_label)

            temp_acc = sum(predict_y == y)/n
            acc.append(temp_acc)

            temp_acc2 = sum(predict_y2 == y)/n
            acc2.append(temp_acc2)

    return acc, acc2


def perm_importance2(model, X, y):
    n = len(y)
    acc = []

    for i in range(X.shape[1]):
        X1 = copy.deepcopy(X)
        temp = X[:, i]
        temp = np.random.permutation(temp)
        X1[:, i] = temp
        clus_label = model.fit_predict(X1)

        predict_y = labelCluster(clus_label)

        temp_acc = sum(predict_y == y)/n
        acc.append(temp_acc)

    return acc



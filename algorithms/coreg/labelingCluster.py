import pandas as pd
from sklearn import preprocessing


# labeling cluster according to the metric values of modules
def labelCluster(fea, clus_label):
    fea = preprocessing.scale(fea)
    fea = pd.DataFrame(fea)
    fea['clus_label'] = clus_label
    fea1 = fea[fea['clus_label'] == 0].iloc[:, :-1]
    fea2 = fea[fea['clus_label'] == 1].iloc[:, :-1]
    mean_fea1 = fea1.mean().mean()
    mean_fea2 = fea2.mean().mean()
    if mean_fea1 > mean_fea2:
        for i, label in enumerate(clus_label):
            if clus_label[i] == 0:
                clus_label[i] = 1
            else:
                clus_label[i] = 0

    return clus_label


# labeling cluster according to the number of modules
def labelCluster_v2(clus_label):
    n1 = clus_label[clus_label == 1].size # defective
    n2 = clus_label[clus_label == 0].size # nondefective

    # cluster with the smaller number of modules is labeled defective
    if n1 > n2:
        for i, label in enumerate(clus_label):
            if clus_label[i] == 0:
                clus_label[i] = 1
            else:
                clus_label[i] = 0
    return clus_label

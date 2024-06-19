import csv
import os
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.coreg.permutation_importance import perm_importance
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2
from utilities.AutoSpearman import AutoSpearman
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, LOC, n_class, v_lambda, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    # test_LOC = LOC[test_idx]

    test_data = [preprocessing.scale(x) for x in test_data]

    m_spectral = MultiviewCoRegSpectralClustering(n_clusters=n_class,
                                                  v_lambda=v_lambda)
    clus_label = m_spectral.fit_predict(test_data)

    # labeling clustering
    predict_y = [labelCluster(x, clus_label) for x in test_data]
    predict_y = np.mean(predict_y, axis=0)

    predict_y2 = labelCluster_v2(clus_label)

    # explanation via permutation feature importance
    acc = sum(predict_y == test_label) / len(test_label)
    acc2 = sum(predict_y2 == test_label) / len(test_label)

    acc_list, acc_list2 = perm_importance(m_spectral, test_data, test_label)
    perm_score = acc - acc_list
    perm_score2 = acc2 - acc_list2

    # save to csv
    fres = create_dir(save_path + model_name + '_permfeaimp1')
    save_results(fres + project_name, perm_score)

    fres = create_dir(save_path + model_name + '_permfeaimp2')
    save_results(fres + project_name, perm_score2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = '../result/'
    model_name = 'MUSDP_perm'
    Reps = 100
    n_class = 2
    v_lambda = 1

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)

    for i in range(pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]

        # construct multi-view data
        X = [data.iloc[:, 0:54], data.iloc[:, 54:59], data.iloc[:, 59:65]]
        y = data.iloc[:, -1]
        LOC = data['CountLineCode']

        # feature selection -> correlation analysis and redundancy analysis
        X = [AutoSpearman(x) for x in X]
        # save to csv
        feas = []
        for x in X:
            feas.extend(x.columns.values)
        fres = create_dir(save_path)
        with open(fres + 'selected_features.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feas)

        X.append(y)

        for loop in range(Reps):
            run_(X, LOC, n_class, v_lambda, save_path, project_name, model_name, loop)

    print('done!')

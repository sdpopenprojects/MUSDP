import csv
import os
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.labelingCluster import labelCluster, labelCluster_v2
from utilities import performanceMeasure, rankMeasure
from utilities.AutoSpearman import AutoSpearman
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, LOC, n_class, v_lambda, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    test_LOC = LOC[test_idx]

    test_data = [preprocessing.scale(x) for x in test_data]

    m_spectral = MultiviewCoRegSpectralClustering(n_clusters=n_class,
                                                  v_lambda=v_lambda,
                                                  max_iter=10)
    clus_label = m_spectral.fit_predict(test_data)

    # labeling clustering
    predict_y = [labelCluster(x, clus_label) for x in test_data]
    predict_y = np.mean(predict_y, axis=0)

    predict_y2 = labelCluster_v2(clus_label)

    # # calculate non-effort-aware classification measure
    AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(test_label, predict_y)
    # # calculate cost-effectiveness measures
    Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(predict_y, test_LOC,
                                                                                          test_label)

    measure = [AUC, G_mean, precision, recall, pf, F1, MCC, Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]

    fres = create_dir(save_path + model_name + '_label1')
    save_results(fres + project_name, measure)

    # # calculate non-effort-aware classification measure
    AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(test_label, predict_y2)
    # # calculate cost-effectiveness measures
    Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(predict_y2, test_LOC,
                                                                                          test_label)

    measure2 = [AUC, G_mean, precision, recall, pf, F1, MCC, Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]

    fres = create_dir(save_path + model_name + '_label2')
    save_results(fres + project_name, measure2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = '../result/'
    model_name = 'MUSDP'
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

        X.append(y)

        for loop in range(Reps):
            run_(X, LOC, n_class, v_lambda, save_path, project_name, model_name, loop)

    print('done!')

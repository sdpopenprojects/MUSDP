import os
import warnings
import pandas as pd

from sklearn import preprocessing
from algorithms.SC import SC
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, _, _ = outofsample_bootstrap(X, randseed)
    LOC = test_data['CountLineCode']

    test_data = preprocessing.scale(test_data)

    # SC
    clus_label = SC(test_data)
    predict_y = clus_label

    # # calculate non-effort-aware classification measure
    AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(test_label, predict_y)
    # # calculate cost-effectiveness measures
    Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(predict_y, LOC, test_label)

    measure = [AUC, G_mean, precision, recall, pf, F1, MCC, Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]

    fres = create_dir(save_path+model_name)
    save_results(fres + project_name, measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = '../result/'
    model_name = 'SC'
    Reps = 100

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    for i in range(pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]

        for loop in range(Reps):
            run_(data, save_path, project_name, model_name, loop)

    print('done!')

import os
import numpy as np
import pandas as pd
from algorithms.ManualDownUp import manualDown, manualUp
from utilities import rankMeasure, performanceMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


if __name__ == '__main__':

    save_path = "../result/"
    model_names = ['ManualDown', 'ManualUp']
    Reps = 100

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    for model_name in model_names:
        for i in range(pro_num):
            project_name = project_names[i]
            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            project_name = project_name[:-4]

            for loop in range(0, Reps):
                print(project_name + ': -> ' + model_name + ' ' + str(loop + 1) + ' round Start!')

                train_data, train_label, test_data, test_label, _, _ = outofsample_bootstrap(data, loop)
                LOC = np.array(test_data['CountLineCode'])

                if model_name == 'ManualDown':
                    # ManualDown assumes a module with larger LOC (lines of code) is more defect-prone ascending by default
                    predict_label = manualDown(LOC)
                    score = LOC
                else:
                    # ManualUp assumes a module with smaller LOC is more defect-prone
                    if 0 in LOC:
                        LOC = LOC+1
                    predict_label = manualUp(LOC)
                    score = 1/LOC

                # # calculate non-effort-aware classification measure
                AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(test_label, predict_label)
                # # calculate cost-effectiveness measures
                Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(score, LOC, test_label, 1)

                measure = [AUC, G_mean, precision, recall, pf, F1, MCC, Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]

                fres = create_dir(save_path + model_name)
                save_results(fres + project_name, measure)

    print('done!')
import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


if __name__ == '__main__':

    path = r"../result/"
    model_name = 'ManualUp'

    files = [f for f in sorted(os.listdir(path + model_name + '/')) if f.endswith('.csv')]
    files_num = len(files)

    files_list = []
    results = []
    median_results = []
    for file in files:
        file_path = os.path.join(path+model_name+'/', file)
        file_name = file[:-4]
        files_list.append(file_name)

        df = pd.read_csv(file_path, header=None)
        results.append(np.array(df))

        res = np.median(df, axis=0)

        median_results.append(res)

    # save to csv file
    results = DataFrame(np.vstack(results))
    results.to_csv(path+'all_result_'+model_name+'.csv', index=None, header=None)
    median_all = np.median(results, axis=0)

    median_results.append(median_all)
    data = DataFrame(median_results)

    files_list.append('Median')
    data.index = files_list

    measurename = ['AUC', 'G_mean', 'precision', 'recall', 'pf', 'F1', 'MCC',
                   'Popt', 'recall_effort', 'precision_effort', 'F1_effort', 'PMI', 'IFA']
    data.columns = measurename

    data.to_csv(path+'result_'+model_name+'.csv')

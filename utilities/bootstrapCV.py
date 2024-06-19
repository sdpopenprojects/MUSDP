import numpy as np
import pandas
from sklearn.utils import resample  # for Bootstrap sampling


# out of sample bootstrap cross validation
def outofsample_bootstrap(data, randseed=42):
    # sampling with replacement, whichever is not used in training data will be used in test data
    if isinstance(data, pandas.DataFrame):
        indexs = list(data.index)
        train_idx = resample(indexs, n_samples=len(indexs), random_state=randseed)

        # picking rest of the data not considered in training data
        test_idx = list(set(indexs) - set(train_idx))

        train_data = data.iloc[train_idx, :-1]
        train_label = data.iloc[train_idx, -1]

        test_data = data.iloc[test_idx, :-1]
        test_label = data.iloc[test_idx, -1]
    else:
        indexs = list(data[0].index)
        train_idx = resample(indexs, n_samples=len(indexs), random_state=randseed)

        # picking rest of the data not considered in training data
        test_idx = list(set(indexs) - set(train_idx))

        train_data = [x.iloc[train_idx, :] for x in data[:-1]]
        train_label = data[-1][train_idx]

        test_data = [x.iloc[test_idx, :] for x in data[:-1]]
        test_label = data[-1][test_idx]

    train_label[train_label > 1] = 1
    test_label[test_label > 1] = 1

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label, train_idx, test_idx

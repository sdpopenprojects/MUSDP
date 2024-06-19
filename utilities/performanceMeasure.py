import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


# # traditional non-effort-aware performance measures
def get_measure(y_true, y_pred):
    # y_pred[y_pred >= 0.5] = 1
    # y_pred[y_pred < 0.5] = 0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if fp + tn != 0:
        pf = fp / (fp + tn)
    else:
        pf = 0

    if recall + precision != 0:
        F1 = 2 * recall * precision / (recall + precision)
    else:
        F1 = 0

    AUC = roc_auc_score(y_true, y_pred)

    temp = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
    if temp != 0:
        MCC = (tp * tn - fn * fp) / np.sqrt(temp)
    else:
        MCC = 0

    g_mean = np.sqrt(recall * (1 - pf))

    return [AUC, g_mean, precision, recall, pf, F1, MCC]

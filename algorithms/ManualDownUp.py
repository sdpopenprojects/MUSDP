import numpy as np
import math


def manualDown(LOC):
    # ManualDown assumes a module with larger LOC (lines of code) is more defect-prone ascending by default
    n = math.floor(len(LOC) / 2)

    idxDown = np.argsort(-LOC, axis=0)
    sort_LOC = LOC[idxDown].flatten()
    sort_LOC[:n] = 1
    sort_LOC[n:] = 0

    ori_idx = np.argsort(idxDown, axis=0)
    pre_label = sort_LOC[ori_idx].flatten()

    return pre_label


def manualUp(LOC):
    # ManualUp assumes a module with smaller LOC is more defect-prone
    n = math.floor(len(LOC) / 2)

    idxUp = np.argsort(LOC, axis=0)
    sort_LOC = LOC[idxUp].flatten()
    sort_LOC[:n] = 1
    sort_LOC[n:] = 0

    ori_idx = np.argsort(idxUp, axis=0)
    pre_label = sort_LOC[ori_idx].flatten()

    return pre_label

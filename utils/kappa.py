import numpy as np


def kappa(confusion_matrix):
    rows = np.sum(confusion_matrix, axis=0)
    cols = np.sum(confusion_matrix, axis=1)
    total_sum = sum(cols)
    p_e = np.dot(rows, cols) / float(total_sum ** 2)
    p_o = np.trace(confusion_matrix) / float(total_sum)
    return (p_o - p_e) / (1 - p_e)
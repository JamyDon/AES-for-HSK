from sklearn.metrics import cohen_kappa_score


def kappa(y_true, y_pred):
    kappa_value = cohen_kappa_score(y_true, y_pred)
    return kappa_value


def confusion_matrix(y_a, y_b, score_num):
    conf_mat = [[0 for _ in range(score_num)] for __ in range(score_num)]

    for a, b in zip(y_a, y_b):
        conf_mat[a][b] += 1

    return conf_mat


def histogram(y, score_num):
    hist = [0 for _ in range(score_num)]

    for a in y:
        hist[a] += 1

    return hist


def quadratic_weighted_kappa(y_a, y_b, score_num):
    y_size = float(len(y_a))

    conf_mat = confusion_matrix(y_a, y_b, score_num)

    hist_a = histogram(y_a, score_num)
    hist_b = histogram(y_b, score_num)

    WO = 0
    WE = 0

    for i in range(score_num):
        for j in range(score_num):
            W_ij = pow(i - j, 2.0) / pow(score_num - 1, 2.0)
            O_ij = conf_mat[i][j]
            E_ij = hist_a[i] * hist_b[j] / y_size
            WO += W_ij * O_ij / score_num
            WE += W_ij * E_ij / score_num

    return 1.0 - WO / WE


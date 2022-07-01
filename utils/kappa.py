from sklearn.metrics import cohen_kappa_score


def kappa(y_true, y_pred):
    kappa_value = cohen_kappa_score(y_true, y_pred)
    return kappa_value

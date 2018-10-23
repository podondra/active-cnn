from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale


def scale(X):
    return minmax_scale(X, feature_range=(-1, 1), axis=1, copy=False)


def balance(X, y):
    return SMOTE().fit_resample(X, y)

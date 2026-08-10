import sklearn
from sklearn.model_selection import train_test_split

SEED = 1


def fit(X, y):
    return train_test_split(X, y, random_state=SEED)

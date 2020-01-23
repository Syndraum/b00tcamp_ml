import math
import numpy as np


class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.thetas = []

    def sigmoid_(x):
        if isinstance(x, list):
            return list(map(sigmoid_, x))
        elif isinstance(x, np.ndarray):
            return np.array(list(map(sigmoid_, x)))
        else:
            return 1.0 / (1.0 + math.exp(-x))

    def vec_log_gradient_(x, y_true, y_pred):
        return np.dot(y_pred - y_true, x)

    def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
        def func(y_true, y_pred):
            return (
                y_true * math.log(y_pred + eps)
                + (1 - y_true) * math.log(1 - y_pred + eps))
        mysum = 0.0
        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
            if len(y_pred) != m or len(y_true) != m:
                return None
            for i in range(m):
                mysum += func(y_true[i], y_pred[i])
            return -mysum / m
        else:
            return -func(y_true, y_pred)

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_train):
        if isinstance(x_train, np.ndarray):
            return np.array(list(map(sigmoid_, x_train)))
        else:
            return 1.0 / (1.0 + math.exp(-x_train))

    def score(self, x_train, y_train):
        pass

import numpy as np
from sigmoid import sigmoid_
import math


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


x = 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1
print(vec_log_loss_(y_true, y_pred, m))

x = np.array([1, 2, 3, 4])
y_true = 0
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x, theta))
m = 1
print(vec_log_loss_(y_true, y_pred, m))    # 10.100041078687479

x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(vec_log_loss_(y_true, y_pred, m))
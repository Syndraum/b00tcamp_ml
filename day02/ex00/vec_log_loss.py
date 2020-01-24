import numpy as np
from sigmoid import sigmoid_
import math


def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    return (-(np.dot(y_true, np.log(y_pred + eps)) + np.dot((1 - y_true), np.log(1 - y_pred + eps)))/m)


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
print(y_true.shape, y_pred.shape)
print(vec_log_loss_(y_true, y_pred, m))

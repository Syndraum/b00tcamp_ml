from sigmoid import sigmoid_
import numpy as np


def vec_log_gradient_(x, y_true, y_pred):
    return np.dot(y_pred - y_true, x)


x = np.array([1, 4.2])
y_true = 1
theta = np.array([0.5, -0.5])
y_pred = sigmoid_(np.dot(x, theta))
print(vec_log_gradient_(x, y_pred, y_true))

x = np.array([1, -0.5, 2.3, -1.5, 3.2])
y_true = 0
theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
y_pred = sigmoid_(np.dot(x, theta))
print(vec_log_gradient_(x, y_true, y_pred))


x_new = np.arange(2, 14).reshape((3, 4))
x_new = np.insert(x_new, 0, 1, axis=1)
# first column of x_new are now intercept values initialized to 1
y_true = np.array([1, 0, 1])
theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
y_pred = sigmoid_(np.dot(x_new, theta))
print(vec_log_gradient_(x_new, y_true, y_pred))

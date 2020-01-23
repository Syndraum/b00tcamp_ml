from sigmoid import sigmoid_
import math


def log_loss_(y_true, y_pred, m, eps=1e-15):
    def func(y_true, y_pred):
        return y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps)
    mysum = 0.0
    if isinstance(y_true, list) and isinstance(y_pred, list):
        if len(y_pred) != m or len(y_true) != m:
            return None
        for i in range(m):
            mysum += func(y_true[i], y_pred[i])
        return -mysum / m
    else:
        return -func(y_true, y_pred)


# x = 4
# y_true = 1
# theta = 0.5
# y_pred = sigmoid_(x * theta)
# m = 1
# print(log_loss_(y_true, y_pred, m))

# x = [1, 2, 3, 4]
# y_true = 0
# theta = [-1.5, 2.3, 1.4, 0.7]
# x_dot_theta = sum([a*b for a, b in zip(x, theta)])
# y_pred = sigmoid_(x_dot_theta)
# m = 1
# print(log_loss_(y_true, y_pred, m))

# x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# y_true = [1, 0, 1]
# theta = [-1.5, 2.3, 1.4, 0.7]
# x_dot_theta = []
# for i in range(len(x_new)):
#     my_sum = 0
#     for j in range(len(x_new[i])):
#         my_sum += x_new[i][j] * theta[j]
#     x_dot_theta.append(my_sum)
# y_pred = sigmoid_(x_dot_theta)
# m = len(y_true)
# print(log_loss_(y_true, y_pred, m))

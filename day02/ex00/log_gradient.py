from sigmoid import sigmoid_
from operator import add


def log_gradient_(x, y_true, y_pred):
    mylist = []
    if all(not isinstance(elmt, list) for elmt in x):
        for elmt in x:
            mylist.append((y_pred - y_true) * elmt)
        return mylist
    else:
        mylist = [0] * len(x[0])
        for i in range(len(x)):
            mylist = list(map(add, mylist, log_gradient_(
                x[i], y_true[i], y_pred[i])))
        return mylist


x = [1, 4.2]
y_true = 1
theta = [0.5, -0.5]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
print(log_gradient_(x, y_pred, y_true))

x = [1, -0.5, 2.3, -1.5, 3.2]
y_true = 0
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
x_dot_theta = sum([a*b for a, b in zip(x, theta)])
y_pred = sigmoid_(x_dot_theta)
print(log_gradient_(x, y_true, y_pred))

x_new = [[1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [1, 10, 11, 12, 13]]
# first column of x_new are intercept values initialized to 1
y_true = [1, 0, 1]
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
x_new_dot_theta = []
for i in range(len(x_new)):
    my_sum = 0
    for j in range(len(x_new[i])):
        my_sum += x_new[i][j] * theta[j]
    x_new_dot_theta.append(my_sum)
y_pred = sigmoid_(x_new_dot_theta)
print(log_gradient_(x_new, y_true, y_pred))

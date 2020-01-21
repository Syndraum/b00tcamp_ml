import numpy as np


def predict_(theta, X):
    n_x = np.insert(X, 0, 1, axis=1)
    return n_x.dot(theta)


def cost_elem_(theta, X, Y):
    output = np.array([])
    predict = predict_(theta, X)
    for i in range(X.shape[0]):
        output = np.append(output, ((predict[i] - Y[i])**2/(2*X.shape[0])))
    return output

def cost_(theta, X, Y):
    return sum(cost_elem_(theta, X, Y))


def vec_gradient(x, y, theta):
    return (x.dot(theta) - y).dot(x)/x.shape[0]


def gradient(x, y, theta):
    gradient = np.array([])
    for t in range(x.shape[1]):
        mysum = 0.0
        for i in range(x.shape[0]):
            mysum += ((theta.dot(x[i]) - y[i]) * x[i][t])
        gradient = np.append(gradient, mysum/x.shape[0])
    return gradient


def fit_(theta, X, Y, alpha, n_cycle):
    n_x = np.insert(X, 0, 1, axis=1)
    for n in range(n_cycle):
        theta -= alpha*(np.dot(n_x.transpose(), np.dot(n_x, theta) - Y)) / (2 * X.shape[0])
    return theta


X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
# print(predict_(theta1, X1))
# print(cost_elem_(theta1, X1, Y1))
# print(cost_(theta1, X1, Y1))
print(fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000))

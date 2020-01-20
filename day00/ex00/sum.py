import numpy as np


def sum_(x, f):
    mysum = 0.0
    try:
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        for value in x:
            mysum += float(f(value))
        return mysum
    except (TypeError) as e:
        return None


def mean(x):
    try:
        return sum_(x, lambda x: x) / len(x)
    except (TypeError) as e:
        return None


def variance(x):
    m = mean(x)
    lst = np.array([])
    for value in x:
        lst = np.append(lst, (value - m)**2)
    return mean(lst)


def std(x):
    try:
        return variance(x)**0.5
    except (TypeError) as e:
        return None


def dot(x, y):
    mysum = 0.0
    try:
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        if not isinstance(y, np.ndarray) or y.size == 0:
            return None
        if len(x) != len(y):
            return None
    except (TypeError) as e:
        return None
    else:
        for i in range(len(x)):
            mysum += float(x[i] * y[i])
        return mysum


def mat_vec_prod(x, y):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if len(x.shape) != 2 or x.shape[1] != len(y):
        print("XShAPE")
        return None
    if len(y.shape) != 2 or y.shape[1] != 1:
        print("YShAPE")
        return None
    lst = np.array([])
    for vector in x:
        lst = np.append(lst, dot(vector, y))
    return np.expand_dims(lst, axis=1).astype(int)


def defmat_mat_prod(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if len(x.shape) != 2 or len(y.shape) != 2:
        return None
    if x.shape[1] != y.shape[0]:
        return None
    mat = np.array([])
    for vector in np.swapaxes(y, 0, 1):
        mat = np.append(mat, mat_vec_prod(x, vector.reshape(x.shape[1], 1)))
    return np.swapaxes(mat.reshape(x.shape[0], y.shape[1]), 0, 1).astype(int)


def mse(y, y_hat):
    if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if y_hat.shape[0] != y.shape[0]:
        return None
    mysum = 0.0
    for i in range(len(y)):
        mysum += (y_hat[i] - y[i])**2
    return mysum/len(y)


def vec_mse(y, y_hat):
    if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if y_hat.shape[0] != y.shape[0]:
        return None
    tmp = y_hat - y
    return dot(tmp, tmp)/len(y)


def linear_mse(x, y, theta):
    mysum = 0.0
    for i in range(len(y)):
        mysum += (dot(theta, x[i]) - y[i])**2
    return mysum/len(y)


def vec_linear_mse(x, y, theta):
    tmp = (x.dot(theta) - y)
    return dot(tmp, tmp)/len(y)


Z = np.array([0])
X = np.array([0, 15, -9, 7, 12, 3, -21])
X1 = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
W = np.array([
    [-8, 8, -6, 14, 14, -9, -4],
    [2, -11, -2, -11, 14, -2, 14],
    [-13, -2, -5, 3, -8, -4, 13],
    [2, 13, -14, -15, -14, -15, 13],
    [2, -1, 12, 3, -7, -3, -6]])
Z = np.array([
    [-6, -1, -8, 7, -8],
    [7, 4, 0, -10, -10],
    [7, -13, 2, 2, -11],
    [3, 14, 7, 7, -4],
    [-1, -3, -8, -4, -14],
    [9, -14, 9, 12, -7],
    [-9, -4, -10, -3, 6]])
# print(sum_(X, lambda x: x**2))
# print(mean(X**2))
# print(variance(X))
# print(std(Y))
# print(dot(Z, Y))
# print(mat_vec_prod(W, X1))
# print(defmat_mat_prod(W, Z))
# print(defmat_mat_prod(Z, W))
# print(W.dot(X))
# print(vec_mse(X, Y))
# print(vec_mse(X, X))

X = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
W = np.array([0,0,0])

print(vec_linear_mse(X, Y, Z))
print(vec_linear_mse(X, Y, W))

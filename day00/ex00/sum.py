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


Z = np.array([0])
X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7, 1))
X1 = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7, 1))
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

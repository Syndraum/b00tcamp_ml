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
    return sum_(x, lambda x: x) / len(x)


X1 = np.array([])
X = np.array([0, 15, -9, 7, 12, 3, -21])
# print(sum_(X, lambda x: x**2))
print(mean(X**2))

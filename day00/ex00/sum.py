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


X1 = np.array([])
X = np.array([0, 15, -9, 7, 12, 3, -21])
# print(sum_(X, lambda x: x**2))
# print(mean(X**2))
print(variance(X/2))

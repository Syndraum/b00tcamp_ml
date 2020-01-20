import numpy as np


def sum_(x, f):
    mysum = 0.0
    try:
        if x.size == 0:
            return None
        for value in x:
            mysum += f(value)
        return mysum
    except (TypeError, AttributeError):
        return None


X = np.array([])
print(sum_(3, lambda x: x+2))

import numpy as np


def regularization(theta, lambda_):
    mysum = 0.0
    for n in theta:
        mysum += n**2
    return mysum * lambda_


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(regularization(X, 0.3))
print(regularization(X, 0.01))
print(regularization(X, 0))
import numpy as np


class MyLinearRegression():
    def __init__(self, theta):
        self.theta = theta

    def predict_(self, X):
        Xpp = np.insert(X, 0, 1, axis=1)
        return Xpp.dot(self.theta)

    def cost_elem_(self, X, Y):
        output = np.array([])
        predict = self.predict_(X)
        for i in range(X.shape[0]):
            output = np.append(output, ((predict[i] - Y[i])**2/(2*X.shape[0])))
        return output

    def cost_(self, X, Y):
        return sum(self.cost_elem_(X, Y))

    def fit_(self, X, Y, alpha, n_cycle):
        Xpp = np.insert(X, 0, 1, axis=1)
        for n in range(n_cycle):
            self.theta -= alpha*(np.dot(Xpp.transpose(), np.dot(Xpp, self.theta) - Y)) / (2 * X.shape[0])


# X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,144.]])
# Y = np.array([[23.], [48.], [218.]])
# mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
# print(mylr.predict_(X))
# print(mylr.cost_elem_(X,Y))
# print(mylr.cost_(X,Y))
# mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
# print(mylr.theta)
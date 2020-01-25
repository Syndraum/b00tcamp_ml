import math
import numpy as np
import pandas as pd


class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.thetas = np.zeros((1, 1))

    def vec_log_gradient_(self, x, y_true, y_pred):
        return np.dot(y_pred - y_true, x)

    def sigmoid_(self, x):
        return (1.0 / (1.0 + np.exp(-x)))

    def vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
        return (
            -(
                np.dot(y_true, np.log(y_pred + eps))
                + np.dot((1 - y_true), np.log(1 - y_pred + eps)))/m)

    def fit(self, x_train, y_train):
        self.thetas = np.ones((x_train.shape[1], 1))
        for n in range(self.max_iter + 1):
            predict = self.predict(x_train)
            sig = self.sigmoid_(np.dot(x_train, self.thetas))
            if self.verbose is True and n % (self.max_iter/10) == 0:
                loss = self.vec_log_loss_(y_train.reshape(y_train.shape[0]), sig.reshape(predict.shape[0]), len(y_train))
                print(f"epoch  {n}\t: loss {loss}")
            self.thetas -= self.alpha*(np.dot(x_train.T, sig - y_train)) / (x_train.shape[0])

    def predict(self, x_train):
        return np.around(self.sigmoid_(np.dot(x_train, self.thetas)))

    def score(self, x_train, y_train):
        return (self.predict(x_train) == y_train).mean()


df_train = pd.read_csv(
    '../dataset/train_dataset_clean.csv',
    delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv(
    '../dataset/test_dataset_clean.csv',
    delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]
model = LogisticRegressionBatchGd(
    alpha=0.01, max_iter=1500, verbose=True, learning_rate='constant')
Y = y_train.to_numpy()
y_train = Y.reshape(Y.shape[0], 1)
Y = y_test.to_numpy()
y_test = Y.reshape(Y.shape[0], 1)
model.fit(x_train, y_train)
print(f'Score on train dataset : {model.score(x_train, y_train)}')
y_pred = model.predict(x_test)
# print(y_pred)
# print(y_test)
print(f'Score on test dataset : {(y_pred == y_test).mean()}')

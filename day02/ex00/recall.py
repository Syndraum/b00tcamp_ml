import numpy as np
from sklearn.metrics import recall_score


def recall_score_(y_true, y_pred, pos_label=1):
    return (np.logical_and(y_pred == y_true, y_pred == pos_label).sum()
    / (np.logical_and(y_pred == y_true, y_pred == pos_label).sum()
    + np.logical_and(y_pred != y_true, y_pred != pos_label).sum()))


# y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
# y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
# print(recall_score_(y_true, y_pred))
# print(recall_score(y_true, y_pred))

# y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog','dog', 'dog'])
# y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet','dog', 'norminet'])
# print(recall_score_(y_true, y_pred, pos_label='dog'))
# print(recall_score(y_true, y_pred, pos_label='dog'))

# print(recall_score_(y_true, y_pred, pos_label='norminet'))
# print(recall_score(y_true, y_pred, pos_label='norminet'))
import numpy as np
from recall import recall_score_
from precision import precision_score_
from sklearn.metrics import f1_score


def f1_score_(y_true, y_pred, pos_label=1):
    precision = precision_score_(y_true, y_pred, pos_label)
    recall = recall_score_(y_true, y_pred, pos_label)
    return (2 * (precision * recall) / (precision + recall))


y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(f1_score_(y_true, y_pred))
print(f1_score(y_true, y_pred))

y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog','dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet','dog', 'norminet'])
print(f1_score_(y_true, y_pred, pos_label='dog'))
print(f1_score(y_true, y_pred, pos_label='dog'))

print(f1_score_(y_true, y_pred, pos_label='norminet'))
print(f1_score(y_true, y_pred, pos_label='norminet'))

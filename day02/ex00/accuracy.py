import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_score_(y_true, y_pred):
    return (y_true == y_pred).mean()

y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(accuracy_score_(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog','dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet','dog', 'norminet'])
print(accuracy_score_(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
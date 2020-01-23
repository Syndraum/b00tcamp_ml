import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
Y = np.array(data[["Sell_price"]])
myLR_ne = MyLR([[1.0], [1.0], [1.0], [1.0]])
myLR_lgd = MyLR([[1.0], [1.0], [1.0], [1.0]])
myLR_lgd.fit_(X, Y, alpha=5e-5, n_cycle=10000)
myLR_ne.normalequation_(X, Y)
print("======== MSE LGD ========")
print(myLR_lgd.linear_mse(X, Y))
print("======== MSE NORMA ========")
print(myLR_ne.linear_mse(X, Y))

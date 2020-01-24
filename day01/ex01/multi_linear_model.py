import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR


data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Xage = np.array(data["Age"]).reshape(-1, 1)
Xthrush = np.array(data["Thrust_power"]).reshape(-1, 1)
Xmeter = np.array(data["Terameters"]).reshape(-1, 1)
Yprice = np.array(data[["Sell_price"]])
# myLR_age = MyLR([[1000.0], [-1.0]])
myLR = MyLR([[1.0], [1.0], [1.0], [1.0]])
myLR_age = MyLR([[1000.0], [-1.0]])
myLR_thrush = MyLR([[1.0], [-1.0]])
myLR_meter = MyLR([[1.0], [-1.0]])
myLR.fit_(X, Yprice, alpha=2e-6, n_cycle=600000)
# myLR_age.fit_(Xage, Yprice, alpha=0.025, n_cycle=100000)
# myLR_thrush.fit_(Xthrush, Yprice, alpha=0.00000007, n_cycle=100000)
# myLR_meter.fit_(Xmeter, Yprice, alpha=0.00017, n_cycle=200000)
# print(myLR_age.theta)

# print(myLR.linear_mse(X, Yprice))
# myLR.fit_(X, Yprice, alpha=1e-4, n_cycle=600000)
# print(myLR.theta)

# myLR_age.fit_(X[:, 0].reshape(-1, 1), Yprice, alpha=2.5e-5, n_cycle=100000)
# RMSE_age = myLR_age.linear_mse(Xage[:, 0].reshape(-1, 1), Yprice)
# print(RMSE_age)
# print(myLR.linear_mse(X, Yprice))


def fig_age():
    age = myLR.predict_(X)
    plt.plot(X[:, 0].reshape(-1, 1), Yprice, 'o', color='#141C95', label='Sell Price')
    plt.plot(X[:, 0].reshape(-1, 1), age, 'o', color='cyan', label='Sell Price')
    plt.grid(True)
    plt.show()


def figage():
    age = myLR_age.predict_(Xage)
    plt.plot(Xage, Yprice, 'o', color='#141C95', label='Sell Price')
    plt.plot(Xage, age, 'o', color='cyan', label='Sell Price')
    plt.grid(True)
    plt.show()


def figthrush():
    thrush = myLR_thrush.predict_(Xthrush)
    plt.plot(Xthrush, Yprice, 'o', color='green', label='Sell Price')
    plt.plot(Xthrush, thrush, 'o', color='#00FF00', label='Sell Price')
    plt.grid(True)
    plt.show()


def figmeter():
    thrush = myLR_meter.predict_(Xmeter)
    plt.plot(Xmeter, Yprice, 'o', color='#A33BEF', label='Sell Price')
    plt.plot(Xmeter, thrush, 'o', color='#EF3BD6', label='Sell Price')
    plt.grid(True)
    plt.show()

# figage()
# figthrush()
# figmeter()
fig_age()

# print(RMSE_age)
